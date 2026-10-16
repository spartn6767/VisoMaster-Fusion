"""Luminance-preserving hair recoloring on top of the face parser's hair class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import kornia.color as kc
import numpy as np
import torch
import torch.nn.functional as F

from app.processors.video_utils.alignment_store import HAIR_PROB_SIZE

HAIR_LABEL = 17
MASK_RAMP_WIDTH = 0.3

CropKey = tuple[float, ...]


@dataclass
class HairMaskCache:
    """Per-face access to cached hair probabilities in the alignments store.

    ``load`` returns the temporally smoothed 0..1 map (or None on a miss) and
    ``save`` stores this frame's uint8 map, both keyed by the makeup crop.
    """

    load: Callable[[CropKey], Optional[np.ndarray]]
    save: Callable[[np.ndarray, CropKey], object]


@torch.no_grad()
def hair_probability(logits: torch.Tensor) -> torch.Tensor:
    """[1,19,H,W] parser logits -> [1,H,W] hair probability."""
    return torch.softmax(logits.float(), dim=1)[:, HAIR_LABEL]


@torch.no_grad()
def prob_to_cache(prob: torch.Tensor) -> np.ndarray:
    """[1,H,W] probability -> HAIR_PROB_SIZE square uint8 map for the store."""
    small = F.interpolate(
        prob.float().unsqueeze(0), size=(HAIR_PROB_SIZE, HAIR_PROB_SIZE), mode="area"
    )[0, 0]
    return small.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()


@torch.no_grad()
def prob_from_cache(
    cached: np.ndarray, size: tuple[int, int], device: torch.device | str
) -> torch.Tensor:
    """0..1 float map from the store -> [1,H,W] probability on ``device``."""
    t = torch.from_numpy(np.ascontiguousarray(cached, dtype=np.float32)).to(device)
    return F.interpolate(
        t.view(1, 1, *t.shape), size=size, mode="bilinear", align_corners=False
    )[0]


@torch.no_grad()
def soft_hair_mask(
    logits: torch.Tensor,
    threshold: float = 0.5,
    feather_px: float = 0.0,
) -> torch.Tensor:
    """Turn [1,19,H,W] parser logits into a soft [1,H,W] hair mask in [0,1]."""
    return mask_from_probability(hair_probability(logits), threshold, feather_px)


@torch.no_grad()
def mask_from_probability(
    probs: torch.Tensor,
    threshold: float = 0.5,
    feather_px: float = 0.0,
) -> torch.Tensor:
    """Soft [1,H,W] hair mask from a hair probability map.

    Pixels ramp from 0 to 1 around ``threshold`` instead of flipping on a hard
    argmax, which keeps edges and flyaways soft and stable. Feathering only
    softens inward, so color never spills past the parser's hair boundary.
    """
    lo = max(0.0, threshold - MASK_RAMP_WIDTH / 2)
    hi = min(1.0, threshold + MASK_RAMP_WIDTH / 2)
    mask = ((probs - lo) / max(hi - lo, 1e-6)).clamp(0.0, 1.0)
    if feather_px <= 0:
        return mask
    return torch.minimum(mask, feather_mask(mask, feather_px))


@torch.no_grad()
def feather_mask(mask: torch.Tensor, feather_px: float) -> torch.Tensor:
    """Gaussian-feather a [1,H,W] mask by roughly ``feather_px`` pixels."""
    if feather_px <= 0:
        return mask
    sigma = float(feather_px) / 2.0
    radius = max(1, int(round(sigma * 3)))
    coords = torch.arange(-radius, radius + 1, device=mask.device, dtype=torch.float32)
    kernel = torch.exp(-(coords**2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    x = mask.float().unsqueeze(0)
    x = F.pad(x, (radius, radius, 0, 0), mode="replicate")
    x = F.conv2d(x, kernel.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, radius, radius), mode="replicate")
    x = F.conv2d(x, kernel.view(1, 1, -1, 1))
    return x.squeeze(0).clamp(0.0, 1.0)


@torch.no_grad()
def hair_similarity(
    lab: torch.Tensor, mask: torch.Tensor, color_match: float
) -> torch.Tensor:
    """[1,H,W] weight in [0,1]: how much each pixel looks like the hair.

    Robust Lab statistics (median and MAD) come from the mask's confident core.
    Pixels within ``k`` deviations keep full weight and fade beyond it, where
    ``k`` shrinks as ``color_match`` goes from 0 (off) to 1 (strict).
    """
    color_match = float(max(0.0, min(1.0, color_match)))
    if color_match <= 0.0:
        return torch.ones_like(mask)
    core = mask[0] >= 0.8
    if int(core.sum()) < 50:
        core = mask[0] > 0.0
    if int(core.sum()) < 50:
        return torch.ones_like(mask)
    vals = lab[:, core]
    median = vals.median(dim=1).values
    mad = (vals - median.view(3, 1)).abs().median(dim=1).values * 1.4826
    k = 4.0 - 2.5 * color_match
    scale_l = max(float(mad[0]) * k, 4.0)
    scale_ab = max(float(mad[1:].max()) * k, 3.0)
    d_l = (lab[0] - median[0]) / scale_l
    d_ab = (lab[1:3] - median[1:3].view(2, 1, 1)) / scale_ab
    dist = torch.sqrt(d_l * d_l + (d_ab * d_ab).sum(dim=0))
    return torch.exp(-2.0 * (dist - 1.0).clamp(min=0.0) ** 2).unsqueeze(0)


@torch.no_grad()
def recolor_hair_lab(
    img: torch.Tensor,
    mask: torch.Tensor,
    color: Sequence[float],
    strength: float = 1.0,
    lightness: float = 0.5,
    color_match: float = 0.0,
) -> torch.Tensor:
    """Recolor the masked hair of a [3,H,W] uint8 image in Lab space.

    Hue and chroma move to the target color while each pixel keeps its own
    brightness offset from the hair's mean, so strand highlights and shadows
    survive. ``lightness`` shifts the hair's mean brightness toward the target
    color's (0 keeps it, 1 matches it). ``color_match`` (0..1) fades out masked
    pixels that don't look like the hair, such as background or skin showing
    between strands. Returns a [3,H,W] uint8 image.
    """
    strength = float(max(0.0, min(1.0, strength)))
    lightness = float(max(0.0, min(1.0, lightness)))
    w = mask.float().clamp(0.0, 1.0)
    if w.dim() == 2:
        w = w.unsqueeze(0)
    if w.shape[-2:] != img.shape[-2:]:
        w = F.interpolate(
            w.unsqueeze(0), size=img.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(0)

    weight_sum = w.sum()
    if strength <= 0.0 or weight_sum < 1.0:
        return img

    img_f = img.float().div(255.0).unsqueeze(0)
    lab = kc.rgb_to_lab(img_f)[0]
    target_rgb = torch.tensor(
        [float(c) / 255.0 for c in color], dtype=torch.float32, device=img.device
    ).view(1, 3, 1, 1)
    target = kc.rgb_to_lab(target_rgb)[0, :, 0, 0]

    w = w * hair_similarity(lab, w, color_match)
    weight_sum = w.sum()
    if weight_sum < 1.0:
        return img
    mean = (lab * w).flatten(1).sum(dim=1) / weight_sum
    new_l = lab[0:1] + lightness * (target[0] - mean[0])
    new_l = new_l.clamp(0.0, 100.0)
    # Darker pixels can't hold the target's full chroma in sRGB; scaling it by
    # relative lightness avoids gamut clipping that would crush the strands.
    chroma_scale = (new_l / max(float(target[0]), 1.0)).clamp(0.0, 1.0)
    new_ab = target[1:3].view(2, 1, 1) * chroma_scale + (
        lab[1:3] - mean[1:3].view(2, 1, 1)
    )
    recolored = torch.cat([new_l, new_ab], dim=0)
    rgb = kc.lab_to_rgb(recolored.unsqueeze(0))[0].clamp(0.0, 1.0)

    alpha = w * strength
    out = img_f[0] * (1.0 - alpha) + rgb * alpha
    return out.mul(255.0).round().clamp(0, 255).to(torch.uint8)


@torch.no_grad()
def sample_hair_color(
    img: torch.Tensor,
    prob: torch.Tensor,
    min_prob: float = 0.6,
    min_pixels: int = 200,
) -> Optional[tuple[float, float, float]]:
    """Typical RGB hair color of ``img`` [3,H,W] uint8 under ``prob`` [1,H,W].

    Uses the per-channel Lab median of confident hair pixels so highlights and
    shadows don't skew it. Returns None when too little hair is visible.
    """
    sel = prob[0].to(img.device) >= min_prob
    if int(sel.sum()) < min_pixels:
        return None
    lab = kc.rgb_to_lab(img.float().div(255.0).unsqueeze(0))[0]
    median = lab[:, sel].median(dim=1).values
    rgb = kc.lab_to_rgb(median.view(1, 3, 1, 1))[0, :, 0, 0].clamp(0.0, 1.0)
    r, g, b = (float(v) for v in rgb.mul(255.0).round().cpu())
    return r, g, b
