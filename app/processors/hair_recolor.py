"""Luminance-preserving hair recoloring on top of the face parser's hair class."""

from __future__ import annotations

from typing import Sequence

import kornia.color as kc
import torch
import torch.nn.functional as F

HAIR_LABEL = 17
MASK_RAMP_WIDTH = 0.3


@torch.no_grad()
def soft_hair_mask(
    logits: torch.Tensor,
    threshold: float = 0.5,
    feather_px: float = 0.0,
) -> torch.Tensor:
    """Turn [1,19,H,W] parser logits into a soft [1,H,W] hair mask in [0,1].

    Pixels ramp from 0 to 1 around ``threshold`` (hair probability) instead of
    flipping on a hard argmax, which keeps edges and flyaways soft and stable.
    """
    probs = torch.softmax(logits.float(), dim=1)[:, HAIR_LABEL]
    lo = max(0.0, threshold - MASK_RAMP_WIDTH / 2)
    hi = min(1.0, threshold + MASK_RAMP_WIDTH / 2)
    mask = ((probs - lo) / max(hi - lo, 1e-6)).clamp(0.0, 1.0)
    return feather_mask(mask, feather_px)


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
def recolor_hair_lab(
    img: torch.Tensor,
    mask: torch.Tensor,
    color: Sequence[float],
    strength: float = 1.0,
    lightness: float = 0.5,
) -> torch.Tensor:
    """Recolor the masked hair of a [3,H,W] uint8 image in Lab space.

    Hue and chroma move to the target color while each pixel keeps its own
    brightness offset from the hair's mean, so strand highlights and shadows
    survive. ``lightness`` shifts the hair's mean brightness toward the target
    color's (0 keeps it, 1 matches it). Returns a [3,H,W] uint8 image.
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
