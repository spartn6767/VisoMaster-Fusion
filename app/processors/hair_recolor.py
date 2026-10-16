"""Luminance-preserving hair recoloring on top of the face parser's hair class."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import kornia.color as kc
import kornia.geometry.transform as kgm
import numpy as np
import torch
import torch.nn.functional as F

from app.processors.utils import faceutil
from app.processors.video_utils.alignment_store import HAIR_PROB_SIZE

HAIR_LABEL = 17
MASK_RAMP_WIDTH = 0.3
# Bounds on how much a strand's saturation may differ from the hair average.
MIN_CHROMA_RATIO = 0.5
MAX_CHROMA_RATIO = 1.25

CropKey = tuple[float, ...]
ProbFn = Callable[[torch.Tensor], torch.Tensor]

FACE_PARSER = "Face Parser"
HAIR_SEGMENTER = "Hair Segmenter"
SEGMENTER_HAIR_LABEL = 1
SEGMENTER_SIZE = 256
# How far the segmenter area reaches above the eyes, in face units.
SEGMENTER_AREA_TOP = 3.0
# Leading cache-key value that keeps segmenter masks apart from parser ones.
SEGMENTER_KEY = 0.0


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
    argmax, which keeps edges and flyaways soft and stable.
    """
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


def _box_mean(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Mean over a (2r+1)^2 window clipped to the image, for [1,1,H,W] input.

    Uses a summed-area table so large radii (full-resolution frames) stay cheap.
    """
    h, w = x.shape[-2:]
    table = F.pad(x.double(), (1, 0, 1, 0)).cumsum(-1).cumsum(-2)[0, 0]
    ys = torch.arange(h, device=x.device)
    xs = torch.arange(w, device=x.device)
    y0, y1 = (ys - radius).clamp(0, h), (ys + radius + 1).clamp(0, h)
    x0, x1 = (xs - radius).clamp(0, w), (xs + radius + 1).clamp(0, w)
    total = table[y1][:, x1] - table[y0][:, x1] - table[y1][:, x0] + table[y0][:, x0]
    area = (y1 - y0).view(-1, 1) * (x1 - x0).view(1, -1)
    return (total / area).to(x.dtype).view(1, 1, h, w)


@torch.no_grad()
def guided_refine_mask(
    img: torch.Tensor, mask: torch.Tensor, radius: int, eps: float = 1e-3
) -> torch.Tensor:
    """Snap a coarse [1,H,W] mask onto image edges with a guided filter.

    Uses the image's grayscale as the guide (He et al.), so the mask follows
    the brightness edges of real strands and drops the gaps between them.
    """
    radius = int(radius)
    if radius <= 0:
        return mask
    guide = img.float().div(255.0)
    guide = (0.299 * guide[0] + 0.587 * guide[1] + 0.114 * guide[2]).view(
        1, 1, *guide.shape[-2:]
    )
    p = mask.float().to(guide.device).view(1, 1, *mask.shape[-2:])
    if p.shape[-2:] != guide.shape[-2:]:
        p = F.interpolate(
            p, size=guide.shape[-2:], mode="bilinear", align_corners=False
        )

    def box(x: torch.Tensor) -> torch.Tensor:
        return _box_mean(x, radius)

    mean_i = box(guide)
    mean_p = box(p)
    var_i = box(guide * guide) - mean_i * mean_i
    cov_ip = box(guide * p) - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    q = box(a) * guide + box(b)
    return q.clamp(0.0, 1.0).view(1, *guide.shape[-2:])


@torch.no_grad()
def recolor_hair_lab(
    img: torch.Tensor,
    mask: torch.Tensor,
    color: Sequence[float],
    strength: float = 1.0,
    lightness: float = 0.5,
) -> torch.Tensor:
    """Recolor the masked hair of a [3,H,W] uint8 image in Lab space.

    Every pixel takes the target's hue; brightness and relative saturation
    stay per pixel, so strand highlights and shadows survive without shadows
    drifting to the opposite (e.g. blue) hue. ``lightness`` shifts the hair's mean brightness toward the target
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
    chroma = lab[1:3].norm(dim=0, keepdim=True)
    mean_chroma = (chroma * w).sum() / weight_sum
    chroma_ratio = (chroma / mean_chroma.clamp(min=1.0)).clamp(
        MIN_CHROMA_RATIO, MAX_CHROMA_RATIO
    )
    new_ab = target[1:3].view(2, 1, 1) * chroma_scale * chroma_ratio
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


def hair_area(parameters: dict) -> CropKey:
    """(scale, vy_ratio) of the crop the hair parser sees, from Hair Area sliders."""
    return (
        float(parameters.get("HairMakeupAreaSizeDecimalSlider", 2.5)),
        float(parameters.get("HairMakeupAreaShiftDecimalSlider", -0.125)),
    )


def uses_segmenter(parameters: dict) -> bool:
    return parameters.get("HairMakeupMaskModelSelection", FACE_PARSER) == HAIR_SEGMENTER


def segmenter_area(parameters: dict) -> tuple[float, float]:
    """(width, reach_down) of the hair segmenter area, in face units."""
    return (
        float(parameters.get("HairMakeupAreaWidthDecimalSlider", 7.0)),
        float(parameters.get("HairMakeupAreaReachDownDecimalSlider", 8.0)),
    )


def hair_area_key(parameters: dict) -> CropKey:
    """Identifies the hair mask model and its area, e.g. for cache invalidation."""
    if uses_segmenter(parameters):
        return (SEGMENTER_KEY, *segmenter_area(parameters))
    return hair_area(parameters)


@torch.no_grad()
def segmenter_hair_probability(logits: torch.Tensor) -> torch.Tensor:
    """[1,6,H,W] selfie multiclass logits -> [1,H,W] hair probability."""
    return torch.softmax(logits.float(), dim=1)[:, SEGMENTER_HAIR_LABEL]


def segmenter_box(
    kps: np.ndarray, width: float, reach_down: float
) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) frame box the hair segmenter sees.

    Upright and centred on the face, from above the head down ``reach_down``
    face units below the eyes, so long hair is covered without losing the top.
    A face unit is the larger of eye distance and eye-to-mouth distance, which
    stays stable when the head turns.
    """
    pts = np.asarray(kps, dtype=np.float32)[:5]
    eyes = (pts[0] + pts[1]) / 2
    mouth = (pts[3] + pts[4]) / 2
    unit = max(
        float(np.linalg.norm(pts[1] - pts[0])),
        float(np.linalg.norm(mouth - eyes)),
        1.0,
    )
    cx = float(eyes[0] + mouth[0]) / 2
    left = math.floor(cx - width * unit / 2)
    top = math.floor(float(eyes[1]) - SEGMENTER_AREA_TOP * unit)
    right = max(math.ceil(cx + width * unit / 2), left + 1)
    bottom = max(math.ceil(float(eyes[1]) + reach_down * unit), top + 1)
    return left, top, right, bottom


@torch.no_grad()
def crop_box(
    img: torch.Tensor, box: Sequence[int], size: int = SEGMENTER_SIZE
) -> torch.Tensor:
    """[3,size,size] uint8 resample of the frame inside ``box`` (black off-frame)."""
    left, top, right, bottom = (int(v) for v in box)
    h, w = img.shape[-2:]
    canvas = torch.zeros(
        (3, bottom - top, right - left), dtype=torch.float32, device=img.device
    )
    y0, y1 = max(top, 0), min(bottom, h)
    x0, x1 = max(left, 0), min(right, w)
    if y1 > y0 and x1 > x0:
        canvas[:, y0 - top : y1 - top, x0 - left : x1 - left] = img[
            :, y0:y1, x0:x1
        ].float()
    out = F.interpolate(
        canvas.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0]
    return out.round().clamp(0, 255).to(torch.uint8)


@torch.no_grad()
def hair_prob_for_crop(
    crop: torch.Tensor,
    prob_fn: ProbFn,
    hair_cache: Optional[HairMaskCache],
    crop_key: CropKey,
) -> torch.Tensor:
    """[1,H,W] hair probability of a crop, from the cache when possible."""
    size = (int(crop.shape[-2]), int(crop.shape[-1]))
    cached = hair_cache.load(crop_key) if hair_cache is not None else None
    if cached is not None:
        return prob_from_cache(cached, size, crop.device)
    prob = prob_fn(crop)
    if hair_cache is not None:
        hair_cache.save(prob_to_cache(prob), crop_key)
        smoothed = hair_cache.load(crop_key)
        if smoothed is not None:
            prob = prob_from_cache(smoothed, size, crop.device)
    return prob


@torch.no_grad()
def hair_crop_and_prob(
    img: torch.Tensor,
    kps: np.ndarray,
    parameters: dict,
    parser_logits: Callable[[torch.Tensor], torch.Tensor],
    segmenter_logits: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(uint8 crop, [1,H,W] hair probability) of one face's hair area, uncached."""
    if uses_segmenter(parameters) and segmenter_logits is not None:
        crop = crop_box(img, segmenter_box(kps, *segmenter_area(parameters)))
        return crop, segmenter_hair_probability(segmenter_logits(crop))
    scale, vy_ratio = hair_area(parameters)
    crop, _, _ = faceutil.warp_face_by_face_landmark_x(
        img, kps, dsize=512, scale=scale, vy_ratio=vy_ratio
    )
    if crop.dtype != torch.uint8:
        crop = crop.round().clamp(0, 255).to(torch.uint8)
    return crop, hair_probability(parser_logits(crop))


@torch.no_grad()
def mask_to_frame_roi(
    mask: torch.Tensor, M_c2o: np.ndarray, frame_hw: Sequence[int]
) -> Optional[tuple[torch.Tensor, tuple[int, int, int, int]]]:
    """Warp a [1,h,w] crop mask into the frame.

    Returns the mask over the crop's bounding box in the frame and that box as
    (top, bottom, left, right), or None when the crop is off-frame.
    """
    frame_h, frame_w = int(frame_hw[0]), int(frame_hw[1])
    m = np.asarray(M_c2o, dtype=np.float32)[:2]
    h, w = mask.shape[-2:]
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], np.float32)
    pts = corners @ m[:, :2].T + m[:, 2]
    left = max(math.floor(float(pts[:, 0].min())), 0)
    top = max(math.floor(float(pts[:, 1].min())), 0)
    right = min(math.ceil(float(pts[:, 0].max())) + 1, frame_w)
    bottom = min(math.ceil(float(pts[:, 1].max())) + 1, frame_h)
    if right <= left or bottom <= top:
        return None
    local = m.copy()
    local[0, 2] -= left
    local[1, 2] -= top
    warped = kgm.warp_affine(
        mask.float().unsqueeze(0),
        torch.from_numpy(local).to(mask.device).unsqueeze(0),
        dsize=(bottom - top, right - left),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0].clamp(0.0, 1.0)
    return warped, (top, bottom, left, right)


# (mask over the frame ROI, ROI as (top, bottom, left, right), frame pixels per
# 512px-crop pixel)
FrameMask = tuple[torch.Tensor, tuple[int, int, int, int], float]


def _mask_params(parameters: dict) -> tuple[float, float]:
    return (
        parameters.get("HairMakeupMaskThresholdDecimalSlider", 0.5),
        parameters.get("HairMakeupEdgeSoftnessSlider", 4),
    )


@torch.no_grad()
def _parser_frame_mask(
    img: torch.Tensor,
    kps: np.ndarray,
    parameters: dict,
    parser_logits: Callable[[torch.Tensor], torch.Tensor],
    hair_cache: Optional[HairMaskCache],
) -> Optional[FrameMask]:
    crop_key = hair_area(parameters)
    crop, _, M_c2o = faceutil.warp_face_by_face_landmark_x(
        img, kps, dsize=512, scale=crop_key[0], vy_ratio=crop_key[1]
    )
    if crop.dtype != torch.uint8:
        crop = crop.round().clamp(0, 255).to(torch.uint8)
    prob = hair_prob_for_crop(
        crop, lambda c: hair_probability(parser_logits(c)), hair_cache, crop_key
    )
    threshold, feather = _mask_params(parameters)
    mask = mask_from_probability(prob, threshold=threshold, feather_px=feather)
    placed = mask_to_frame_roi(mask, M_c2o, img.shape[-2:])
    if placed is None:
        return None
    frame_px = math.sqrt(abs(float(np.linalg.det(np.asarray(M_c2o)[:2, :2]))))
    return placed[0], placed[1], frame_px


_segmenter_diag_done = False


@torch.no_grad()
def _segmenter_frame_mask(
    img: torch.Tensor,
    kps: np.ndarray,
    parameters: dict,
    segmenter_logits: Callable[[torch.Tensor], torch.Tensor],
    hair_cache: Optional[HairMaskCache],
) -> Optional[FrameMask]:
    key = hair_area_key(parameters)
    left, top, right, bottom = segmenter_box(kps, *segmenter_area(parameters))
    h, w = img.shape[-2:]
    y0, y1 = max(top, 0), min(bottom, h)
    x0, x1 = max(left, 0), min(right, w)
    if y1 <= y0 or x1 <= x0:
        return None
    crop = crop_box(img, (left, top, right, bottom))
    prob = hair_prob_for_crop(
        crop,
        lambda c: segmenter_hair_probability(segmenter_logits(c)),
        hair_cache,
        key,
    )
    box_h, box_w = bottom - top, right - left
    # Threshold and feather on a grid with 512px on the long side, so Edge
    # Softness means the same as with the face parser's 512px crop.
    frame_px = max(box_h, box_w) / 512.0
    grid = (max(1, round(box_h / frame_px)), max(1, round(box_w / frame_px)))
    prob = F.interpolate(
        prob.float().unsqueeze(0), size=grid, mode="bilinear", align_corners=False
    )[0]
    threshold, feather = _mask_params(parameters)
    mask = mask_from_probability(prob, threshold=threshold, feather_px=feather)
    full = F.interpolate(
        mask.unsqueeze(0), size=(box_h, box_w), mode="bilinear", align_corners=False
    )[0]
    roi_mask = full[:, y0 - top : y1 - top, x0 - left : x1 - left]
    global _segmenter_diag_done
    if not _segmenter_diag_done:
        _segmenter_diag_done = True
        print(
            f"[INFO] Hair Segmenter mask: frame={tuple(img.shape)} {img.dtype} "
            f"box={(left, top, right, bottom)} crop={tuple(crop.shape)} "
            f"{crop.dtype} hair_prob_max={float(prob.max()):.2f} "
            f"threshold={threshold:.2f} mask_max={float(roi_mask.max()):.2f} "
            f"hair_area={float((roi_mask > 0).float().mean()):.1%}"
        )
    return roi_mask, (y0, y1, x0, x1), frame_px


@torch.no_grad()
def recolor_hair_in_frame(
    img: torch.Tensor,
    kps: np.ndarray,
    parameters: dict,
    parser_logits: Callable[[torch.Tensor], torch.Tensor],
    hair_cache: Optional[HairMaskCache] = None,
    segmenter_logits: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """Recolor one face's hair directly in the [3,H,W] frame.

    The hair mask comes either from the face parser on a 512px crop sized by the
    Hair Area sliders, or from the hair segmenter on a tall area from above the
    head down past the shoulders. Either way the mask is placed in the frame and
    the recolor runs on the frame's own pixels, so strands stay sharp however
    large the area is.
    """
    if uses_segmenter(parameters) and segmenter_logits is not None:
        placed = _segmenter_frame_mask(
            img, kps, parameters, segmenter_logits, hair_cache
        )
    else:
        placed = _parser_frame_mask(img, kps, parameters, parser_logits, hair_cache)
    if placed is None:
        return img
    frame_mask, (top, bottom, left, right), frame_px = placed
    if float(frame_mask.max()) <= 0.0:
        return img

    roi = img[:, top:bottom, left:right]
    roi_u8 = (
        roi if roi.dtype == torch.uint8 else roi.round().clamp(0, 255).to(torch.uint8)
    )
    refine_radius = int(parameters.get("HairMakeupStrandRefineSlider", 0))
    if refine_radius > 0:
        # The slider is in 512px-crop pixels; scale it to frame pixels.
        frame_mask = guided_refine_mask(
            roi_u8, frame_mask, max(1, int(round(refine_radius * frame_px)))
        )
    color = [
        parameters.get("HairMakeupRedSlider", 0),
        parameters.get("HairMakeupGreenSlider", 0),
        parameters.get("HairMakeupBlueSlider", 0),
    ]
    recolored = recolor_hair_lab(
        roi_u8,
        frame_mask,
        color,
        strength=parameters.get("HairMakeupStrengthDecimalSlider", 0.8),
        lightness=parameters.get("HairMakeupLightnessDecimalSlider", 0.5),
    )
    out = img.clone()
    out[:, top:bottom, left:right] = torch.where(
        frame_mask > 0, recolored.to(img.dtype), roi
    )
    return out
