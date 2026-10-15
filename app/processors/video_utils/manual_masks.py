"""
Hand-painted face masks stored in a canonical face space.

A mask is a ``(2, MASK_SIZE, MASK_SIZE)`` uint8 array: layer 0 adds to the swap
mask, layer 1 erases from it. The canonical space is fixed by the face's 5
keypoints and does not depend on the swapper model, so a painted mask follows
the face as it moves and survives a model change. An all-zero mask is an
explicit "no manual mask" keyframe.
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

MASK_SIZE = 256
ADD_LAYER = 0
ERASE_LAYER = 1

# ArcFace 5-point template (normalised to its 112px crop), shrunk so the canvas
# also covers forehead, hair and jaw around the face.
_ARCFACE_TEMPLATE = (
    np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float64,
    )
    / 112.0
)
_TEMPLATE_SCALE = 0.55
CANONICAL_KPS = ((_ARCFACE_TEMPLATE - 0.5) * _TEMPLATE_SCALE + 0.5) * MASK_SIZE


def empty_mask() -> np.ndarray:
    return np.zeros((2, MASK_SIZE, MASK_SIZE), dtype=np.uint8)


def as_mask(value: object) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.shape != (2, MASK_SIZE, MASK_SIZE):
        return None
    return arr.astype(np.uint8, copy=True)


def is_empty(mask: Optional[np.ndarray]) -> bool:
    return mask is None or not mask.any()


def similarity_matrix(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares 2D similarity (rotation, uniform scale, shift) src -> dst, 3x3."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    s, d = src - src_mean, dst - dst_mean
    denom = float(np.sum(s**2))
    if denom <= 1e-12:
        raise ValueError("Degenerate keypoints")
    a = float(np.sum(s[:, 0] * d[:, 0] + s[:, 1] * d[:, 1])) / denom
    b = float(np.sum(s[:, 0] * d[:, 1] - s[:, 1] * d[:, 0])) / denom
    tx = dst_mean[0] - (a * src_mean[0] - b * src_mean[1])
    ty = dst_mean[1] - (b * src_mean[0] + a * src_mean[1])
    return np.array([[a, -b, tx], [b, a, ty], [0.0, 0.0, 1.0]], dtype=np.float64)


def frame_to_mask_matrix(kps_5: np.ndarray) -> np.ndarray:
    """3x3 matrix from frame pixels to canonical mask pixels."""
    return similarity_matrix(np.asarray(kps_5)[:5], CANONICAL_KPS)


def paint_stroke(
    mask: Optional[np.ndarray],
    points: Sequence[Sequence[float]],
    kps_5: np.ndarray,
    radius: float,
    erase: bool = False,
    feather: float = 0.0,
) -> np.ndarray:
    """Returns a new mask with a stroke (frame pixel coords) painted on it.

    Painting one layer clears the other under the stroke, so the latest stroke wins.
    """
    out = empty_mask() if mask is None else mask.copy()
    if not len(points):
        return out
    m = frame_to_mask_matrix(kps_5)
    scale = float(np.hypot(m[0, 0], m[1, 0]))
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    pts = pts @ m[:2, :2].T + m[:2, 2]
    r = max(1, int(round(radius * scale)))
    stroke = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.uint8)
    ipts = np.round(pts).astype(np.int32)
    for p in ipts:
        cv2.circle(stroke, (int(p[0]), int(p[1])), r, 255, -1, cv2.LINE_AA)
    if len(ipts) > 1:
        cv2.polylines(stroke, [ipts.reshape(-1, 1, 2)], False, 255, 2 * r, cv2.LINE_AA)
    sigma = feather * scale
    if sigma > 0.25:
        k = int(sigma * 3) * 2 + 1
        stroke = cv2.GaussianBlur(stroke, (k, k), sigma)
    target, other = (ERASE_LAYER, ADD_LAYER) if erase else (ADD_LAYER, ERASE_LAYER)
    out[target] = np.maximum(out[target], stroke)
    out[other] = np.minimum(out[other], 255 - stroke)
    return out


def warp_mask(
    mask: np.ndarray, matrix: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    """Warps every layer with a 3x3 mask->target matrix; returns float32 in [0, 1]."""
    w, h = int(size[0]), int(size[1])
    layers = [
        cv2.warpAffine(
            np.ascontiguousarray(mask[i]),
            np.asarray(matrix, dtype=np.float64)[:2],
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        for i in range(mask.shape[0])
    ]
    return np.stack(layers).astype(np.float32) / 255.0


def mask_to_frame(
    mask: np.ndarray, kps_5: np.ndarray, frame_w: int, frame_h: int
) -> np.ndarray:
    """Both layers in frame pixel space, float32 [0, 1], shape (2, h, w)."""
    return warp_mask(
        mask, np.linalg.inv(frame_to_mask_matrix(kps_5)), (frame_w, frame_h)
    )


def mask_to_crop(
    mask: np.ndarray, kps_5: np.ndarray, frame_to_crop: np.ndarray, crop_size: int
) -> np.ndarray:
    """Both layers in a face-crop space given its 3x3 frame->crop matrix."""
    matrix = np.asarray(frame_to_crop, dtype=np.float64) @ np.linalg.inv(
        frame_to_mask_matrix(kps_5)
    )
    return warp_mask(mask, matrix, (crop_size, crop_size))


def crop_to_canonical(
    crop_mask: np.ndarray, kps_5: np.ndarray, crop_to_frame: np.ndarray
) -> np.ndarray:
    """A float [0, 1] face-crop mask in canonical mask space as uint8 (MASK_SIZE^2)."""
    matrix = frame_to_mask_matrix(kps_5) @ np.asarray(crop_to_frame, dtype=np.float64)
    src = np.clip(np.asarray(crop_mask, dtype=np.float32), 0.0, 1.0) * 255.0
    return cv2.warpAffine(
        src.astype(np.uint8),
        matrix[:2],
        (MASK_SIZE, MASK_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def combine(swap_mask: np.ndarray, add: np.ndarray, erase: np.ndarray) -> np.ndarray:
    """max(swap_mask, add) * (1 - erase); works for numpy or torch operands."""
    return np.maximum(swap_mask, add) * (1.0 - erase)
