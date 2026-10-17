"""Visible "modified by AI" watermark burned into recorded frames."""

from __future__ import annotations

import contextlib
import os
import subprocess
from typing import Any, Mapping

import cv2
import numpy as np

DEFAULT_TEXT = "Parts of this video are modified by AI"
TEXT_KEY = "RecordingWatermarkText"
PREVIEW_KEY = "ShowWatermarkInPreviewToggle"

_FONT = cv2.FONT_HERSHEY_SIMPLEX
# Text height as a fraction of the frame height, so it looks alike at 720p and 4K.
_HEIGHT_FRACTION = 0.03
# Longer text wraps onto more lines instead of running past this width.
_MAX_WIDTH_FRACTION = 0.6
_MARGIN_FRACTION = 0.025
_OPACITY = 0.7


def watermark_text(control: Mapping[str, Any]) -> str:
    return str(control.get(TEXT_KEY, DEFAULT_TEXT) or "").strip()


def should_draw(
    control: Mapping[str, Any],
    recording: bool,
    watermark_recording: bool,
    single_frame: bool,
) -> bool:
    """Recorded frames follow the answer given when recording started; the
    preview only shows the watermark with Show Watermark in Preview on."""
    if recording and not single_frame:
        return watermark_recording
    return bool(control.get(PREVIEW_KEY, False))


def _wrap(text: str, scale: float, thickness: int, max_width: float) -> list[str]:
    lines: list[str] = []
    line = ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and cv2.getTextSize(trial, _FONT, scale, thickness)[0][0] > max_width:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


def draw_watermark(frame: np.ndarray, text: str) -> np.ndarray:
    """Draws ``text`` (white with a dark outline, semi-transparent) in the
    bottom-right corner of an HxWx3 uint8 frame and returns the frame."""
    text = text.strip()
    if not text or frame.ndim != 3 or min(frame.shape[:2]) < 8:
        return frame
    if not frame.flags.writeable:
        frame = frame.copy()
    h, w = frame.shape[:2]
    (_, base_h), _ = cv2.getTextSize("Ag", _FONT, 1.0, 2)
    scale = max(_HEIGHT_FRACTION * h / base_h, 0.3)
    thickness = max(1, round(scale * 1.5))
    outline = max(1, round(scale * 1.2))
    margin = max(4, round(_MARGIN_FRACTION * h))
    lines = _wrap(text, scale, thickness, _MAX_WIDTH_FRACTION * w)
    sizes = [cv2.getTextSize(line, _FONT, scale, thickness) for line in lines]
    line_h = max(size[1] + baseline for size, baseline in sizes)
    gap = round(line_h * 0.35)
    block_w = max(size[0] for size, _ in sizes) + 2 * outline
    block_h = len(lines) * line_h + (len(lines) - 1) * gap + 2 * outline
    x1, y1 = max(0, w - margin - block_w), max(0, h - margin - block_h)
    roi = frame[y1 : y1 + block_h, x1 : x1 + block_w]
    # The outline is the text mask grown outwards, so it lines up with the text
    # whatever font engine OpenCV uses (thicker strokes can change glyph widths).
    mask = np.zeros(roi.shape[:2], np.uint8)
    for i, (line, ((line_w, text_h), _)) in enumerate(zip(lines, sizes)):
        origin = (block_w - outline - line_w, outline + i * (line_h + gap) + text_h)
        cv2.putText(mask, line, origin, _FONT, scale, 255, thickness, cv2.LINE_AA)
    size = 2 * outline + 1
    halo = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    text_a = (mask.astype(np.float32) / 255.0 * _OPACITY)[..., None]
    halo_a = (halo.astype(np.float32) / 255.0 * _OPACITY)[..., None]
    out = roi.astype(np.float32) * (1.0 - halo_a)
    out = out * (1.0 - text_a) + 255.0 * text_a
    roi[:] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    return frame


def tag_video_file(path: str, text: str) -> bool:
    """Stores ``text`` as the video file's comment metadata (stream copy, in place)."""
    if not text or not path or not os.path.isfile(path):
        return False
    root, ext = os.path.splitext(path)
    tagged = f"{root}.tagging{ext}"
    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-map",
        "0:v",
        "-map",
        "0:a?",
        "-c",
        "copy",
        "-metadata",
        f"comment={text}",
        "-y",
        tagged,
    ]
    try:
        subprocess.run(args, check=True)
        os.replace(tagged, path)
        return True
    except Exception as e:
        print(f"[WARN] Could not add the watermark metadata to {path}: {e}")
        with contextlib.suppress(OSError):
            os.remove(tagged)
        return False
