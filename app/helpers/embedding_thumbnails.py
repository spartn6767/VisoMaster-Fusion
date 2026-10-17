"""Picture and creation details for merged embeddings.

A merged embedding has no face of its own, so the source face whose
embedding is closest to the merged one stands in as its picture. The
picture is kept as a small base64 JPEG so it can live in the JSON files
that already hold the embedding.
"""

from __future__ import annotations

import base64
import binascii
import html
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

THUMBNAIL_SIZE = 96
METADATA_KEYS = ("thumbnail", "merge_method", "source_count")
_SPATIAL_KEYS = frozenset({"kps_5"})


def closest_face_index(
    merged_store: Mapping[str, Any], source_stores: Sequence[Mapping[str, Any]]
) -> int | None:
    """Index of the source whose embeddings best match the merged ones.

    Uses the mean cosine similarity over the recognition models both share.
    """
    best_index: int | None = None
    best_score = -np.inf
    for index, store in enumerate(source_stores):
        scores = []
        for model, merged in merged_store.items():
            if model in _SPATIAL_KEYS or model not in store:
                continue
            a = np.asarray(merged, dtype=np.float64).ravel()
            b = np.asarray(store[model], dtype=np.float64).ravel()
            if a.shape != b.shape:
                continue
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom > 0:
                scores.append(float(a @ b / denom))
        if scores and float(np.mean(scores)) > best_score:
            best_index, best_score = index, float(np.mean(scores))
    return best_index


def encode_thumbnail(face_bgr: np.ndarray | None) -> str | None:
    if face_bgr is None:
        return None
    face = np.asarray(face_bgr)
    if face.ndim != 3 or face.shape[2] != 3 or face.size == 0:
        return None
    face = cv2.resize(
        np.ascontiguousarray(face, dtype=np.uint8),
        (THUMBNAIL_SIZE, THUMBNAIL_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    ok, buffer = cv2.imencode(".jpg", face, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer.tobytes()).decode("ascii") if ok else None


def thumbnail_bytes(thumbnail: Any) -> bytes | None:
    if not isinstance(thumbnail, str) or not thumbnail:
        return None
    try:
        return base64.b64decode(thumbnail, validate=True)
    except (binascii.Error, ValueError):
        return None


def describe_merge(merge_method: Any, source_count: Any) -> str:
    if not isinstance(source_count, int) or source_count < 1:
        return ""
    if source_count == 1:
        return "Made from 1 face"
    if merge_method:
        return f"{merge_method} of {source_count} faces"
    return f"Merged from {source_count} faces"


def metadata_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: payload[key] for key in METADATA_KEYS if payload.get(key) is not None}


def metadata_of(embedding_button: Any) -> dict[str, Any]:
    return {
        key: getattr(embedding_button, key)
        for key in METADATA_KEYS
        if getattr(embedding_button, key, None) is not None
    }


def tooltip_text(
    name: str,
    thumbnail: Any,
    merge_method: Any,
    source_count: Any,
    has_kv_maps: bool = False,
) -> str:
    description = describe_merge(merge_method, source_count)
    has_picture = thumbnail_bytes(thumbnail) is not None
    if has_picture:
        picture = (
            f'<img src="data:image/jpeg;base64,{html.escape(thumbnail)}" '
            f'width="{THUMBNAIL_SIZE}" height="{THUMBNAIL_SIZE}">'
        )
    else:
        picture = (
            '<table border="1" cellspacing="0"><tr>'
            f'<td width="{THUMBNAIL_SIZE}" height="{THUMBNAIL_SIZE}" '
            'align="center" valign="middle">no img</td></tr></table>'
        )
    parts = [picture]
    parts.append(f"<b>{html.escape(name)}</b>")
    if description:
        parts.append(html.escape(description))
    if has_picture:
        parts.append("Picture: the closest source face")
    if has_kv_maps:
        parts.append("Includes K/V Maps")
    return "<br>".join(parts)
