"""
Persistent, frame-indexed store of face alignments (bbox, kps_5, dense kps, kps_203,
embedding, face_id) so detection is computed once per frame and reused on seek,
preview and render.

Coordinates are stored normalised to the frame size they were detected on, so an
entry written at preview resolution can be reused at render resolution.

Each frame records the detector-settings signature it was produced with. A lookup
with a different signature is a miss, except for frames containing a manually
corrected face, which always win.
"""

from __future__ import annotations

import bisect
import copy
import hashlib
import json
import math
import os
import threading
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from app.processors.video_utils.manual_masks import HAIR_SPACE, as_mask, is_empty

ALIGNMENTS_FORMAT_VERSION = 1
SIDECAR_SUFFIX = ".alignments.npz"
# Side of the cached hair-probability map, in the 512px makeup crop.
HAIR_PROB_SIZE = 128

SOURCE_SEQUENTIAL = "sequential"
SOURCE_SINGLE = "single"
SOURCE_MANUAL = "manual"

# Sources a sequential (playback/render) consumer accepts. Single-frame detections
# are unsmoothed and not target-filtered, so playback re-detects over them.
SEQUENTIAL_SOURCES = frozenset({SOURCE_SEQUENTIAL, SOURCE_MANUAL})
ALL_SOURCES = frozenset({SOURCE_SEQUENTIAL, SOURCE_SINGLE, SOURCE_MANUAL})

SIGNATURE_CONTROL_KEYS: tuple[str, ...] = (
    "DetectorModelSelection",
    "DetectorScoreSlider",
    "MaxFacesToDetectSlider",
    "AutoRotationToggle",
    "LandmarkDetectToggle",
    "LandmarkDetectModelSelection",
    "LandmarkDetectScoreSlider",
    "DetectFromPointsToggle",
    "LandmarkMeanEyesToggle",
    "RecognitionModelSelection",
    "KPSSmoothingEnableToggle",
    "KPSEmaAlphaSlider",
    "FaceTrackingEnableToggle",
    "ByteTrackTrackThreshSlider",
    "ByteTrackMatchThreshSlider",
    "ByteTrackTrackBufferSlider",
)


def detector_signature(
    control: Mapping[str, Any], target_face_ids: Iterable[Any] = ()
) -> str:
    """Hash of every setting that changes which faces are detected and where."""
    payload = {
        "control": {k: control.get(k) for k in SIGNATURE_CONTROL_KEYS},
        "targets": sorted(str(face_id) for face_id in target_face_ids),
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def alignments_cache_enabled(control: Mapping[str, Any]) -> bool:
    """VR180 and manual rotation run their own coordinate spaces and bypass the cache."""
    return (
        bool(control.get("AlignmentsCacheEnableToggle", True))
        and not control.get("VR180ModeEnableToggle", False)
        and not control.get("ManualRotationEnableToggle", False)
    )


def faces_require_203(
    face_parameters: Optional[Mapping[str, Any]], is_master_edit_active: bool
) -> bool:
    """True when any face uses a feature that needs the dense 203-point landmarks."""
    if not face_parameters:
        return False
    for face_params in face_parameters.values():
        if not isinstance(face_params, Mapping):
            continue
        if face_params.get("FaceExpressionEnableBothToggle", False) or face_params.get(
            "AutoMouthExpressionEnableToggle", False
        ):
            return True
        if is_master_edit_active and any(
            face_params.get(key, False)
            for key in (
                "FaceEditorEnableToggle",
                "FaceMakeupEnableToggle",
                "HairMakeupEnableToggle",
                "EyeBrowsMakeupEnableToggle",
                "LipsMakeupEnableToggle",
            )
        ):
            return True
    return False


def sidecar_path_for(json_path: str) -> str:
    """'<dir>/<name>.json' -> '<dir>/<name>.alignments.npz'."""
    root, ext = os.path.splitext(json_path)
    return (root if ext.lower() == ".json" else json_path) + SIDECAR_SUFFIX


def _as_hair_prob(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.shape != (HAIR_PROB_SIZE, HAIR_PROB_SIZE):
        return None
    return np.ascontiguousarray(arr, dtype=np.uint8)


def _matching_hair_prob(
    frame: "FrameAlignments", face_key: Optional[str], index: int
) -> Optional[np.ndarray]:
    if face_key is None:
        if 0 <= index < len(frame.faces):
            return frame.faces[index].hair_prob
        return None
    for face in frame.faces:
        if face.hair_prob is not None and face_key in (
            face.assigned_face_id,
            face.face_id,
        ):
            return face.hair_prob
    return None


def _as_points(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != 2:
        return None
    return arr


def _scale(points: Optional[np.ndarray], sx: float, sy: float) -> Optional[np.ndarray]:
    if points is None:
        return None
    out: np.ndarray = points.astype(np.float32, copy=True)
    out[:, 0] *= sx
    out[:, 1] *= sy
    return out


@dataclass
class AlignmentFace:
    """One face on one frame. Spatial fields are normalised to [0, 1] frame space."""

    bbox: np.ndarray
    kps_5: np.ndarray
    kps_all: Optional[np.ndarray] = None
    kps_203: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    face_id: Optional[str] = None
    # Target face the user pinned this face to; skips embedding matching.
    assigned_face_id: Optional[str] = None
    source: str = SOURCE_SEQUENTIAL
    # Hand-painted (2, 256, 256) add/erase mask in canonical face space; all-zero
    # means "no manual mask from here on" for this face.
    mask: Optional[np.ndarray] = None
    # Use the computed mask on this frame only; a stored mask still holds forward.
    mask_skip: bool = False
    # Hand-painted (2, HAIR_MASK_SIZE^2) add/erase edits of the Hair Makeup mask in
    # HAIR_SPACE; held forward and reset exactly like ``mask``.
    hair_mask: Optional[np.ndarray] = None
    hair_mask_skip: bool = False
    # Cached face-parser hair probability (uint8, HAIR_PROB_SIZE square) in the
    # makeup crop; only filled when hair mask stabilization is on.
    hair_prob: Optional[np.ndarray] = None
    # Manual bbox whose landmarks the worker still has to detect (not persisted).
    refine: bool = False

    def to_pixels(self, frame_w: int, frame_h: int) -> dict[str, Any]:
        bbox: np.ndarray = self.bbox.astype(np.float32, copy=True)
        bbox[[0, 2]] *= frame_w
        bbox[[1, 3]] *= frame_h
        return {
            "bbox": bbox,
            "kps_5": _scale(self.kps_5, frame_w, frame_h),
            "kps_all": _scale(self.kps_all, frame_w, frame_h),
            "kps_203": _scale(self.kps_203, frame_w, frame_h),
            "embedding": None if self.embedding is None else self.embedding.copy(),
            "face_id": self.face_id,
            "assigned_face_id": self.assigned_face_id,
            "source": self.source,
            "mask": None if self.mask is None else self.mask.copy(),
            "mask_skip": self.mask_skip,
            "hair_mask": None if self.hair_mask is None else self.hair_mask.copy(),
            "hair_mask_skip": self.hair_mask_skip,
            "hair_prob": self.hair_prob,
            "refine": self.refine,
        }

    @classmethod
    def from_pixels(
        cls, face: Mapping[str, Any], frame_w: int, frame_h: int, source: str
    ) -> "AlignmentFace":
        sx, sy = 1.0 / float(frame_w), 1.0 / float(frame_h)
        bbox = np.asarray(face["bbox"], dtype=np.float32).reshape(4).copy()
        bbox[[0, 2]] *= sx
        bbox[[1, 3]] *= sy
        kps_5 = _as_points(face.get("kps_5"))
        if kps_5 is None:
            raise ValueError("Alignment face requires kps_5")
        embedding = face.get("embedding")
        face_id = face.get("face_id")
        return cls(
            bbox=bbox,
            kps_5=_scale(kps_5, sx, sy),  # type: ignore[arg-type]
            kps_all=_scale(_as_points(face.get("kps_all")), sx, sy),
            kps_203=_scale(_as_points(face.get("kps_203")), sx, sy),
            embedding=None
            if embedding is None
            else np.asarray(embedding, dtype=np.float32).reshape(-1).copy(),
            face_id=None if face_id is None else str(face_id),
            assigned_face_id=str(face["assigned_face_id"])
            if face.get("assigned_face_id")
            else None,
            source=str(face.get("source", source)),
            mask=as_mask(face.get("mask")),
            mask_skip=bool(face.get("mask_skip", False)),
            hair_mask=as_mask(face.get("hair_mask"), HAIR_SPACE),
            hair_mask_skip=bool(face.get("hair_mask_skip", False)),
            hair_prob=_as_hair_prob(face.get("hair_prob")),
            refine=bool(face.get("refine", False)),
        )


@dataclass
class FrameAlignments:
    signature: str
    faces: list[AlignmentFace] = field(default_factory=list)

    @property
    def is_pinned(self) -> bool:
        return any(f.source == SOURCE_MANUAL for f in self.faces)

    @property
    def sources(self) -> set[str]:
        return {f.source for f in self.faces}


def _copy_frame(frame: Optional[FrameAlignments]) -> Optional[FrameAlignments]:
    """Copies the face records but shares their arrays, which edits replace
    rather than write into."""
    if frame is None:
        return None
    return FrameAlignments(frame.signature, [copy.copy(f) for f in frame.faces])


UNDO_LIMIT = 100
SAVED_HISTORY_FILES = 5


def _nearest(frames: list[int], frame_number: int, step: int) -> Optional[int]:
    """Closest entry of sorted ``frames`` before (step < 0) or after (step > 0)."""
    if step < 0:
        i = bisect.bisect_left(frames, int(frame_number))
        return frames[i - 1] if i > 0 else None
    i = bisect.bisect_right(frames, int(frame_number))
    return frames[i] if i < len(frames) else None


@dataclass
class _EditStep:
    # whole: the step covers every frame, so applying it replaces the whole map.
    whole: bool
    before: dict[int, Optional[FrameAlignments]]
    after: dict[int, Optional[FrameAlignments]]

    @property
    def frames(self) -> list[int]:
        return [] if self.whole else sorted(self.before)


def faces_to_detection_arrays(
    faces: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rebuilds the (bboxes, kpss_5, kpss, kpss_203) tuple the detector produces."""
    n = len(faces)
    if n == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0, 5, 2), dtype=np.float32),
            np.empty((0, 68, 2), dtype=np.float32),
            np.empty((0, 203, 2), dtype=np.float32),
        )
    bboxes = np.stack([f["bbox"] for f in faces]).astype(np.float32)
    kpss_5 = np.stack([f["kps_5"] for f in faces]).astype(np.float32)

    dense = [f["kps_all"] if f["kps_all"] is not None else f["kps_5"] for f in faces]
    try:
        kpss = np.stack(dense).astype(np.float32)
    except ValueError:
        kpss = np.empty(n, dtype=object)
        for i, d in enumerate(dense):
            kpss[i] = d

    if all(f["kps_203"] is not None for f in faces):
        kpss_203 = np.stack([f["kps_203"] for f in faces]).astype(np.float32)
    else:
        kpss_203 = np.empty((0, 203, 2), dtype=np.float32)
    return bboxes, kpss_5, kpss, kpss_203


class AlignmentStore:
    """Thread-safe frame -> faces map shared by the feeder, workers and UI."""

    def __init__(self) -> None:
        self.hair_crop: Optional[tuple[float, ...]] = None
        self._lock = threading.RLock()
        self._frames: dict[int, FrameAlignments] = {}
        self.media_path: Optional[str] = None
        self._revision = 0
        self._saved: dict[str, int] = {}
        self._mask_index: dict[str, tuple[int, list[int]]] = {}
        self._undo: list[_EditStep] = []
        self._redo: list[_EditStep] = []
        self._edit_depth = 0
        self._edit_whole = False
        self._edit_before: dict[int, Optional[FrameAlignments]] = {}
        self._edit_revision = 0
        self._edited_index: tuple[int, list[int]] = (-1, [])
        # History as of each save, keyed by file path; reloading the unchanged
        # file brings it back: {path: (mtime, undo, redo)}.
        self._saved_history: dict[str, tuple[float, list, list]] = {}

    # --- Introspection ---------------------------------------------------
    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    def __contains__(self, frame_number: object) -> bool:
        with self._lock:
            return frame_number in self._frames

    def frames(self) -> list[int]:
        with self._lock:
            return sorted(self._frames)

    def face_count(self) -> int:
        with self._lock:
            return sum(len(f.faces) for f in self._frames.values())

    @property
    def revision(self) -> int:
        """Bumped by every change; equal values mean nothing changed in between."""
        with self._lock:
            return self._revision

    def face_frames(self) -> list[int]:
        """Frames with stored faces, sorted."""
        with self._lock:
            return sorted(n for n, f in self._frames.items() if f.faces)

    def nearest_frame(self, frame_number: int, step: int) -> Optional[int]:
        """Closest frame with stored faces before (step < 0) or after (step > 0)."""
        return _nearest(self.face_frames(), frame_number, step)

    def edited_frames(self) -> list[int]:
        """Frames holding manual edits (alignments or painted masks), sorted."""
        with self._lock:
            if self._edited_index[0] != self._revision:
                self._edited_index = (
                    self._revision,
                    sorted(n for n, f in self._frames.items() if f.is_pinned),
                )
            return self._edited_index[1]

    def nearest_edited(self, frame_number: int, step: int) -> Optional[int]:
        """Closest manually edited frame before (step < 0) or after (step > 0)."""
        return _nearest(self.edited_frames(), frame_number, step)

    def frames_to_check(self, jump: float = 0.25, max_gap: int = 3) -> list[int]:
        """Frames worth reviewing: a face's landmarks jump by more than ``jump`` x
        its size from the previous frame, or a face vanishes for up to ``max_gap``
        stored frames. Only faces matched to a target face (face_id) are checked."""
        with self._lock:
            stored = set(self._frames)
            items = sorted(
                (n, [(f.face_id, f.bbox, f.kps_5) for f in frame.faces])
                for n, frame in self._frames.items()
            )
        flagged: set[int] = set()
        last: dict[str, tuple[int, np.ndarray]] = {}
        for n, faces in items:
            for face_id, bbox, kps in faces:
                if face_id is None:
                    continue
                previous = last.get(face_id)
                last[face_id] = (n, kps)
                if previous is None:
                    continue
                prev_n, prev_kps = previous
                if n == prev_n + 1:
                    size = max(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]), 1e-6)
                    moved = np.linalg.norm(
                        np.asarray(kps[:5]) - np.asarray(prev_kps[:5]), axis=1
                    ).mean()
                    if moved / size > jump:
                        flagged.add(n)
                elif n - prev_n - 1 <= max_gap and all(
                    m in stored for m in range(prev_n + 1, n)
                ):
                    flagged.update(range(prev_n + 1, n))
        return sorted(flagged)

    def matches_media(self, media_path: Optional[str]) -> bool:
        with self._lock:
            return self.media_path is not None and _norm(self.media_path) == _norm(
                media_path
            )

    def has_unsaved_changes(self) -> bool:
        """True if frames are stored that no workspace/job save has written yet."""
        with self._lock:
            return bool(self._frames) and self._revision not in self._saved.values()

    # --- Mutation --------------------------------------------------------
    def clear(self) -> None:
        """Drops every frame. Outside an edit() block this also drops the undo history."""
        with self._lock:
            if self._frames:
                self._revision += 1
            self._frames.clear()
            if self._edit_depth == 0:
                self.clear_history()

    def bind_media(self, media_path: Optional[str]) -> None:
        """Associates the store with a media file, discarding entries for any other."""
        with self._lock:
            if self.media_path is not None and _norm(self.media_path) == _norm(
                media_path
            ):
                return
            self.clear()
            self.clear_history()
            self._edit_before = {}
            self.media_path = media_path
            self._revision += 1

    def rename_media(self, old_path: Optional[str], new_path: str) -> None:
        """Follows a rename of the bound media file, keeping its entries."""
        with self._lock:
            if self.media_path is not None and _norm(self.media_path) == _norm(
                old_path
            ):
                self.media_path = new_path

    def remove_frame(self, frame_number: int) -> None:
        with self._lock:
            if self._frames.pop(int(frame_number), None) is not None:
                self._revision += 1

    def add_face(
        self,
        frame_number: int,
        face: Mapping[str, Any],
        frame_w: int,
        frame_h: int,
    ) -> int:
        """Appends a manual face (pixel coords) to a frame; returns its index."""
        record = AlignmentFace.from_pixels(face, frame_w, frame_h, SOURCE_MANUAL)
        record.source = SOURCE_MANUAL
        with self._lock:
            frame = self._frames.get(int(frame_number))
            if frame is None:
                frame = FrameAlignments("")
                self._frames[int(frame_number)] = frame
            frame.faces.append(record)
            self._revision += 1
            return len(frame.faces) - 1

    def remove_face(self, frame_number: int, index: int) -> bool:
        """Removes one face; the remaining faces become manual so the frame stays as edited."""
        with self._lock:
            frame = self._frames.get(int(frame_number))
            if frame is None or not 0 <= index < len(frame.faces):
                return False
            frame.faces.pop(index)
            for face in frame.faces:
                face.source = SOURCE_MANUAL
            self._revision += 1
            return True

    def put(
        self,
        frame_number: int,
        faces: list[Mapping[str, Any]],
        frame_w: int,
        frame_h: int,
        signature: str,
        source: str = SOURCE_SEQUENTIAL,
    ) -> bool:
        """Stores faces (pixel coords) for a frame. Never overwrites a manual frame
        with an automatic one, nor a sequential frame with a single-frame one."""
        if frame_w <= 0 or frame_h <= 0:
            return False
        try:
            records = [
                AlignmentFace.from_pixels(f, frame_w, frame_h, source) for f in faces
            ]
        except (KeyError, ValueError, TypeError):
            return False
        with self._lock:
            existing = self._frames.get(int(frame_number))
            if existing is not None and source != SOURCE_MANUAL:
                if existing.is_pinned:
                    return False
                if (
                    source == SOURCE_SINGLE
                    and SOURCE_SEQUENTIAL in existing.sources
                    and existing.signature == signature
                ):
                    return False
            self._frames[int(frame_number)] = FrameAlignments(signature, records)
            self._revision += 1
            return True

    def update_face(
        self,
        frame_number: int,
        index: int,
        frame_w: int,
        frame_h: int,
        **fields: Any,
    ) -> bool:
        """Updates fields of one stored face. Spatial fields are given in pixels."""
        with self._lock:
            frame = self._frames.get(int(frame_number))
            if frame is None or not 0 <= index < len(frame.faces):
                return False
            face = frame.faces[index]
            merged = face.to_pixels(frame_w, frame_h)
            merged.update(fields)
            frame.faces[index] = AlignmentFace.from_pixels(
                merged, frame_w, frame_h, merged.get("source", face.source)
            )
            self._revision += 1
            return True

    # --- Undo / redo -----------------------------------------------------
    @contextmanager
    def edit(self, frames: Optional[Iterable[int]] = None) -> Iterator[None]:
        """Records what the block changes on ``frames`` (None: every frame) as one
        undo step. Nested blocks join the outermost step."""
        with self._lock:
            if self._edit_depth == 0:
                self._edit_whole = False
                self._edit_before = {}
                self._edit_revision = self._revision
            self._edit_depth += 1
            if frames is None:
                if not self._edit_whole:
                    before = {n: _copy_frame(f) for n, f in self._frames.items()}
                    before.update(self._edit_before)
                    self._edit_before = before
                    self._edit_whole = True
            elif not self._edit_whole:
                for n in frames:
                    if int(n) not in self._edit_before:
                        self._edit_before[int(n)] = _copy_frame(
                            self._frames.get(int(n))
                        )
        try:
            yield
        finally:
            with self._lock:
                self._edit_depth -= 1
                if self._edit_depth == 0:
                    self._finish_edit()

    def _finish_edit(self) -> None:
        before, self._edit_before = self._edit_before, {}
        if self._revision == self._edit_revision or not (before or self._edit_whole):
            return
        keys = list(self._frames) if self._edit_whole else list(before)
        after = {n: _copy_frame(self._frames.get(n)) for n in keys}
        self._undo.append(_EditStep(self._edit_whole, before, after))
        del self._undo[:-UNDO_LIMIT]
        self._redo.clear()

    def _apply(self, frames: dict[int, Optional[FrameAlignments]], whole: bool) -> None:
        if whole:
            self._frames = {}
        for n, frame in frames.items():
            if frame is None:
                self._frames.pop(n, None)
            else:
                self._frames[n] = _copy_frame(frame)  # type: ignore[assignment]
        self._revision += 1

    def undo(self) -> Optional[list[int]]:
        """Reverts the last edit. Returns the frames it changed ([] for all of
        them), or None if there is nothing to undo."""
        with self._lock:
            if not self._undo:
                return None
            step = self._undo.pop()
            self._apply(step.before, step.whole)
            self._redo.append(step)
            return step.frames

    def redo(self) -> Optional[list[int]]:
        """Re-applies the last undone edit; same return value as undo()."""
        with self._lock:
            if not self._redo:
                return None
            step = self._redo.pop()
            self._apply(step.after, step.whole)
            self._undo.append(step)
            return step.frames

    def can_undo(self) -> bool:
        with self._lock:
            return bool(self._undo)

    def can_redo(self) -> bool:
        with self._lock:
            return bool(self._redo)

    def clear_history(self) -> None:
        with self._lock:
            self._undo.clear()
            self._redo.clear()

    # --- Lookup ----------------------------------------------------------
    def get(
        self,
        frame_number: int,
        frame_w: int,
        frame_h: int,
        signature: Optional[str],
        accept_sources: frozenset[str] = ALL_SOURCES,
    ) -> Optional[list[dict[str, Any]]]:
        """Returns the faces for a frame in pixel coords, or None on a miss."""
        with self._lock:
            frame = self._frames.get(int(frame_number))
            if frame is None:
                return None
            if not frame.is_pinned:
                if signature is not None and frame.signature != signature:
                    return None
                if not frame.sources.issubset(accept_sources):
                    return None
            return [f.to_pixels(frame_w, frame_h) for f in frame.faces]

    # --- Painted masks ---------------------------------------------------
    def _mask_frames(self, layer: str = "mask") -> list[int]:
        cached = self._mask_index.get(layer)
        if cached is None or cached[0] != self._revision:
            cached = (
                self._revision,
                sorted(
                    n
                    for n, frame in self._frames.items()
                    if any(getattr(face, layer) is not None for face in frame.faces)
                ),
            )
            self._mask_index[layer] = cached
        return cached[1]

    def mask_frame_count(self, layer: str = "mask") -> int:
        with self._lock:
            return len(self._mask_frames(layer))

    def effective_mask(
        self,
        frame_number: int,
        face_index: Optional[int] = None,
        face_id: Optional[str] = None,
        layer: str = "mask",
    ) -> Optional[np.ndarray]:
        """The painted mask for a face: its own, else held from the latest earlier
        frame with a mask for the same face_id. None if there is none or it was reset.
        ``layer`` is "mask" (swap mask) or "hair_mask" (Hair Makeup edits)."""
        frame_number = int(frame_number)
        with self._lock:
            frame = self._frames.get(frame_number)
            if (
                frame is not None
                and face_index is not None
                and 0 <= face_index < len(frame.faces)
            ):
                own = frame.faces[face_index]
                if getattr(own, f"{layer}_skip"):
                    return None
                own_mask = getattr(own, layer)
                if own_mask is not None:
                    return None if is_empty(own_mask) else own_mask
                if face_id is None:
                    face_id = own.face_id
            if face_id is None:
                return None
            masked = self._mask_frames(layer)
            for n in reversed(masked[: bisect.bisect_right(masked, frame_number)]):
                if n == frame_number and face_index is not None:
                    continue
                held = self._frames.get(n)
                if held is None:
                    continue
                for face in held.faces:
                    held_mask = getattr(face, layer)
                    if face.face_id == face_id and held_mask is not None:
                        return None if is_empty(held_mask) else held_mask
            return None

    def clear_masks(self, layer: str = "mask") -> int:
        """Drops every painted mask of a layer; returns how many faces had one."""
        with self._lock:
            cleared = 0
            for frame in self._frames.values():
                for face in frame.faces:
                    if getattr(face, layer) is not None or getattr(
                        face, f"{layer}_skip"
                    ):
                        setattr(face, layer, None)
                        setattr(face, f"{layer}_skip", False)
                        cleared += 1
            if cleared:
                self._revision += 1
            return cleared

    # --- Copy / persistence ---------------------------------------------
    # --- Cached hair masks ----------------------------------------------
    def set_hair_prob(
        self,
        frame_number: int,
        index: int,
        prob: np.ndarray,
        crop_key: Iterable[float],
    ) -> bool:
        """Caches one face's hair probability. A different makeup crop drops
        every cached map, since they no longer line up."""
        crop = tuple(float(v) for v in crop_key)
        prob = _as_hair_prob(prob)  # type: ignore[assignment]
        if prob is None:
            return False
        with self._lock:
            frame = self._frames.get(int(frame_number))
            if frame is None or not 0 <= index < len(frame.faces):
                return False
            if self.hair_crop != crop:
                self._drop_hair_probs()
                self.hair_crop = crop
            frame.faces[index].hair_prob = prob
            self._revision += 1
            return True

    def smoothed_hair_prob(
        self,
        frame_number: int,
        index: int,
        face_key: Optional[str],
        crop_key: Iterable[float],
        radius: int,
    ) -> Optional[np.ndarray]:
        """Gaussian-weighted average (0..1 float) of this face's cached hair
        probability over +-radius frames. None if this frame has none cached."""
        crop = tuple(float(v) for v in crop_key)
        frame_number = int(frame_number)
        with self._lock:
            if self.hair_crop != crop:
                return None
            frame = self._frames.get(frame_number)
            if frame is None or not 0 <= index < len(frame.faces):
                return None
            own = frame.faces[index]
            if own.hair_prob is None:
                return None
            key = face_key if face_key is not None else own.face_id
            total = own.hair_prob.astype(np.float32)
            weight = 1.0
            radius = max(0, int(radius))
            sigma = max(radius / 2.0, 0.5)
            for offset in range(1, radius + 1):
                w = math.exp(-(offset * offset) / (2.0 * sigma * sigma))
                for neighbour in (frame_number - offset, frame_number + offset):
                    other = self._frames.get(neighbour)
                    if other is None:
                        continue
                    prob = _matching_hair_prob(other, key, index)
                    if prob is not None:
                        total += w * prob
                        weight += w
            return total / (weight * 255.0)

    def hair_frame_count(self) -> int:
        with self._lock:
            return sum(
                1
                for frame in self._frames.values()
                if any(face.hair_prob is not None for face in frame.faces)
            )

    def clear_hair_probs(self) -> None:
        with self._lock:
            self._drop_hair_probs()
            self._revision += 1

    def _drop_hair_probs(self) -> None:
        for frame in self._frames.values():
            for face in frame.faces:
                face.hair_prob = None

    def snapshot(self) -> "AlignmentStore":
        with self._lock:
            clone = AlignmentStore()
            clone._frames = copy.deepcopy(self._frames)
            clone.media_path = self.media_path
            clone.hair_crop = self.hair_crop
            return clone

    def restore(self, other: "AlignmentStore") -> None:
        with other._lock:
            frames = copy.deepcopy(other._frames)
            media_path = other.media_path
            hair_crop = other.hair_crop
        with self._lock:
            self._frames = frames
            self.media_path = media_path
            self.hair_crop = hair_crop
            self._revision += 1
            self.clear_history()

    def save(self, path: str, mark_saved: bool = True) -> dict[str, Any]:
        """Writes the store to a compressed .npz sidecar and returns the header
        dict to embed in the workspace/job JSON. Skips the write if nothing
        changed since the last save to the same path. With mark_saved=False
        (autosave) the store still counts as unsaved."""
        with self._lock:
            path = os.path.abspath(path)
            header = self._header(path)
            if self._saved.get(path) == self._revision and os.path.exists(path):
                self._remember_history(path)
                return header
            arrays = self._pack()
            revision = self._revision
        tmp_path = path + ".tmp.npz"
        np.savez_compressed(tmp_path, **arrays)  # type: ignore[arg-type]
        os.replace(tmp_path, path)
        if mark_saved:
            with self._lock:
                self._saved[path] = revision
                if self._revision == revision:
                    self._remember_history(path)
        return header

    def _remember_history(self, path: str) -> None:
        self._saved_history.pop(path, None)
        if self._undo or self._redo:
            self._saved_history[path] = (
                os.path.getmtime(path),
                list(self._undo),
                list(self._redo),
            )
            for old in list(self._saved_history)[:-SAVED_HISTORY_FILES]:
                del self._saved_history[old]

    def load(self, path: str, mark_saved: bool = True) -> None:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if int(meta.get("version", 0)) > ALIGNMENTS_FORMAT_VERSION:
                raise ValueError(
                    f"Alignments file version {meta.get('version')} is newer than supported"
                )
            frames = _unpack(data)
        hair_crop = meta.get("hair_crop")
        with self._lock:
            self._frames = frames
            self.media_path = meta.get("media_path")
            self.hair_crop = tuple(float(v) for v in hair_crop) if hair_crop else None
            self._revision += 1
            self.clear_history()
            path = os.path.abspath(path)
            if mark_saved:
                self._saved[path] = self._revision
            remembered = self._saved_history.get(path)
            if remembered is not None and remembered[0] == os.path.getmtime(path):
                self._undo, self._redo = list(remembered[1]), list(remembered[2])

    def load_from_header(
        self, header: Optional[Mapping[str, Any]], base_dir: Optional[str] = None
    ) -> bool:
        """Loads the sidecar referenced by a workspace/job 'alignments' header."""
        if not isinstance(header, Mapping):
            return False
        candidates = []
        if header.get("file"):
            candidates.append(str(header["file"]))
        if base_dir and header.get("file_name"):
            candidates.append(os.path.join(base_dir, str(header["file_name"])))
        for candidate in candidates:
            if os.path.isfile(candidate):
                try:
                    self.load(candidate)
                except Exception as e:  # corrupt sidecar must not block the load
                    print(f"[WARN] Could not load alignments from {candidate}: {e}")
                    return False
                if header.get("media_path") and not self.media_path:
                    self.media_path = str(header["media_path"])
                print(
                    f"[INFO] Loaded alignments for {len(self)} frames from {candidate}"
                )
                return True
        if candidates:
            print(f"[WARN] Alignments file not found: {candidates[0]}")
        return False

    def _header(self, path: str) -> dict[str, Any]:
        return {
            "version": ALIGNMENTS_FORMAT_VERSION,
            "file": path,
            "file_name": os.path.basename(path),
            "media_path": self.media_path,
            "frames": len(self._frames),
            "faces": sum(len(f.faces) for f in self._frames.values()),
        }

    def _pack(self) -> dict[str, np.ndarray]:
        frame_numbers = sorted(self._frames)
        signatures: list[str] = []
        face_frame: list[int] = []
        sources: list[str] = []
        face_ids: list[str] = []
        assigned_ids: list[str] = []
        bboxes: list[np.ndarray] = []
        kps5: list[np.ndarray] = []
        dense_offsets = [0]
        dense: list[np.ndarray] = []
        kps203_index: list[int] = []
        kps203: list[np.ndarray] = []
        emb_offsets = [0]
        embs: list[np.ndarray] = []
        mask_index: list[int] = []
        mask_skip: list[bool] = []
        masks: list[np.ndarray] = []
        hair_mask_index: list[int] = []
        hair_mask_skip: list[bool] = []
        hair_masks: list[np.ndarray] = []
        hair_index: list[int] = []
        hair_probs: list[np.ndarray] = []

        for fi, frame_number in enumerate(frame_numbers):
            frame = self._frames[frame_number]
            signatures.append(frame.signature)
            for face in frame.faces:
                face_frame.append(fi)
                sources.append(face.source)
                face_ids.append(face.face_id or "")
                assigned_ids.append(face.assigned_face_id or "")
                bboxes.append(face.bbox)
                kps5.append(face.kps_5)
                if face.kps_all is not None:
                    dense.append(face.kps_all)
                    dense_offsets.append(dense_offsets[-1] + len(face.kps_all))
                else:
                    dense_offsets.append(dense_offsets[-1])
                if face.kps_203 is not None and face.kps_203.shape == (203, 2):
                    kps203_index.append(len(kps203))
                    kps203.append(face.kps_203)
                else:
                    kps203_index.append(-1)
                if face.embedding is not None:
                    embs.append(face.embedding)
                    emb_offsets.append(emb_offsets[-1] + len(face.embedding))
                else:
                    emb_offsets.append(emb_offsets[-1])
                mask_skip.append(face.mask_skip)
                if face.mask is not None:
                    mask_index.append(len(masks))
                    masks.append(face.mask)
                else:
                    mask_index.append(-1)
                hair_mask_skip.append(face.hair_mask_skip)
                if face.hair_mask is not None:
                    hair_mask_index.append(len(hair_masks))
                    hair_masks.append(face.hair_mask)
                else:
                    hair_mask_index.append(-1)
                if face.hair_prob is not None:
                    hair_index.append(len(hair_probs))
                    hair_probs.append(face.hair_prob)
                else:
                    hair_index.append(-1)

        meta = {
            "version": ALIGNMENTS_FORMAT_VERSION,
            "media_path": self.media_path,
            "hair_crop": list(self.hair_crop) if self.hair_crop else None,
        }
        return {
            "meta": np.array(json.dumps(meta)),
            "frame_numbers": np.asarray(frame_numbers, dtype=np.int64),
            "frame_signatures": np.asarray(signatures, dtype=np.str_),
            "face_frame": np.asarray(face_frame, dtype=np.int64),
            "face_source": np.asarray(sources, dtype=np.str_),
            "face_id": np.asarray(face_ids, dtype=np.str_),
            "assigned_face_id": np.asarray(assigned_ids, dtype=np.str_),
            "bbox": np.asarray(bboxes, dtype=np.float32).reshape(-1, 4),
            "kps_5": np.asarray(kps5, dtype=np.float32).reshape(-1, 5, 2),
            "kps_all_offsets": np.asarray(dense_offsets, dtype=np.int64),
            "kps_all": (
                np.concatenate(dense).astype(np.float32)
                if dense
                else np.empty((0, 2), dtype=np.float32)
            ),
            "kps_203_index": np.asarray(kps203_index, dtype=np.int64),
            "kps_203": np.asarray(kps203, dtype=np.float32).reshape(-1, 203, 2),
            "embedding_offsets": np.asarray(emb_offsets, dtype=np.int64),
            "embedding": (
                np.concatenate(embs).astype(np.float16)
                if embs
                else np.empty((0,), dtype=np.float16)
            ),
            "mask_index": np.asarray(mask_index, dtype=np.int64),
            "mask_skip": np.asarray(mask_skip, dtype=bool),
            "masks": (
                np.stack(masks).astype(np.uint8)
                if masks
                else np.empty((0, 2, 256, 256), dtype=np.uint8)
            ),
            "hair_mask_index": np.asarray(hair_mask_index, dtype=np.int64),
            "hair_mask_skip": np.asarray(hair_mask_skip, dtype=bool),
            "hair_masks": (
                np.stack(hair_masks).astype(np.uint8)
                if hair_masks
                else np.empty((0, 2, HAIR_SPACE.size, HAIR_SPACE.size), np.uint8)
            ),
            "hair_index": np.asarray(hair_index, dtype=np.int64),
            "hair_probs": (
                np.stack(hair_probs).astype(np.uint8)
                if hair_probs
                else np.empty((0, HAIR_PROB_SIZE, HAIR_PROB_SIZE), dtype=np.uint8)
            ),
        }


def _unpack(data: Any) -> dict[int, FrameAlignments]:
    frame_numbers = data["frame_numbers"]
    signatures = data["frame_signatures"]
    frames: dict[int, FrameAlignments] = {
        int(n): FrameAlignments(str(s)) for n, s in zip(frame_numbers, signatures)
    }
    dense_offsets = data["kps_all_offsets"]
    emb_offsets = data["embedding_offsets"]
    kps203_index = data["kps_203_index"]
    has_masks = "mask_index" in data.files
    assigned_ids = (
        data["assigned_face_id"] if "assigned_face_id" in data.files else None
    )
    mask_index = data["mask_index"] if has_masks else None
    masks = data["masks"] if has_masks else None
    mask_skip = data["mask_skip"] if "mask_skip" in data.files else None
    has_hair_masks = "hair_mask_index" in data.files
    hair_mask_index = data["hair_mask_index"] if has_hair_masks else None
    hair_masks = data["hair_masks"] if has_hair_masks else None
    hair_mask_skip = data["hair_mask_skip"] if has_hair_masks else None
    has_hair = "hair_index" in data.files
    hair_index = data["hair_index"] if has_hair else None
    hair_probs = data["hair_probs"] if has_hair else None
    for i, fi in enumerate(data["face_frame"]):
        d0, d1 = int(dense_offsets[i]), int(dense_offsets[i + 1])
        e0, e1 = int(emb_offsets[i]), int(emb_offsets[i + 1])
        k = int(kps203_index[i])
        face_id = str(data["face_id"][i])
        assigned_id = str(assigned_ids[i]) if assigned_ids is not None else ""
        m = int(mask_index[i]) if mask_index is not None else -1
        h = int(hair_index[i]) if hair_index is not None else -1
        hm = int(hair_mask_index[i]) if hair_mask_index is not None else -1
        frames[int(frame_numbers[fi])].faces.append(
            AlignmentFace(
                bbox=data["bbox"][i].astype(np.float32),
                kps_5=data["kps_5"][i].astype(np.float32),
                kps_all=data["kps_all"][d0:d1].astype(np.float32) if d1 > d0 else None,
                kps_203=data["kps_203"][k].astype(np.float32) if k >= 0 else None,
                embedding=data["embedding"][e0:e1].astype(np.float32)
                if e1 > e0
                else None,
                face_id=face_id or None,
                assigned_face_id=assigned_id or None,
                source=str(data["face_source"][i]),
                mask=as_mask(masks[m]) if masks is not None and m >= 0 else None,
                mask_skip=bool(mask_skip[i]) if mask_skip is not None else False,
                hair_mask=as_mask(hair_masks[hm], HAIR_SPACE)
                if hair_masks is not None and hm >= 0
                else None,
                hair_mask_skip=bool(hair_mask_skip[i])
                if hair_mask_skip is not None
                else False,
                hair_prob=_as_hair_prob(hair_probs[h])
                if hair_probs is not None and h >= 0
                else None,
            )
        )
    return frames


def _norm(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return os.path.normcase(os.path.normpath(os.path.abspath(str(path))))
