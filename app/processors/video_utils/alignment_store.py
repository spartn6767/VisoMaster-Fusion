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

import copy
import hashlib
import json
import os
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

ALIGNMENTS_FORMAT_VERSION = 1
SIDECAR_SUFFIX = ".alignments.npz"

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
    source: str = SOURCE_SEQUENTIAL

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
            "source": self.source,
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
            source=str(face.get("source", source)),
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
        self._lock = threading.RLock()
        self._frames: dict[int, FrameAlignments] = {}
        self.media_path: Optional[str] = None
        self._revision = 0
        self._saved: dict[str, int] = {}

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

    def matches_media(self, media_path: Optional[str]) -> bool:
        with self._lock:
            return self.media_path is not None and _norm(self.media_path) == _norm(
                media_path
            )

    # --- Mutation --------------------------------------------------------
    def clear(self) -> None:
        with self._lock:
            if self._frames:
                self._revision += 1
            self._frames.clear()

    def bind_media(self, media_path: Optional[str]) -> None:
        """Associates the store with a media file, discarding entries for any other."""
        with self._lock:
            if self.media_path is not None and _norm(self.media_path) == _norm(
                media_path
            ):
                return
            self.clear()
            self.media_path = media_path
            self._revision += 1

    def remove_frame(self, frame_number: int) -> None:
        with self._lock:
            if self._frames.pop(int(frame_number), None) is not None:
                self._revision += 1

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

    # --- Copy / persistence ---------------------------------------------
    def snapshot(self) -> "AlignmentStore":
        with self._lock:
            clone = AlignmentStore()
            clone._frames = copy.deepcopy(self._frames)
            clone.media_path = self.media_path
            return clone

    def restore(self, other: "AlignmentStore") -> None:
        with other._lock:
            frames = copy.deepcopy(other._frames)
            media_path = other.media_path
        with self._lock:
            self._frames = frames
            self.media_path = media_path
            self._revision += 1

    def save(self, path: str) -> dict[str, Any]:
        """Writes the store to a compressed .npz sidecar and returns the header
        dict to embed in the workspace/job JSON. Skips the write if nothing
        changed since the last save to the same path."""
        with self._lock:
            path = os.path.abspath(path)
            header = self._header(path)
            if self._saved.get(path) == self._revision and os.path.exists(path):
                return header
            arrays = self._pack()
            revision = self._revision
        tmp_path = path + ".tmp.npz"
        np.savez_compressed(tmp_path, **arrays)  # type: ignore[arg-type]
        os.replace(tmp_path, path)
        with self._lock:
            self._saved[path] = revision
        return header

    def load(self, path: str) -> None:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if int(meta.get("version", 0)) > ALIGNMENTS_FORMAT_VERSION:
                raise ValueError(
                    f"Alignments file version {meta.get('version')} is newer than supported"
                )
            frames = _unpack(data)
        with self._lock:
            self._frames = frames
            self.media_path = meta.get("media_path")
            self._revision += 1
            self._saved[os.path.abspath(path)] = self._revision

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
        bboxes: list[np.ndarray] = []
        kps5: list[np.ndarray] = []
        dense_offsets = [0]
        dense: list[np.ndarray] = []
        kps203_index: list[int] = []
        kps203: list[np.ndarray] = []
        emb_offsets = [0]
        embs: list[np.ndarray] = []

        for fi, frame_number in enumerate(frame_numbers):
            frame = self._frames[frame_number]
            signatures.append(frame.signature)
            for face in frame.faces:
                face_frame.append(fi)
                sources.append(face.source)
                face_ids.append(face.face_id or "")
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

        meta = {"version": ALIGNMENTS_FORMAT_VERSION, "media_path": self.media_path}
        return {
            "meta": np.array(json.dumps(meta)),
            "frame_numbers": np.asarray(frame_numbers, dtype=np.int64),
            "frame_signatures": np.asarray(signatures, dtype=np.str_),
            "face_frame": np.asarray(face_frame, dtype=np.int64),
            "face_source": np.asarray(sources, dtype=np.str_),
            "face_id": np.asarray(face_ids, dtype=np.str_),
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
    for i, fi in enumerate(data["face_frame"]):
        d0, d1 = int(dense_offsets[i]), int(dense_offsets[i + 1])
        e0, e1 = int(emb_offsets[i]), int(emb_offsets[i + 1])
        k = int(kps203_index[i])
        face_id = str(data["face_id"][i])
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
                source=str(data["face_source"][i]),
            )
        )
    return frames


def _norm(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return os.path.normcase(os.path.normpath(os.path.abspath(str(path))))
