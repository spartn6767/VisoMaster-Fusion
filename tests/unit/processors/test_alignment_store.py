"""Unit tests for app.processors.video_utils.alignment_store."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from app.processors.video_utils.alignment_store import (
    ALL_SOURCES,
    SEQUENTIAL_SOURCES,
    SOURCE_MANUAL,
    SOURCE_SEQUENTIAL,
    SOURCE_SINGLE,
    AlignmentStore,
    alignments_cache_enabled,
    detector_signature,
    faces_require_203,
    faces_to_detection_arrays,
    sidecar_path_for,
)
from app.processors.video_utils.media_pipeline import MediaPipeline

W, H = 1920, 1080


def _face(offset: float = 0.0, with_203: bool = True, face_id: str | None = "7"):
    rng = np.random.default_rng(int(offset) + 1)
    return {
        "bbox": np.array([100 + offset, 200, 300 + offset, 450], dtype=np.float32),
        "kps_5": rng.uniform(100, 400, (5, 2)).astype(np.float32),
        "kps_all": rng.uniform(100, 400, (68, 2)).astype(np.float32),
        "kps_203": rng.uniform(100, 400, (203, 2)).astype(np.float32)
        if with_203
        else None,
        "embedding": rng.standard_normal(512).astype(np.float32),
        "face_id": face_id,
    }


def test_put_get_rescales_to_requested_frame_size():
    store = AlignmentStore()
    face = _face()
    assert store.put(10, [face], W, H, "sig")

    same = store.get(10, W, H, "sig")
    assert same is not None and len(same) == 1
    np.testing.assert_allclose(same[0]["bbox"], face["bbox"], rtol=1e-5)
    np.testing.assert_allclose(same[0]["kps_203"], face["kps_203"], rtol=1e-5)
    assert same[0]["face_id"] == "7"
    assert same[0]["source"] == SOURCE_SEQUENTIAL

    half = store.get(10, W // 2, H // 2, "sig")
    assert half is not None
    np.testing.assert_allclose(half[0]["kps_5"], face["kps_5"] / 2, rtol=1e-5)


def test_miss_on_unknown_frame_and_signature_mismatch():
    store = AlignmentStore()
    store.put(1, [_face()], W, H, "sig-a")
    assert store.get(2, W, H, "sig-a") is None
    assert store.get(1, W, H, "sig-b") is None


def test_sequential_lookup_rejects_single_frame_entries():
    store = AlignmentStore()
    store.put(3, [_face()], W, H, "sig", source=SOURCE_SINGLE)
    assert store.get(3, W, H, "sig", SEQUENTIAL_SOURCES) is None
    assert store.get(3, W, H, "sig", ALL_SOURCES) is not None


def test_single_frame_does_not_replace_matching_sequential_entry():
    store = AlignmentStore()
    store.put(4, [_face(0)], W, H, "sig", source=SOURCE_SEQUENTIAL)
    assert not store.put(4, [_face(50)], W, H, "sig", source=SOURCE_SINGLE)
    assert store.put(4, [_face(50)], W, H, "other", source=SOURCE_SINGLE)


def test_manual_entries_are_pinned():
    store = AlignmentStore()
    manual = dict(_face(), source=SOURCE_MANUAL)
    assert store.put(5, [manual], W, H, "old", source=SOURCE_MANUAL)
    assert not store.put(5, [_face(30)], W, H, "new")
    hit = store.get(5, W, H, "new", SEQUENTIAL_SOURCES)
    assert hit is not None and hit[0]["source"] == SOURCE_MANUAL


def test_empty_frame_is_a_hit():
    store = AlignmentStore()
    store.put(6, [], W, H, "sig")
    assert store.get(6, W, H, "sig") == []


def test_update_face_merges_fields():
    store = AlignmentStore()
    store.put(7, [_face(with_203=False)], W, H, "sig")
    new_203 = np.full((203, 2), 50.0, dtype=np.float32)
    assert store.update_face(7, 0, W, H, kps_203=new_203)
    hit = store.get(7, W, H, "sig")
    assert hit is not None
    np.testing.assert_allclose(hit[0]["kps_203"], new_203, rtol=1e-5)
    assert not store.update_face(7, 3, W, H, kps_203=new_203)


def test_bind_media_clears_only_on_change(tmp_path):
    store = AlignmentStore()
    video = str(tmp_path / "a.mp4")
    store.bind_media(video)
    store.put(1, [_face()], W, H, "sig")
    store.bind_media(os.path.join(str(tmp_path), ".", "a.mp4"))
    assert len(store) == 1
    assert store.matches_media(video)
    store.bind_media(str(tmp_path / "b.mp4"))
    assert len(store) == 0
    assert not store.matches_media(video)


def test_snapshot_restore_is_independent():
    store = AlignmentStore()
    store.put(1, [_face()], W, H, "sig")
    snap = store.snapshot()
    store.clear()
    assert len(store) == 0
    store.restore(snap)
    assert store.get(1, W, H, "sig") is not None


def test_save_load_round_trip(tmp_path):
    store = AlignmentStore()
    store.bind_media("/videos/clip.mp4")
    faces = [_face(0), _face(40, with_203=False, face_id=None)]
    store.put(0, faces, W, H, "sig")
    store.put(9, [], W, H, "sig")
    store.put(12, [dict(_face(80), source=SOURCE_MANUAL)], W, H, "x", SOURCE_MANUAL)

    sidecar = sidecar_path_for(str(tmp_path / "job.json"))
    assert sidecar.endswith("job.alignments.npz")
    header = store.save(sidecar)
    json.dumps(header)
    assert header["frames"] == 3 and header["faces"] == 3

    loaded = AlignmentStore()
    assert loaded.load_from_header(header)
    assert loaded.frames() == [0, 9, 12]
    assert loaded.matches_media("/videos/clip.mp4")
    got = loaded.get(0, W, H, "sig")
    assert got is not None and len(got) == 2
    np.testing.assert_allclose(got[0]["bbox"], faces[0]["bbox"], atol=0.05)
    np.testing.assert_allclose(got[0]["kps_203"], faces[0]["kps_203"], atol=0.05)
    np.testing.assert_allclose(got[0]["embedding"], faces[0]["embedding"], atol=1e-2)
    assert got[0]["embedding"].dtype == np.float32
    assert got[1]["kps_203"] is None and got[1]["face_id"] is None
    assert loaded.get(9, W, H, "sig") == []
    manual = loaded.get(12, W, H, "anything")
    assert manual is not None and manual[0]["source"] == SOURCE_MANUAL


def test_load_from_header_falls_back_to_file_name(tmp_path):
    store = AlignmentStore()
    store.put(0, [_face()], W, H, "sig")
    header = store.save(str(tmp_path / "w.alignments.npz"))
    header = dict(header, file=str(tmp_path / "moved" / "w.alignments.npz"))

    loaded = AlignmentStore()
    assert loaded.load_from_header(header, str(tmp_path))
    assert len(loaded) == 1
    assert not AlignmentStore().load_from_header(header)
    assert not AlignmentStore().load_from_header(None)


def test_corrupt_sidecar_does_not_raise(tmp_path):
    bad = tmp_path / "bad.alignments.npz"
    bad.write_bytes(b"not a zip")
    assert not AlignmentStore().load_from_header({"file": str(bad)})


def test_faces_to_detection_arrays_shapes():
    bboxes, kpss_5, kpss, kpss_203 = faces_to_detection_arrays([])
    assert bboxes.shape == (0, 4) and kpss_5.shape == (0, 5, 2)
    assert kpss.shape[0] == 0 and kpss_203.shape[0] == 0

    store = AlignmentStore()
    store.put(0, [_face(0), _face(10)], W, H, "sig")
    bboxes, kpss_5, kpss, kpss_203 = faces_to_detection_arrays(
        store.get(0, W, H, "sig")  # type: ignore[arg-type]
    )
    assert bboxes.shape == (2, 4)
    assert kpss_5.shape == (2, 5, 2)
    assert kpss.shape == (2, 68, 2)
    assert kpss_203.shape == (2, 203, 2)

    store.put(1, [_face(0), _face(10, with_203=False)], W, H, "sig")
    *_, kpss_203 = faces_to_detection_arrays(store.get(1, W, H, "sig"))  # type: ignore[arg-type]
    assert kpss_203.shape[0] == 0


def test_detector_signature_tracks_settings_and_targets():
    control = {"DetectorModelSelection": "RetinaFace", "DetectorScoreSlider": 50}
    base = detector_signature(control, ["1", "2"])
    assert base == detector_signature(dict(control), ["2", "1"])
    assert base != detector_signature(control, ["1"])
    assert base != detector_signature(dict(control, DetectorScoreSlider=60), ["1", "2"])
    assert base == detector_signature(dict(control, SomeUnrelatedSlider=3), ["1", "2"])


def test_alignments_cache_enabled():
    assert alignments_cache_enabled({})
    assert not alignments_cache_enabled({"AlignmentsCacheEnableToggle": False})
    assert not alignments_cache_enabled({"VR180ModeEnableToggle": True})
    assert not alignments_cache_enabled({"ManualRotationEnableToggle": True})


def test_faces_require_203():
    assert not faces_require_203({}, True)
    assert faces_require_203({"1": {"FaceExpressionEnableBothToggle": True}}, False)
    assert not faces_require_203({"1": {"FaceEditorEnableToggle": True}}, False)
    assert faces_require_203({"1": {"FaceEditorEnableToggle": True}}, True)


@pytest.mark.parametrize(
    "file_type, media, enabled, expect_hit",
    [
        ("video", "/v.mp4", True, True),
        ("video", "/other.mp4", True, False),
        ("webcam", "/v.mp4", True, False),
        ("video", "/v.mp4", False, False),
    ],
)
def test_media_pipeline_cache_lookup(file_type, media, enabled, expect_hit):
    store = AlignmentStore()
    store.bind_media("/v.mp4")
    control = {"AlignmentsCacheEnableToggle": enabled}
    store.put(3, [_face()], W, H, detector_signature(control, ["1"]))

    fake = SimpleNamespace(
        main_window=SimpleNamespace(alignments=store, target_faces={"1": object()}),
        vp=SimpleNamespace(file_type=file_type, media_path=media),
    )
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    hit = MediaPipeline._lookup_cached_alignments(fake, 3, frame, control)  # type: ignore[arg-type]
    assert (hit is not None) is expect_hit


def test_workspace_sidecar_helpers_round_trip(tmp_path):
    from app.ui.widgets.actions.save_load_actions import (
        load_alignments_sidecar,
        save_alignments_sidecar,
    )

    json_path = tmp_path / "workspace.json"
    assert save_alignments_sidecar(SimpleNamespace(), json_path) is None
    assert (
        save_alignments_sidecar(SimpleNamespace(alignments=AlignmentStore()), json_path)
        is None
    )

    source = SimpleNamespace(alignments=AlignmentStore())
    source.alignments.put(4, [_face()], W, H, "sig")
    header = save_alignments_sidecar(source, json_path)
    assert header is not None
    assert (tmp_path / "workspace.alignments.npz").is_file()

    target = SimpleNamespace(alignments=AlignmentStore())
    load_alignments_sidecar(target, {"alignments": header}, json_path)
    assert target.alignments.frames() == [4]
    load_alignments_sidecar(SimpleNamespace(), {"alignments": header}, json_path)


@pytest.mark.parametrize("confirm, expect_cleared", [(True, True), (False, False)])
def test_clear_alignments_cache_button(monkeypatch, confirm, expect_cleared):
    from PySide6 import QtWidgets

    from app.ui.widgets.actions import control_actions

    buttons = QtWidgets.QMessageBox.StandardButton
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *a, **k: buttons.Yes if confirm else buttons.No,
    )
    refreshed: list = []
    monkeypatch.setattr(
        control_actions.common_widget_actions,
        "refresh_frame",
        lambda mw, *a, **k: refreshed.append(mw),
    )

    main_window = SimpleNamespace(alignments=AlignmentStore())
    main_window.alignments.put(1, [_face()], W, H, "sig")
    control_actions.clear_alignments_cache(main_window, False)

    assert (len(main_window.alignments) == 0) is expect_cleared
    assert bool(refreshed) is expect_cleared


def test_nearest_frame_skips_gaps_and_empty_frames():
    store = AlignmentStore()
    face = {"bbox": [0.1, 0.1, 0.5, 0.5], "kps_5": [[0.2, 0.2]] * 5}
    for n in (3, 10, 20):
        store.put(n, [face], 1, 1, "sig")
    store.put(15, [], 1, 1, "sig")
    assert store.nearest_frame(10, -1) == 3
    assert store.nearest_frame(10, 1) == 20
    assert store.nearest_frame(12, 1) == 20
    assert store.nearest_frame(12, -1) == 10
    assert store.nearest_frame(3, -1) is None
    assert store.nearest_frame(20, 1) is None


def test_rename_media_keeps_entries(tmp_path):
    store = AlignmentStore()
    old = str(tmp_path / "a.mp4")
    new = str(tmp_path / "renamed.mp4")
    store.bind_media(old)
    store.put(1, [_face()], W, H, "sig")
    store.rename_media(str(tmp_path / "other.mp4"), new)
    assert store.matches_media(old)
    store.rename_media(old, new)
    assert store.matches_media(new)
    store.bind_media(new)
    assert len(store) == 1
