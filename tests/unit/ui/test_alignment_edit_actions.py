from types import SimpleNamespace

import numpy as np
import pytest

from app.processors.video_utils.alignment_store import AlignmentStore, SOURCE_MANUAL
from app.ui.widgets.actions import alignment_edit_actions as aea


@pytest.fixture
def main_window(monkeypatch):
    monkeypatch.setattr(aea, "_refresh", lambda mw: None)
    store = AlignmentStore()
    store.bind_media("/v.mp4")
    return SimpleNamespace(
        alignments=store,
        video_processor=SimpleNamespace(media_path="/v.mp4"),
    )


def test_template_kps_inside_bbox():
    kps = aea.template_kps_for_bbox([0.2, 0.2, 0.6, 0.8])
    assert kps.shape == (5, 2)
    assert np.all((kps[:, 0] > 0.2) & (kps[:, 0] < 0.6))
    assert np.all((kps[:, 1] > 0.2) & (kps[:, 1] < 0.8))


def test_bbox_from_corners_normalises_order():
    np.testing.assert_allclose(
        aea.bbox_from_corners((0.5, 0.6), (0.1, 0.2)), [0.1, 0.2, 0.5, 0.6]
    )


def test_face_index_at_prefers_smallest_box():
    faces = [{"bbox": [0, 0, 1, 1]}, {"bbox": [0.4, 0.4, 0.6, 0.6]}]
    assert aea.face_index_at(faces, 0.5, 0.5) == 1
    assert aea.face_index_at(faces, 0.1, 0.1) == 0
    assert aea.face_index_at(faces[1:], 0.1, 0.1) is None


def test_add_manual_face_creates_pinned_refine_entry(main_window):
    idx = aea.add_manual_face(main_window, 7, np.array([0.1, 0.1, 0.3, 0.4]))

    faces = main_window.alignments.get(7, 100, 100, "other-signature")
    assert idx == 0 and faces is not None
    assert faces[0]["source"] == SOURCE_MANUAL and faces[0]["refine"] is True
    np.testing.assert_allclose(faces[0]["bbox"], [10, 10, 30, 40], atol=1e-3)


def test_move_landmark_overrides_point_and_pins(main_window):
    store = main_window.alignments
    face = {
        "bbox": np.array([10, 10, 50, 50], np.float32),
        "kps_5": np.full((5, 2), 20, np.float32),
        "kps_203": np.ones((203, 2), np.float32),
        "embedding": np.ones(4, np.float32),
    }
    store.put(3, [face], 100, 100, "sig", "single")

    assert aea.move_landmark(main_window, 3, 0, 2, 0.4, 0.45)

    stored = store.get(3, 100, 100, "changed-signature")
    assert stored is not None
    assert stored[0]["source"] == SOURCE_MANUAL
    np.testing.assert_allclose(stored[0]["kps_5"][2], [40, 45], atol=1e-3)
    assert stored[0]["kps_203"] is None and stored[0]["embedding"] is None
    assert not aea.move_landmark(main_window, 3, 5, 0, 0.1, 0.1)


def test_delete_face_keeps_remaining_faces_pinned(main_window):
    store = main_window.alignments
    faces = [
        {
            "bbox": np.array([0, 0, 10, 10], np.float32),
            "kps_5": np.ones((5, 2), np.float32),
        },
        {
            "bbox": np.array([20, 20, 30, 30], np.float32),
            "kps_5": np.ones((5, 2), np.float32),
        },
    ]
    store.put(1, faces, 100, 100, "sig")

    aea.delete_face(main_window, 1, 0)

    remaining = store.get(1, 100, 100, "other")
    assert remaining is not None and len(remaining) == 1
    assert remaining[0]["source"] == SOURCE_MANUAL


def test_refine_flag_is_not_persisted(tmp_path):
    store = AlignmentStore()
    store.add_face(
        0,
        {"bbox": [0, 0, 10, 10], "kps_5": np.ones((5, 2)), "refine": True},
        100,
        100,
    )
    path = str(tmp_path / "a.alignments.npz")
    store.save(path)
    loaded = AlignmentStore()
    loaded.load(path)

    faces = loaded.get(0, 100, 100, None)
    assert faces is not None and faces[0]["refine"] is False
    assert faces[0]["source"] == SOURCE_MANUAL


def _face(x1, y1, x2, y2, face_id=None):
    return {
        "bbox": np.array([x1, y1, x2, y2], np.float32),
        "kps_5": np.full((5, 2), (x1 + x2) / 2, np.float32),
        "embedding": np.arange(4, dtype=np.float32),
        "face_id": face_id,
    }


def test_nearest_face_index_falls_back_to_closest_centre():
    faces = [{"bbox": [0, 0, 0.1, 0.1]}, {"bbox": [0.8, 0.8, 0.9, 0.9]}]
    assert aea.nearest_face_index(faces, 0.7, 0.7) == 1
    assert aea.nearest_face_index(faces, 0.05, 0.05) == 0
    assert aea.nearest_face_index([], 0.5, 0.5) is None


def test_copy_previous_and_paste_adds_manual_face(main_window):
    store = main_window.alignments
    store.put(4, [_face(10, 10, 30, 30, "7")], 100, 100, "sig")

    assert aea.copy_face(main_window, 4, 0)
    assert aea.paste_face(main_window, 5)

    pasted = store.get(5, 100, 100, "other")
    assert pasted is not None and len(pasted) == 1
    assert pasted[0]["source"] == SOURCE_MANUAL and pasted[0]["face_id"] == "7"
    np.testing.assert_allclose(pasted[0]["bbox"], [10, 10, 30, 30], atol=1e-3)
    np.testing.assert_allclose(pasted[0]["embedding"], np.arange(4))
    original = store.get(4, 100, 100, "sig")
    assert original is not None and original[0]["source"] == "sequential"


def test_paste_replaces_face_under_cursor(main_window):
    store = main_window.alignments
    store.put(1, [_face(10, 10, 30, 30)], 100, 100, "sig")
    store.put(2, [_face(50, 50, 70, 70), _face(0, 0, 5, 5)], 100, 100, "sig")
    aea.copy_face(main_window, 1, 0)

    assert aea.paste_face(main_window, 2, replace_index=0)

    faces = store.get(2, 100, 100, None)
    assert faces is not None and len(faces) == 2
    np.testing.assert_allclose(faces[0]["bbox"], [10, 10, 30, 30], atol=1e-3)
    assert faces[0]["source"] == SOURCE_MANUAL


def test_paste_from_previous_frame_leaves_clipboard_untouched(main_window):
    store = main_window.alignments
    store.put(4, [_face(10, 10, 30, 30, "7")], 100, 100, "sig")
    store.put(5, [_face(60, 60, 80, 80)], 100, 100, "sig")

    assert aea.paste_from_frame(main_window, 4, 0, 5, replace_index=0)
    faces = store.get(5, 100, 100, "other")
    assert faces is not None and len(faces) == 1
    assert faces[0]["source"] == SOURCE_MANUAL and faces[0]["face_id"] == "7"
    np.testing.assert_allclose(faces[0]["bbox"], [10, 10, 30, 30], atol=1e-3)
    assert getattr(main_window, aea._CLIPBOARD_ATTR, None) is None

    assert aea.paste_from_frame(main_window, 4, 0, 6)
    assert len(store.get(6, 100, 100, "other") or []) == 1
    assert not aea.paste_from_frame(main_window, 9, 0, 6)


def test_paste_without_copy_does_nothing(main_window):
    assert not aea.paste_face(main_window, 0)
    assert len(main_window.alignments) == 0


def test_move_face_translates_bbox_and_landmarks(main_window):
    store = main_window.alignments
    face = _face(10, 10, 30, 30)
    face["kps_203"] = np.full((203, 2), 20, np.float32)
    store.put(0, [face], 100, 100, "sig")

    assert aea.move_face(main_window, 0, 0, 0.05, -0.02)

    moved = store.get(0, 100, 100, "other")
    assert moved is not None and moved[0]["source"] == SOURCE_MANUAL
    np.testing.assert_allclose(moved[0]["bbox"], [15, 8, 35, 28], atol=1e-3)
    np.testing.assert_allclose(moved[0]["kps_5"][0], [25, 18], atol=1e-3)
    np.testing.assert_allclose(moved[0]["kps_203"][0], [25, 18], atol=1e-3)
    assert moved[0]["embedding"] is None
    assert not aea.move_face(main_window, 0, 3, 0.1, 0.1)


def test_resize_face_stretches_landmarks_with_bbox(main_window):
    store = main_window.alignments
    face = _face(10, 10, 30, 30)
    face["kps_5"] = np.array(
        [[15, 15], [25, 15], [20, 20], [15, 25], [25, 25]], np.float32
    )
    store.put(0, [face], 100, 100, "sig")

    assert aea.resize_face(main_window, 0, 0, [0.10, 0.10, 0.50, 0.70])

    resized = store.get(0, 100, 100, "other")
    assert resized is not None and resized[0]["source"] == SOURCE_MANUAL
    np.testing.assert_allclose(resized[0]["bbox"], [10, 10, 50, 70], atol=1e-3)
    np.testing.assert_allclose(resized[0]["kps_5"][0], [20, 25], atol=1e-3)
    np.testing.assert_allclose(resized[0]["kps_5"][4], [40, 55], atol=1e-3)
    assert resized[0]["embedding"] is None
    assert not aea.resize_face(main_window, 0, 2, [0, 0, 1, 1])
