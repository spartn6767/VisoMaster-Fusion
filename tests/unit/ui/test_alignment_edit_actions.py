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


def test_has_unsaved_changes_tracks_saves(main_window, tmp_path):
    store = main_window.alignments
    assert not store.has_unsaved_changes()
    store.put(1, [_face(10, 10, 30, 30)], 100, 100, "sig")
    assert store.has_unsaved_changes()
    store.save(str(tmp_path / "w.alignments.npz"))
    assert not store.has_unsaved_changes()
    aea.delete_face(main_window, 1, 0)
    store.put(2, [_face(10, 10, 30, 30)], 100, 100, "sig")
    assert store.has_unsaved_changes()


@pytest.mark.parametrize(
    "choice, expected, keeps_unsaved",
    [("cancel", False, True), ("discard", True, True), ("save", True, False)],
)
def test_switch_target_media_prompt(
    main_window, monkeypatch, tmp_path, choice, expected, keeps_unsaved
):
    store = main_window.alignments
    store.put(1, [_face(10, 10, 30, 30)], 100, 100, "sig")
    asked = []
    monkeypatch.setattr(
        aea,
        "_ask_discard_alignments",
        lambda mw, n, name, verb, text: asked.append((n, name, verb)) or choice,
    )
    monkeypatch.setattr(
        aea, "_save_workspace", lambda mw: store.save(str(tmp_path / "w.npz"))
    )

    assert aea.confirm_switch_target_media(main_window, "/v.mp4")
    assert aea.confirm_switch_target_media(main_window, "/other.mp4") is expected
    assert asked == [(1, "v.mp4", "Switch")]
    assert store.has_unsaved_changes() is keeps_unsaved

    with aea.suppress_media_switch_prompt(main_window):
        assert aea.confirm_switch_target_media(main_window, "/other.mp4")
    assert len(asked) == 1


@pytest.mark.parametrize(
    "choice, expected, keeps_unsaved",
    [("cancel", False, True), ("discard", True, True), ("save", True, False)],
)
def test_load_workspace_prompt(
    main_window, monkeypatch, tmp_path, choice, expected, keeps_unsaved
):
    store = main_window.alignments
    asked = []
    monkeypatch.setattr(
        aea,
        "_ask_discard_alignments",
        lambda mw, n, name, verb, text: asked.append((n, name, verb)) or choice,
    )
    monkeypatch.setattr(
        aea, "_save_workspace", lambda mw: store.save(str(tmp_path / "w.npz"))
    )

    assert aea.confirm_load_workspace(main_window)
    assert asked == []

    store.put(1, [_face(10, 10, 30, 30)], 100, 100, "sig")
    assert aea.confirm_load_workspace(main_window) is expected
    assert asked == [(1, "v.mp4", "Load")]
    assert store.has_unsaved_changes() is keeps_unsaved

    with aea.suppress_media_switch_prompt(main_window):
        assert aea.confirm_load_workspace(main_window)
    assert len(asked) == 1


def test_load_saved_workspace_stops_when_prompt_cancelled(monkeypatch, tmp_path):
    from app.ui.widgets.actions import save_load_actions

    workspace = tmp_path / "w.json"
    workspace.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        save_load_actions.video_control_actions,
        "block_if_issue_scan_active",
        lambda *a: False,
    )
    monkeypatch.setattr(aea, "confirm_load_workspace", lambda mw: False)

    def fail(*args):
        raise AssertionError("workspace state was cleared")

    monkeypatch.setattr(
        save_load_actions.list_view_actions, "clear_stop_loading_input_media", fail
    )

    save_load_actions.load_saved_workspace(SimpleNamespace(), str(workspace))


def test_reset_all_alignments_needs_confirmation(main_window, monkeypatch):
    store = main_window.alignments
    store.put(1, [_face(10, 10, 30, 30)], 100, 100, "sig")
    store.add_face(2, _face(10, 10, 30, 30), 100, 100)
    monkeypatch.setattr(aea, "refresh_alignment_overlay", lambda mw: None)
    monkeypatch.setattr(aea, "_confirm_reset_all_alignments", lambda mw, n: False)
    assert aea.reset_all_alignments(main_window) == 0
    assert len(store) == 2
    monkeypatch.setattr(aea, "_confirm_reset_all_alignments", lambda mw, n: True)
    assert aea.reset_all_alignments(main_window) == 2
    assert len(store) == 0


def test_assign_target_pins_face_and_persists(main_window, tmp_path):
    store = main_window.alignments
    store.put(1, [_face(10, 10, 30, 30)], 100, 100, "sig")
    assert aea.assign_target(main_window, 1, 0, "target-1")
    face = store.get(1, 1, 1, "other")[0]
    assert face["assigned_face_id"] == "target-1"
    assert face["source"] == SOURCE_MANUAL

    path = str(tmp_path / "w.alignments.npz")
    store.save(path)
    loaded = AlignmentStore()
    loaded.load(path)
    assert loaded.get(1, 1, 1, None)[0]["assigned_face_id"] == "target-1"

    assert aea.assign_target(main_window, 1, 0, None)
    assert store.get(1, 1, 1, None)[0]["assigned_face_id"] is None
    assert not aea.assign_target(main_window, 1, 5, "target-1")


def test_add_as_target_face_creates_and_assigns(main_window, monkeypatch):
    from app.ui.widgets.actions import card_actions

    store = main_window.alignments
    store.add_face(2, _face(10, 10, 30, 30), 100, 100)
    calls = []
    monkeypatch.setattr(
        card_actions,
        "add_target_face_from_frame",
        lambda mw, frame, kps: calls.append((frame, len(kps))) or "new-target",
    )
    assert aea.add_as_target_face(main_window, 2, 0) == "new-target"
    assert calls == [(2, 5)]
    assert store.get(2, 1, 1, None)[0]["assigned_face_id"] == "new-target"


def test_pasted_face_keeps_assignment(main_window):
    store = main_window.alignments
    store.put(1, [_face(10, 10, 30, 30)], 100, 100, "sig")
    aea.assign_target(main_window, 1, 0, "target-1")
    store.put(3, [_face(12, 12, 32, 32)], 100, 100, "sig")
    assert aea.paste_from_frame(main_window, 1, 0, 3, 0)
    assert store.get(3, 1, 1, None)[0]["assigned_face_id"] == "target-1"


def _kps_face(x: float, face_id: str = "A") -> dict:
    return {
        "bbox": np.array([x, 0.1, x + 0.2, 0.3], np.float32),
        "kps_5": np.full((5, 2), x + 0.1, np.float32),
        "face_id": face_id,
    }


def test_fill_between_edits_and_jump_keys(main_window):
    store = main_window.alignments
    frame = {"n": 0}
    main_window.videoSeekSlider = SimpleNamespace(
        value=lambda: frame["n"], setValue=lambda n: frame.__setitem__("n", n)
    )
    main_window.video_processor.file_type = "video"
    store.add_face(0, _kps_face(0.0), 1, 1)
    store.add_face(4, _kps_face(0.4), 1, 1)
    store.put(2, [_kps_face(0.9)], 1, 1, "sig")

    assert aea.jump_to_edited(main_window, 1) and frame["n"] == 4
    assert aea.jump_to_edited(main_window, -1) and frame["n"] == 0
    assert not aea.jump_to_edited(main_window, -1)

    assert aea.fill_between_edits(main_window, 4, 0) == 3
    for n, x in ((1, 0.1), (2, 0.2), (3, 0.3)):
        faces = store.get(n, 1, 1, None)
        assert len(faces) == 1 and faces[0]["source"] == SOURCE_MANUAL
        np.testing.assert_allclose(faces[0]["bbox"][0], x, atol=1e-6)
    assert store.edited_frames() == [0, 1, 2, 3, 4]

    store.undo()
    assert 1 not in store and 3 not in store
    np.testing.assert_allclose(store.get(2, 1, 1, None)[0]["bbox"][0], 0.9, atol=1e-6)
    assert aea.fill_between_edits(main_window, 0, 0) == 0


def test_arrow_keys_step_frames_only_in_edit_modes(main_window, monkeypatch):
    from app.ui.widgets.actions import video_control_actions as vca

    calls = []
    monkeypatch.setattr(
        vca, "advance_video_slider_by_n_frames", lambda mw: calls.append(1)
    )
    monkeypatch.setattr(
        vca, "rewind_video_slider_by_n_frames", lambda mw: calls.append(-1)
    )
    main_window.video_processor = SimpleNamespace(
        file_type="video", media_path="/v.mp4", processing=False
    )
    main_window.control = {"AlignmentsCacheEnableToggle": True}
    main_window.alignment_edit_enabled = False
    main_window.mask_paint_enabled = False
    assert not aea.step_frame_shortcut(main_window, True) and calls == []
    main_window.mask_paint_enabled = True
    assert aea.step_frame_shortcut(main_window, True)
    main_window.mask_paint_enabled, main_window.alignment_edit_enabled = False, True
    assert aea.step_frame_shortcut(main_window, False)
    assert calls == [1, -1]


def test_edit_mode_keys_run_bar_actions(main_window, monkeypatch):
    from PySide6 import QtCore

    from app.ui.widgets.actions import edit_toolbar_actions as eta
    from app.ui.widgets.actions import mask_paint_actions as mpa

    calls = []
    monkeypatch.setattr(
        eta, "alignment_bar_action", lambda mw, k: calls.append(("align", k))
    )
    monkeypatch.setattr(mpa, "toolbar_action", lambda mw, k: calls.append(("mask", k)))
    main_window.video_processor = SimpleNamespace(
        file_type="video", media_path="/v.mp4", processing=False
    )
    main_window.control = {"AlignmentsCacheEnableToggle": True}
    main_window.alignment_edit_enabled = False
    main_window.mask_paint_enabled = False
    Qt = QtCore.Qt

    def key(k, mods=Qt.KeyboardModifier.NoModifier):
        return SimpleNamespace(key=lambda: k, modifiers=lambda: mods)

    assert not aea.edit_mode_key(main_window, key(Qt.Key.Key_C))
    main_window.alignment_edit_enabled = True
    for k in (Qt.Key.Key_C, Qt.Key.Key_V, Qt.Key.Key_Z, Qt.Key.Key_X, Qt.Key.Key_M):
        assert aea.edit_mode_key(main_window, key(k))
    assert not aea.edit_mode_key(
        main_window, key(Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    )
    assert not aea.edit_mode_key(main_window, key(Qt.Key.Key_Q))
    main_window.alignment_edit_enabled, main_window.mask_paint_enabled = False, True
    assert aea.edit_mode_key(main_window, key(Qt.Key.Key_M))
    assert calls == [
        ("align", "copy"),
        ("align", "paste"),
        ("align", "previous"),
        ("align", "next"),
        ("align", "delete"),
        ("mask", "reset_frame"),
    ]


def test_focus_viewer_targets_the_preview(monkeypatch):
    from PySide6 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    focused = []
    view = QtWidgets.QGraphicsView()
    monkeypatch.setattr(view, "setFocus", lambda *a: focused.append(a))
    aea.focus_viewer(SimpleNamespace(graphicsViewFrame=view))
    aea.focus_viewer(SimpleNamespace())
    assert len(focused) == 1
