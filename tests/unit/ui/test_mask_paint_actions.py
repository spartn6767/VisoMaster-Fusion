from types import SimpleNamespace

import numpy as np
import pytest

from app.processors.video_utils import manual_masks as mm
from app.processors.video_utils.alignment_store import SOURCE_MANUAL, AlignmentStore
from app.ui.widgets.actions import alignment_edit_actions as aea
from app.ui.widgets.actions import mask_paint_actions as mpa

W, H = 1000, 1000
KPS = mm.CANONICAL_KPS * 2 + [100, 50]
NOSE = tuple(KPS[2])


@pytest.fixture
def main_window(monkeypatch):
    monkeypatch.setattr(aea, "_refresh", lambda mw: None)
    store = AlignmentStore()
    store.bind_media("/v.mp4")
    for n in range(3):
        store.put(
            n,
            [{"bbox": [0.1, 0.05, 0.6, 0.6], "kps_5": KPS / W, "face_id": "A"}],
            1,
            1,
            "sig",
        )
    return SimpleNamespace(
        alignments=store, video_processor=SimpleNamespace(media_path="/v.mp4")
    )


def test_stroke_pins_frame_and_holds_on_later_frames(main_window):
    store = main_window.alignments
    assert mpa.apply_stroke(main_window, 1, 0, [NOSE], W, H, erase=False)
    assert store.get(1, 1, 1, "other")[0]["source"] == SOURCE_MANUAL
    assert store.effective_mask(0, 0) is None
    assert store.effective_mask(2, 0) is not None

    rgba = mpa.overlay_rgba(main_window, 2, 200, 200)
    assert rgba is not None and rgba[..., 3].max() > 0
    assert mpa.overlay_rgba(main_window, 0, 200, 200) is None


def test_reset_mask_returns_to_settings_from_that_frame(main_window):
    mpa.apply_stroke(main_window, 0, 0, [NOSE], W, H, erase=False)
    assert mpa.reset_mask(main_window, 1, 0)
    store = main_window.alignments
    assert store.effective_mask(0, 0) is not None
    assert store.effective_mask(1, 0) is None
    assert store.effective_mask(2, 0) is None


def test_reset_all_masks_needs_confirmation(main_window, monkeypatch):
    mpa.apply_stroke(main_window, 0, 0, [NOSE], W, H, erase=False)
    monkeypatch.setattr(mpa, "_confirm_reset_all", lambda mw: False)
    assert mpa.reset_all_masks(main_window) == 0
    monkeypatch.setattr(mpa, "_confirm_reset_all", lambda mw: True)
    assert mpa.reset_all_masks(main_window) == 1
    assert main_window.alignments.mask_frame_count() == 0


def test_copy_paste_and_paste_from_frame(main_window):
    store = main_window.alignments
    assert not mpa.copy_mask(main_window, 0, 0)
    mpa.apply_stroke(main_window, 0, 0, [NOSE], W, H, erase=True)
    assert mpa.copy_mask(main_window, 0, 0)
    mpa.reset_mask(main_window, 1, 0)
    assert mpa.paste_mask(main_window, 2, 0)
    assert np.array_equal(store.effective_mask(2, 0), store.effective_mask(0, 0))

    assert mpa.paste_mask_from_frame(main_window, 1, 0, 2, 0)
    assert store.effective_mask(2, 0) is None


def test_reset_mask_frame_only_affects_that_frame(main_window):
    store = main_window.alignments
    mpa.apply_stroke(main_window, 0, 0, [NOSE], W, H, erase=False)
    assert mpa.reset_mask_frame(main_window, 1, 0)
    assert [store.effective_mask(n, 0) is not None for n in range(3)] == [
        True,
        False,
        True,
    ]

    mpa.apply_stroke(main_window, 1, 0, [NOSE], W, H, erase=True)
    painted = store.effective_mask(1, 0)
    assert mpa.reset_mask_frame(main_window, 1, 0)
    assert store.effective_mask(1, 0) is None
    assert np.array_equal(store.effective_mask(2, 0), painted)


def test_overlay_shows_computed_mask_with_painted_edits(main_window):
    computed = np.zeros((256, 256), dtype=np.uint8)
    computed[60:200, 60:200] = 255
    main_window.computed_masks = {
        "media_path": "/v.mp4",
        "frame": 0,
        "faces": {0: computed},
    }
    rgba = mpa.overlay_rgba(main_window, 0, W, H)
    nose = (int(NOSE[1]), int(NOSE[0]))
    assert rgba is not None and rgba[nose][3] > 0 and rgba[nose][1] == 220
    assert mpa.overlay_rgba(main_window, 1, W, H) is None

    mpa.apply_stroke(main_window, 0, 0, [NOSE], W, H, erase=True)
    rgba = mpa.overlay_rgba(main_window, 0, W, H)
    assert tuple(rgba[nose][:3]) == (255, 60, 60)


class _Button:
    def __init__(self, checked, on_uncheck=None):
        self.checked = checked
        self.on_uncheck = on_uncheck

    def isChecked(self):
        return self.checked

    def setChecked(self, value):
        self.checked = value
        if not value and self.on_uncheck:
            self.on_uncheck()


def test_edit_modes_pause_playback_without_prompt(main_window, monkeypatch):
    vp = SimpleNamespace(file_type="video", media_path="/v.mp4", processing=True)
    main_window.video_processor = vp
    main_window.control = {"AlignmentsCacheEnableToggle": True}
    main_window.buttonMediaRecord = _Button(False)
    main_window.buttonMediaPlay = _Button(
        True, lambda: setattr(vp, "processing", False)
    )
    monkeypatch.setattr(
        aea.QtWidgets.QMessageBox,
        "information",
        lambda *a: pytest.fail("prompted"),
    )
    monkeypatch.setattr(aea, "refresh_alignment_overlay", lambda mw: None)
    monkeypatch.setattr(mpa, "sync_mask_paint_ui", lambda mw: None)

    mpa.set_mask_paint_mode(main_window, True)
    assert not main_window.buttonMediaPlay.checked
    assert mpa.is_mask_paint_active(main_window)

    vp.processing = True
    main_window.buttonMediaPlay.checked = True
    aea.set_alignment_edit_mode(main_window, True)
    assert aea.is_alignment_edit_active(main_window)
    assert not main_window.mask_paint_enabled


def test_recording_is_not_paused(main_window):
    vp = SimpleNamespace(file_type="video", media_path="/v.mp4", processing=True)
    main_window.video_processor = vp
    main_window.buttonMediaRecord = _Button(True)
    main_window.buttonMediaPlay = _Button(True)
    aea.pause_for_editing(main_window)
    assert main_window.buttonMediaPlay.checked and vp.processing


def test_brush_bar_drags_within_viewer():
    from PySide6 import QtCore, QtGui, QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    view = QtWidgets.QWidget()
    view.resize(1200, 600)
    main_window = SimpleNamespace(graphicsViewFrame=view)
    bar = mpa._build_brush_bar(main_window)
    handle = bar.findChild(mpa._DragHandle)
    rows = bar.layout()
    assert isinstance(rows, QtWidgets.QVBoxLayout) and rows.count() == 2
    assert (
        rows.itemAt(1).layout().count() == len(mpa.BAR_ACTIONS) + 1
    )  # + "From frame" label
    assert handle is not None and bar.pos() == QtCore.QPoint(
        *mpa.default_bar_position(bar)
    )
    bar.move(QtCore.QPoint(8, 8))

    def send(kind, x, y):
        button = QtCore.Qt.MouseButton.LeftButton
        point = QtCore.QPointF(x, y)
        handle.event(
            QtGui.QMouseEvent(
                kind,
                point,
                point,
                button,
                button,
                QtCore.Qt.KeyboardModifier.NoModifier,
            )
        )

    send(QtCore.QEvent.Type.MouseButtonPress, 2, 2)
    send(QtCore.QEvent.Type.MouseMove, 52, 32)
    send(QtCore.QEvent.Type.MouseButtonRelease, 52, 32)
    assert bar.pos() == QtCore.QPoint(58, 38)
    assert main_window.mask_brush["bar_pos"] == (58, 38)

    far = mpa.clamp_bar_position(bar, QtCore.QPoint(5000, -50))
    assert far == QtCore.QPoint(max(0, 1200 - bar.width()), 0)


def test_brush_bar_actions_use_selected_face(main_window, monkeypatch):
    frame = {"n": 1}
    main_window.videoSeekSlider = SimpleNamespace(value=lambda: frame["n"])
    store = main_window.alignments
    mpa.apply_stroke(main_window, 0, 0, [NOSE], W, H, erase=False)
    assert mpa.active_face(main_window) == 0
    assert mpa.toolbar_action(main_window, "reset_frame")
    assert store.effective_mask(1, 0) is None and store.effective_mask(2, 0) is not None
    assert mpa.toolbar_action(main_window, "previous")
    assert store.effective_mask(1, 0) is not None
    frame["n"] = 0
    assert mpa.toolbar_action(main_window, "copy")
    frame["n"] = 2
    assert mpa.toolbar_action(main_window, "reset_on")
    assert store.effective_mask(2, 0) is None
    assert mpa.toolbar_action(main_window, "paste")
    assert store.effective_mask(2, 0) is not None


def test_brush_bar_close_button_leaves_paint_mode(monkeypatch):
    from PySide6 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    view = QtWidgets.QWidget()
    view.resize(1200, 600)
    main_window = SimpleNamespace(graphicsViewFrame=view)
    calls = []
    monkeypatch.setattr(
        mpa, "set_mask_paint_mode", lambda mw, on: calls.append((mw, on))
    )
    bar = mpa._build_brush_bar(main_window)
    close = bar.findChild(QtWidgets.QToolButton, "closeBarButton")
    assert close is not None and close.isEnabled()
    close.click()
    assert calls == [(main_window, False)]


def test_undo_redo_strokes_and_resets(main_window, monkeypatch):
    monkeypatch.setattr(aea, "refresh_alignment_overlay", lambda mw: None)
    monkeypatch.setattr(mpa, "refresh_mask_overlay", lambda mw: None)
    frame = {"n": 2}
    main_window.videoSeekSlider = SimpleNamespace(
        value=lambda: frame["n"], setValue=lambda n: frame.__setitem__("n", n)
    )
    store = main_window.alignments
    mpa.apply_stroke(main_window, 0, 0, [NOSE], W, H, erase=False)
    assert store.effective_mask(2, 0) is not None
    assert mpa.toolbar_action(main_window, "undo")
    assert frame["n"] == 0
    assert store.effective_mask(0, 0) is None and store.effective_mask(2, 0) is None
    assert store.get(0, 1, 1, "other") is None
    assert mpa.toolbar_action(main_window, "redo")
    assert store.effective_mask(2, 0) is not None

    monkeypatch.setattr(mpa, "_confirm_reset_all", lambda mw: True)
    assert mpa.reset_all_masks(main_window) == 1
    assert aea.undo_edit(main_window)
    assert store.effective_mask(2, 0) is not None
    assert aea.undo_edit(main_window)
    assert not aea.undo_edit(main_window)


def test_this_frame_only_keeps_the_next_frame(main_window):
    store = main_window.alignments
    mpa.apply_stroke(main_window, 0, 0, [NOSE], W, H, erase=False)
    first = store.effective_mask(0, 0).copy()
    mpa.brush_settings(main_window)["frame_only"] = True
    mpa.apply_stroke(main_window, 1, 0, [NOSE], W, H, erase=True)
    assert not np.array_equal(store.effective_mask(1, 0), first)
    assert np.array_equal(store.effective_mask(2, 0), first)

    store.undo()
    assert store.get(2, 1, 1, None)[0]["mask"] is None
    assert np.array_equal(store.effective_mask(1, 0), first)


def test_apply_mask_to_range(main_window):
    store = main_window.alignments
    for n in (3, 4):
        store.put(
            n,
            [{"bbox": [0.1, 0.05, 0.6, 0.6], "kps_5": KPS / W, "face_id": "A"}],
            1,
            1,
            "sig",
        )
    mpa.apply_stroke(main_window, 2, 0, [NOSE], W, H, erase=False)
    painted = store.effective_mask(2, 0).copy()

    assert mpa.apply_mask_to_range(main_window, 0, 0, 1, 2) == 2
    assert store.effective_mask(1, 0) is None and store.effective_mask(2, 0) is None
    assert np.array_equal(store.effective_mask(3, 0), painted)
    assert np.array_equal(store.effective_mask(4, 0), painted)

    store.undo()
    assert np.array_equal(store.effective_mask(2, 0), painted)
    assert store.get(3, 1, 1, None)[0]["mask"] is None


def test_range_action_reports_result(main_window, monkeypatch):
    from app.ui.widgets.actions import common_actions

    toasts = []
    monkeypatch.setattr(
        common_actions,
        "create_and_show_toast_message",
        lambda mw, title, text: toasts.append(text),
    )
    monkeypatch.setattr(mpa, "_ask_range", lambda mw, frame: (0, 2))
    monkeypatch.setattr(mpa, "refresh_mask_overlay", lambda mw: None)
    main_window.videoSeekSlider = SimpleNamespace(value=lambda: 0)
    assert mpa.toolbar_action(main_window, "range") == 3
    assert toasts == ["Mask applied to frames 0\u20132 (3 frames with this face)."]


def test_brush_circle_is_green_to_paint_and_red_to_erase(main_window, monkeypatch):
    from PySide6 import QtCore, QtGui, QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    scene = QtWidgets.QGraphicsScene()
    view = QtWidgets.QGraphicsView(scene)
    image = QtGui.QImage(256, 256, QtGui.QImage.Format.Format_RGB32)
    scene.addPixmap(QtGui.QPixmap.fromImage(image))
    main_window.graphicsViewFrame = view
    monkeypatch.setattr(mpa, "is_mask_paint_active", lambda mw: True)

    def circle_rgb():
        mpa.update_brush_cursor(main_window, view, QtCore.QPointF(128, 128))
        return getattr(main_window, mpa._CURSOR_ATTR)[0].pen().color().getRgb()[:3]

    assert circle_rgb() == mpa._ADD_RGB
    mpa.brush_settings(main_window)["erase"] = True
    assert circle_rgb() == mpa._ERASE_RGB


def test_floating_bar_defaults_to_bottom_right():
    from PySide6 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    view = QtWidgets.QWidget()
    view.resize(800, 600)
    bar = QtWidgets.QFrame(view)
    bar.resize(200, 50)
    assert mpa.default_bar_position(bar) == (592, 542)
    assert mpa.default_bar_position(QtWidgets.QFrame()) == (8, 8)


def test_repeated_paste_walks_frames_skipping_same_masks(main_window):
    store = main_window.alignments
    for n in (3, 4):
        store.put(
            n,
            [{"bbox": [0.1, 0.05, 0.6, 0.6], "kps_5": KPS / W, "face_id": "A"}],
            1,
            1,
            "sig",
        )
    for n, point in enumerate((NOSE, tuple(KPS[0]), tuple(KPS[3]))):
        mpa.apply_stroke(main_window, n, 0, [point], W, H, erase=False)
    masks = [store.effective_mask(n, 0).copy() for n in range(3)]
    store.clear_history()
    main_window.videoSeekSlider = SimpleNamespace(value=lambda: 4)

    def shown():
        return store.effective_mask(4, 0)

    # 3 and 2 show the same mask as frame 4 (held from 2), so they're skipped.
    assert mpa.toolbar_action(main_window, "previous")
    assert np.array_equal(shown(), masks[1])
    assert mpa.paste_source_text(main_window, "mask", 0) == "From frame 1"
    assert mpa.toolbar_action(main_window, "previous")
    assert np.array_equal(shown(), masks[0])
    assert not mpa.toolbar_action(main_window, "previous")
    assert mpa.toolbar_action(main_window, "next")
    assert np.array_equal(shown(), masks[1])
    assert mpa.toolbar_action(main_window, "next")
    assert np.array_equal(shown(), masks[2])
    assert store.get(4, 1, 1, None)[0]["source"] == SOURCE_MANUAL
    # Frame 3 matches the shown mask; next is frame 4 itself: the original.
    assert mpa.toolbar_action(main_window, "next")
    assert store.get(4, 1, 1, None)[0]["source"] != SOURCE_MANUAL
    assert mpa.paste_source_text(main_window, "mask", 0) == ""
    assert not store.can_undo()

    assert mpa.toolbar_action(main_window, "previous")
    assert mpa.toolbar_action(main_window, "previous")
    assert store.undo() is not None and not store.can_undo()
    assert store.get(4, 1, 1, None)[0]["source"] != SOURCE_MANUAL


def test_paste_run_ends_after_another_edit(main_window):
    mpa.apply_stroke(main_window, 0, 0, [NOSE], W, H, erase=False)
    main_window.videoSeekSlider = SimpleNamespace(value=lambda: 2)
    assert mpa.toolbar_action(main_window, "reset_frame")
    assert mpa.toolbar_action(main_window, "previous")
    assert mpa.paste_source_text(main_window, "mask", 0) == "From frame 1"
    mpa.apply_stroke(main_window, 2, 0, [tuple(KPS[0])], W, H, erase=False)
    assert mpa.paste_source_text(main_window, "mask", 0) == ""
