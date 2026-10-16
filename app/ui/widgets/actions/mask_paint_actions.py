"""Hand-painted face masks in the viewer ("Paint Mask" mode).

Strokes are stored on the face's alignment entry in canonical face space (see
``manual_masks``) and override the computed swap mask. A painted mask holds on the
same target face in later frames until the next frame painted for that face;
"Reset Mask to Settings" writes an empty keyframe so the computed mask applies again.
"""

from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from app.processors.video_utils import manual_masks
from app.processors.video_utils.alignment_store import SOURCE_MANUAL
from app.ui.widgets.actions import alignment_edit_actions as aea

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

_OVERLAY_ATTR = "_mask_paint_items"
_STROKE_ATTR = "_mask_paint_stroke"
_BRUSH_BAR_ATTR = "_mask_brush_bar"
_CLIPBOARD_ATTR = "mask_clipboard"
_ACTIVE_FACE_ATTR = "_mask_active_face"
_BAR_BUTTONS_ATTR = "_mask_bar_buttons"
_CURSOR_ATTR = "_mask_brush_cursor"
_ADD_RGB = (0, 220, 120)
_ERASE_RGB = (255, 60, 60)
DEFAULT_BRUSH = {
    "size": 40,
    "feather": 6,
    "opacity": 45,
    "erase": False,
    "frame_only": False,
}


def brush_settings(main_window: "MainWindow") -> dict[str, Any]:
    brush = getattr(main_window, "mask_brush", None)
    if not isinstance(brush, dict):
        brush = dict(DEFAULT_BRUSH)
        main_window.mask_brush = brush
    return brush


def is_mask_paint_active(main_window: "MainWindow") -> bool:
    return bool(getattr(main_window, "mask_paint_enabled", False)) and (
        aea.can_edit_alignments(main_window)
    )


def set_mask_paint_mode(main_window: "MainWindow", enabled: bool) -> None:
    main_window.mask_paint_enabled = bool(enabled)
    hide_brush_cursor(main_window)
    view = getattr(main_window, "graphicsViewFrame", None)
    if enabled and isinstance(view, QtWidgets.QGraphicsView):
        view.viewport().setMouseTracking(True)
    if enabled:
        aea.focus_viewer(main_window)
    if enabled:
        main_window.alignment_edit_enabled = False
        aea.pause_for_editing(main_window)
        # Re-render so the worker captures the computed masks for the overlay.
        aea._refresh(main_window)
        if not aea.can_edit_alignments(main_window):
            QtWidgets.QMessageBox.information(
                main_window,
                "Paint Mask",
                "Masks can be painted on a paused video frame with "
                "Settings > Detectors > Reuse Saved Alignments on, and Face Compare, "
                "Face Mask view, VR180 and Manual Rotation off.",
            )
    aea.refresh_alignment_overlay(main_window)
    sync_mask_paint_ui(main_window)


def sync_mask_paint_ui(main_window: "MainWindow") -> None:
    """Shows/hides the brush bar and redraws the mask overlay."""
    active = is_mask_paint_active(main_window)
    bar = getattr(main_window, _BRUSH_BAR_ATTR, None)
    if active and bar is None:
        bar = _build_brush_bar(main_window)
    if bar is not None:
        bar.setVisible(active)
        if active:
            bar.move(clamp_bar_position(bar, bar.pos()))
            bar.raise_()
    refresh_mask_overlay(main_window)
    from app.ui.widgets.actions import edit_toolbar_actions

    edit_toolbar_actions.sync_edit_toolbar(main_window)


BAR_ACTIONS = (
    ("undo", "Undo", "Undo the last alignment or mask edit (Ctrl+Z)."),
    ("redo", "Redo", "Redo the last undone edit (Ctrl+Shift+Z)."),
    ("copy", "Copy", "Copy the selected face's mask (C)."),
    ("paste", "Paste", "Paste the copied mask onto the selected face (V)."),
    (
        "previous",
        "< Paste",
        "Paste the mask from the nearest earlier stored frame (Z).",
    ),
    ("next", "Paste >", "Paste the mask from the nearest later stored frame (X)."),
    ("reset_frame", "Reset Frame", "Computed mask on this frame only (M)."),
    ("reset_on", "Reset From Here", "Computed mask from this frame on."),
    ("reset_all", "Reset All", "Remove every painted mask in this video."),
    (
        "range",
        "Apply to Range...",
        "Use the selected face's mask on every frame from one frame to another.",
    ),
)


def set_active_face(main_window: "MainWindow", frame_number: int, face_index: int):
    """The face the brush bar's actions apply to (last painted or right-clicked)."""
    setattr(main_window, _ACTIVE_FACE_ATTR, (int(frame_number), int(face_index)))


def active_face(main_window: "MainWindow") -> Optional[int]:
    """Selected face on the shown frame; the only face if there is just one."""
    frame_number = aea.current_frame_number(main_window)
    faces = aea._frame_faces(main_window, frame_number)
    chosen = getattr(main_window, _ACTIVE_FACE_ATTR, None)
    if chosen is not None and chosen[0] == frame_number and chosen[1] < len(faces):
        return chosen[1]
    return 0 if len(faces) == 1 else None


def _adjacent_source(
    main_window: "MainWindow", frame_number: int, face_index: int, step: int
) -> Optional[tuple[int, int]]:
    other = main_window.alignments.nearest_frame(frame_number, step)
    if other is None:
        return None
    x1, y1, x2, y2 = aea._frame_faces(main_window, frame_number)[face_index]["bbox"]
    other_index = aea.nearest_face_index(
        aea._frame_faces(main_window, other), (x1 + x2) / 2.0, (y1 + y2) / 2.0
    )
    return None if other_index is None else (other, other_index)


def toolbar_action(main_window: "MainWindow", key: str, *args: Any) -> Any:
    """Runs a brush-bar action on the selected face of the shown frame."""
    if key == "reset_all":
        return reset_all_masks(main_window)
    if key in ("undo", "redo"):
        return aea.undo_edit(main_window, redo=key == "redo")
    frame_number = aea.current_frame_number(main_window)
    face_index = active_face(main_window)
    result: Any = False
    if face_index is None:
        return result
    if key == "copy":
        result = copy_mask(main_window, frame_number, face_index)
    elif key == "paste":
        result = paste_mask(main_window, frame_number, face_index)
    elif key == "reset_frame":
        result = reset_mask_frame(main_window, frame_number, face_index)
    elif key == "reset_on":
        result = reset_mask(main_window, frame_number, face_index)
    elif key == "range":
        frames = _ask_range(main_window, frame_number)
        if frames is not None:
            result = apply_mask_to_range(main_window, frame_number, face_index, *frames)
            _show_range_result(main_window, *frames, result)
    elif key in ("previous", "next"):
        source = _adjacent_source(
            main_window, frame_number, face_index, -1 if key == "previous" else 1
        )
        if source is not None:
            result = paste_mask_from_frame(
                main_window, *source, frame_number, face_index
            )
    _sync_bar_buttons(main_window)
    return result


def _sync_bar_buttons(main_window: "MainWindow") -> None:
    buttons = getattr(main_window, _BAR_BUTTONS_ATTR, None)
    if not buttons:
        return
    store = main_window.alignments
    frame_number = aea.current_frame_number(main_window)
    face_index = active_face(main_window)
    enabled = {
        "reset_all": store.mask_frame_count() > 0,
        "undo": store.can_undo(),
        "redo": store.can_redo(),
    }
    if face_index is not None:
        has_mask = store.effective_mask(frame_number, face_index) is not None
        enabled.update(
            copy=has_mask,
            paste=getattr(main_window, _CLIPBOARD_ATTR, None) is not None,
            previous=_adjacent_source(main_window, frame_number, face_index, -1)
            is not None,
            next=_adjacent_source(main_window, frame_number, face_index, 1) is not None,
            reset_frame=has_mask,
            reset_on=has_mask,
            range=_face_id(main_window, frame_number, face_index) is not None,
        )
    for key, button in buttons.items():
        button.setEnabled(bool(enabled.get(key, False)))


def clamp_bar_position(bar: QtWidgets.QWidget, point: QtCore.QPoint) -> QtCore.QPoint:
    """Keeps the floating brush bar fully inside its parent (the viewer)."""
    parent = bar.parentWidget()
    if parent is None:
        return point
    max_x = max(0, parent.width() - bar.width())
    max_y = max(0, parent.height() - bar.height())
    return QtCore.QPoint(min(max(0, point.x()), max_x), min(max(0, point.y()), max_y))


class _DragHandle(QtWidgets.QLabel):
    """Grip that drags its brush bar around the viewer."""

    def __init__(
        self, bar: QtWidgets.QWidget, on_moved: Callable[[QtCore.QPoint], None]
    ):
        super().__init__("\u2261", bar)
        self._bar = bar
        self._on_moved = on_moved
        self._grab: Optional[QtCore.QPoint] = None
        self.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)
        self.setToolTip("Drag to move the brush bar.")

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._grab = event.position().toPoint()
            event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._grab is None:
            return
        delta = event.position().toPoint() - self._grab
        self._on_moved(clamp_bar_position(self._bar, self._bar.pos() + delta))
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self._grab = None
        event.accept()


def close_bar_button(
    bar: QtWidgets.QWidget, mode: str, close: Callable[[], None]
) -> QtWidgets.QToolButton:
    """✕ button that leaves the bar's edit mode, like unchecking its toolbar button."""
    button = QtWidgets.QToolButton(bar)
    button.setObjectName("closeBarButton")
    button.setText("\u2715")
    button.setToolTip(f"Close {mode}")
    button.setAutoRaise(True)
    button.clicked.connect(lambda *_: close())
    return button


def _build_brush_bar(main_window: "MainWindow") -> QtWidgets.QFrame:
    brush = brush_settings(main_window)
    bar = QtWidgets.QFrame(main_window.graphicsViewFrame)
    bar.setObjectName("maskBrushBar")
    bar.setStyleSheet(
        "#maskBrushBar { background: rgba(20, 20, 20, 200); border-radius: 4px; }"
    )
    rows = QtWidgets.QVBoxLayout(bar)
    rows.setContentsMargins(6, 4, 6, 4)
    rows.setSpacing(4)
    layout = QtWidgets.QHBoxLayout()
    actions_row = QtWidgets.QHBoxLayout()
    rows.addLayout(layout)
    rows.addLayout(actions_row)

    def moved(point: QtCore.QPoint) -> None:
        bar.move(point)
        brush["bar_pos"] = (point.x(), point.y())

    layout.addWidget(_DragHandle(bar, moved))

    erase = QtWidgets.QCheckBox("Erase (Shift)", bar)
    erase.setChecked(bool(brush["erase"]))
    erase.toggled.connect(lambda checked: brush.__setitem__("erase", bool(checked)))
    layout.addWidget(erase)

    frame_only = QtWidgets.QCheckBox("This Frame Only", bar)
    frame_only.setChecked(bool(brush.get("frame_only", False)))
    frame_only.setToolTip(
        "Strokes and pastes change only this frame; the next frame keeps the mask "
        "it showed before."
    )
    frame_only.toggled.connect(
        lambda checked: brush.__setitem__("frame_only", bool(checked))
    )
    layout.addWidget(frame_only)

    def add_spin(label: str, key: str, low: int, high: int, tip: str) -> None:
        layout.addWidget(QtWidgets.QLabel(label, bar))
        spin = QtWidgets.QSpinBox(bar)
        spin.setRange(low, high)
        spin.setValue(int(brush[key]))
        spin.setToolTip(tip)

        def changed(value: int) -> None:
            brush[key] = int(value)
            if key == "opacity":
                refresh_mask_overlay(main_window)

        spin.valueChanged.connect(changed)
        layout.addWidget(spin)

    add_spin("Size", "size", 2, 1000, "Brush diameter in frame pixels.")
    add_spin("Feather", "feather", 0, 200, "Brush edge softness in frame pixels.")
    add_spin(
        "Opacity %", "opacity", 0, 100, "Overlay opacity (preview only, not the mask)."
    )
    layout.addStretch(1)
    layout.addWidget(
        close_bar_button(
            bar, "Paint Mask", lambda: set_mask_paint_mode(main_window, False)
        )
    )
    bar_buttons: dict[str, QtWidgets.QToolButton] = {}
    for key, text, tip in BAR_ACTIONS:
        button = QtWidgets.QToolButton(bar)
        button.setText(text)
        button.setToolTip(tip)
        button.clicked.connect(partial(toolbar_action, main_window, key))
        actions_row.addWidget(button)
        bar_buttons[key] = button
    setattr(main_window, _BAR_BUTTONS_ATTR, bar_buttons)

    bar.adjustSize()
    x, y = brush.get("bar_pos", (8, 8))
    bar.move(clamp_bar_position(bar, QtCore.QPoint(int(x), int(y))))
    setattr(main_window, _BRUSH_BAR_ATTR, bar)
    return bar


# --- Mask edits ------------------------------------------------------------
def _face_kps_px(face: dict, frame_w: float, frame_h: float) -> np.ndarray:
    return np.asarray(face["kps_5"], dtype=np.float64)[:5] * (frame_w, frame_h)


def _face_id(
    main_window: "MainWindow", frame_number: int, face_index: int
) -> Optional[str]:
    faces = aea._frame_faces(main_window, frame_number)
    return faces[face_index].get("face_id") if 0 <= face_index < len(faces) else None


def _keep_frame_mask(
    main_window: "MainWindow", face_id: Optional[str], frame_number: int
) -> None:
    """Stores the mask a face currently shows on ``frame_number`` as its own
    keyframe, so edits on earlier frames stop carrying past it."""
    if face_id is None:
        return
    store = main_window.alignments
    for index, face in enumerate(aea._frame_faces(main_window, frame_number)):
        if face.get("face_id") != face_id:
            continue
        if face.get("mask") is None:
            held = store.effective_mask(frame_number, index)
            store.update_face(
                frame_number,
                index,
                1,
                1,
                mask=manual_masks.empty_mask() if held is None else held.copy(),
                source=SOURCE_MANUAL,
            )
        return


@aea.records_edit
def set_mask(
    main_window: "MainWindow",
    frame_number: int,
    face_index: int,
    mask: Optional[np.ndarray],
    skip: bool = False,
    frame_only: Optional[bool] = None,
) -> bool:
    """Stores a mask keyframe on a face; the frame becomes manual (pinned).
    frame_only (default: the brush bar's This Frame Only) keeps the next frame
    on the mask it showed before."""
    store = main_window.alignments
    if frame_only is None:
        frame_only = bool(brush_settings(main_window).get("frame_only")) and not skip
    with store.edit([frame_number + 1] if frame_only else []):
        if frame_only:
            _keep_frame_mask(
                main_window,
                _face_id(main_window, frame_number, face_index),
                frame_number + 1,
            )
        if not store.update_face(
            frame_number,
            face_index,
            1,
            1,
            mask=mask,
            mask_skip=skip,
            source=SOURCE_MANUAL,
        ):
            return False
    aea._refresh(main_window)
    refresh_mask_overlay(main_window)
    return True


def apply_stroke(
    main_window: "MainWindow",
    frame_number: int,
    face_index: int,
    points: list[tuple[float, float]],
    frame_w: float,
    frame_h: float,
    erase: bool,
) -> bool:
    """Paints a stroke (frame pixel coords) onto a face's effective mask."""
    faces = aea._frame_faces(main_window, frame_number)
    if not 0 <= face_index < len(faces) or not points:
        return False
    brush = brush_settings(main_window)
    mask = manual_masks.paint_stroke(
        main_window.alignments.effective_mask(frame_number, face_index),
        points,
        _face_kps_px(faces[face_index], frame_w, frame_h),
        radius=float(brush["size"]) / 2.0,
        erase=erase,
        feather=float(brush["feather"]),
    )
    return set_mask(main_window, frame_number, face_index, mask)


def reset_mask(main_window: "MainWindow", frame_number: int, face_index: int) -> bool:
    """Back to the computed mask for this face from this frame on."""
    return set_mask(
        main_window,
        frame_number,
        face_index,
        manual_masks.empty_mask(),
        frame_only=False,
    )


def reset_mask_frame(
    main_window: "MainWindow", frame_number: int, face_index: int
) -> bool:
    """Computed mask on this frame only; later frames keep the mask they had."""
    faces = aea._frame_faces(main_window, frame_number)
    if not 0 <= face_index < len(faces):
        return False
    return set_mask(
        main_window, frame_number, face_index, faces[face_index].get("mask"), skip=True
    )


def apply_mask_to_range(
    main_window: "MainWindow",
    frame_number: int,
    face_index: int,
    start: int,
    end: int,
) -> int:
    """Gives the same person the selected face's mask on every stored frame from
    start to end; frames after end keep the mask they showed before. Returns how
    many frames got it."""
    face_id = _face_id(main_window, frame_number, face_index)
    if face_id is None or end < start:
        return 0
    store = main_window.alignments
    mask = store.effective_mask(frame_number, face_index)
    mask = manual_masks.empty_mask() if mask is None else mask.copy()
    frames = [n for n in store.frames() if start <= n <= end]
    keyframe_set = False
    applied = 0
    with store.edit([*frames, end + 1]):
        _keep_frame_mask(main_window, face_id, end + 1)
        for n in frames:
            for index, face in enumerate(aea._frame_faces(main_window, n)):
                if face.get("face_id") != face_id:
                    continue
                store.update_face(
                    n,
                    index,
                    1,
                    1,
                    mask=None if keyframe_set else mask,
                    mask_skip=False,
                    source=SOURCE_MANUAL,
                )
                keyframe_set = True
                applied += 1
                break
    if not keyframe_set:
        return 0
    aea._refresh(main_window)
    refresh_mask_overlay(main_window)
    return applied


def _show_range_result(main_window: "MainWindow", start: int, end: int, applied: int):
    from app.ui.widgets.actions import common_actions

    if applied:
        common_actions.create_and_show_toast_message(
            main_window,
            "Mask Applied",
            f"Mask applied to frames {start}\u2013{end} ({applied} frames with this face).",
        )
    else:
        QtWidgets.QMessageBox.information(
            main_window,
            "Apply Mask to Range",
            f"No stored frames with this face between {start} and {end}.",
        )


def _ask_range(
    main_window: "MainWindow", frame_number: int
) -> Optional[tuple[int, int]]:
    """From/To frame dialog; defaults to this frame up to the next marker."""
    slider = main_window.videoSeekSlider
    last = int(slider.maximum())
    markers = [m for m in getattr(slider, "markers_sorted", []) if m > frame_number]
    dialog = QtWidgets.QDialog(main_window)
    dialog.setWindowTitle("Apply Mask to Range")
    form = QtWidgets.QFormLayout(dialog)
    spins = []
    for label, value in (
        ("From frame", frame_number),
        ("To frame", markers[0] - 1 if markers else last),
    ):
        spin = QtWidgets.QSpinBox(dialog)
        spin.setRange(int(slider.minimum()), last)
        spin.setValue(int(value))
        form.addRow(label, spin)
        spins.append(spin)
    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
        | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
        dialog,
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    form.addRow(buttons)
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None
    start, end = spins[0].value(), spins[1].value()
    return (min(start, end), max(start, end))


def reset_all_masks(main_window: "MainWindow") -> int:
    if not _confirm_reset_all(main_window):
        return 0
    with main_window.alignments.edit():
        cleared = main_window.alignments.clear_masks()
    if cleared:
        aea._refresh(main_window)
        refresh_mask_overlay(main_window)
    return cleared


def _confirm_reset_all(main_window: "MainWindow") -> bool:
    answer = QtWidgets.QMessageBox.question(
        main_window,
        "Reset All Masks",
        "Remove every painted mask in this video? Faces go back to the masks "
        "your mask settings produce.",
    )
    return answer == QtWidgets.QMessageBox.StandardButton.Yes


def copy_mask(main_window: "MainWindow", frame_number: int, face_index: int) -> bool:
    mask = main_window.alignments.effective_mask(frame_number, face_index)
    if mask is None:
        return False
    setattr(main_window, _CLIPBOARD_ATTR, mask.copy())
    return True


def paste_mask(main_window: "MainWindow", frame_number: int, face_index: int) -> bool:
    copied = getattr(main_window, _CLIPBOARD_ATTR, None)
    if copied is None:
        return False
    return set_mask(main_window, frame_number, face_index, copied.copy())


def paste_mask_from_frame(
    main_window: "MainWindow",
    source_frame: int,
    source_index: int,
    frame_number: int,
    face_index: int,
) -> bool:
    """Copies the effective mask of a face on another frame (clipboard untouched)."""
    mask = main_window.alignments.effective_mask(source_frame, source_index)
    return set_mask(
        main_window,
        frame_number,
        face_index,
        manual_masks.empty_mask() if mask is None else mask.copy(),
    )


# --- Overlay ---------------------------------------------------------------
def clear_mask_overlay(main_window: "MainWindow") -> None:
    for item in getattr(main_window, _OVERLAY_ATTR, []):
        try:
            if item.scene() is not None:
                item.scene().removeItem(item)
        except RuntimeError:
            pass
    setattr(main_window, _OVERLAY_ATTR, [])


def computed_mask(
    main_window: "MainWindow", frame_number: int, face_index: int
) -> Optional[np.ndarray]:
    """The worker's computed (settings) mask for a face on the shown frame, if captured."""
    data = getattr(main_window, "computed_masks", None)
    if not isinstance(data, dict) or data.get("frame") != frame_number:
        return None
    if not main_window.alignments.matches_media(data.get("media_path")):
        return None
    return data.get("faces", {}).get(face_index)


def overlay_rgba(
    main_window: "MainWindow", frame_number: int, frame_w: int, frame_h: int
) -> Optional[np.ndarray]:
    """Tinted RGBA preview of the final masks on a frame: the computed mask with
    the painted add/erase applied (green), erased areas in red. None if empty."""
    store = main_window.alignments
    final = np.zeros((frame_h, frame_w), dtype=np.float32)
    erased = np.zeros((frame_h, frame_w), dtype=np.float32)
    found = False
    for face_index, face in enumerate(aea._frame_faces(main_window, frame_number)):
        painted = store.effective_mask(frame_number, face_index)
        computed = computed_mask(main_window, frame_number, face_index)
        if painted is None and computed is None:
            continue
        stack = np.zeros((3, manual_masks.MASK_SIZE, manual_masks.MASK_SIZE), np.uint8)
        if painted is not None:
            stack[:2] = painted
        if computed is not None:
            stack[2] = computed
        add, erase, base = manual_masks.mask_to_frame(
            stack, _face_kps_px(face, frame_w, frame_h), frame_w, frame_h
        )
        np.maximum(final, manual_masks.combine(base, add, erase), out=final)
        np.maximum(erased, erase, out=erased)
        found = True
    if not found:
        return None
    erased *= 1.0 - final
    opacity = float(brush_settings(main_window)["opacity"]) / 100.0
    rgba = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)
    rgba[..., :3] = np.where((erased > final)[..., None], _ERASE_RGB, _ADD_RGB)
    rgba[..., 3] = np.clip(np.maximum(final, erased) * opacity * 255.0, 0, 255)
    return rgba


def refresh_mask_overlay(main_window: "MainWindow") -> None:
    clear_mask_overlay(main_window)
    _sync_bar_buttons(main_window)
    if not is_mask_paint_active(main_window):
        return
    rect = aea._pixmap_rect(main_window)
    if rect is None:
        return
    frame_w, frame_h = int(round(rect.width())), int(round(rect.height()))
    if frame_w <= 0 or frame_h <= 0:
        return
    rgba = overlay_rgba(
        main_window, aea.current_frame_number(main_window), frame_w, frame_h
    )
    if rgba is None:
        return
    image = QtGui.QImage(
        rgba.data, frame_w, frame_h, 4 * frame_w, QtGui.QImage.Format.Format_RGBA8888
    ).copy()
    item = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap.fromImage(image))
    item.setData(0, aea.OVERLAY_ITEM_TAG)
    item.setPos(rect.topLeft())
    item.setZValue(aea._OVERLAY_Z - 1)
    item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
    main_window.graphicsViewFrame.scene().addItem(item)
    setattr(main_window, _OVERLAY_ATTR, [item])


# --- Mouse -----------------------------------------------------------------
def handle_mouse_press(
    main_window: "MainWindow", view: QtWidgets.QGraphicsView, event: QtGui.QMouseEvent
) -> bool:
    if event.button() != QtCore.Qt.MouseButton.LeftButton:
        return False
    if not is_mask_paint_active(main_window):
        return False
    rect = aea._pixmap_rect(main_window)
    pos = view.mapToScene(event.pos())
    if rect is None or not rect.contains(pos):
        return True
    frame_number = aea.current_frame_number(main_window)
    faces = aea._frame_faces(main_window, frame_number)
    face_index = aea.nearest_face_index(
        faces, *aea._to_normalized(rect, pos.x(), pos.y())
    )
    if face_index is None:
        return True
    set_active_face(main_window, frame_number, face_index)
    brush = brush_settings(main_window)
    shift = bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
    erase = bool(brush["erase"]) != shift
    color = QtGui.QColor(*(_ERASE_RGB if erase else _ADD_RGB))
    color.setAlphaF(max(0.15, float(brush["opacity"]) / 100.0))
    pen = QtGui.QPen(color, float(brush["size"]))
    pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
    path = QtGui.QPainterPath(pos)
    path.lineTo(pos)
    item = QtWidgets.QGraphicsPathItem(path)
    item.setPen(pen)
    item.setZValue(aea._OVERLAY_Z)
    view.scene().addItem(item)
    setattr(
        main_window,
        _STROKE_ATTR,
        {
            "frame": frame_number,
            "face": face_index,
            "erase": erase,
            "points": [(pos.x() - rect.left(), pos.y() - rect.top())],
            "item": item,
            "path": path,
            "size": (rect.width(), rect.height()),
        },
    )
    return True


def hide_brush_cursor(main_window: "MainWindow") -> None:
    items = getattr(main_window, _CURSOR_ATTR, None)
    setattr(main_window, _CURSOR_ATTR, None)
    for item in items or ():
        try:
            if item.scene() is not None:
                item.scene().removeItem(item)
        except RuntimeError:  # already deleted with the scene
            pass


def update_brush_cursor(
    main_window: "MainWindow", view: QtWidgets.QGraphicsView, pos: QtCore.QPointF
) -> None:
    """Circle showing the brush size (solid) and where feathering starts (dashed)."""
    hide_brush_cursor(main_window)
    rect = aea._pixmap_rect(main_window)
    if not is_mask_paint_active(main_window) or rect is None or not rect.contains(pos):
        return
    brush = brush_settings(main_window)
    radius = float(brush["size"]) / 2.0
    inner = radius - float(brush["feather"])
    items = []
    for r, style in (
        (radius, QtCore.Qt.PenStyle.SolidLine),
        (inner, QtCore.Qt.PenStyle.DashLine),
    ):
        if r <= 0 or (style == QtCore.Qt.PenStyle.DashLine and r >= radius):
            continue
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 220), 1, style)
        pen.setCosmetic(True)
        item = QtWidgets.QGraphicsEllipseItem(pos.x() - r, pos.y() - r, 2 * r, 2 * r)
        item.setPen(pen)
        item.setZValue(aea._OVERLAY_Z + 1)
        item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        item.setData(0, aea.OVERLAY_ITEM_TAG)
        view.scene().addItem(item)
        items.append(item)
    setattr(main_window, _CURSOR_ATTR, items)


def handle_mouse_move(
    main_window: "MainWindow", view: QtWidgets.QGraphicsView, event: QtGui.QMouseEvent
) -> bool:
    if is_mask_paint_active(main_window):
        update_brush_cursor(main_window, view, view.mapToScene(event.pos()))
    stroke = getattr(main_window, _STROKE_ATTR, None)
    if stroke is None:
        return False
    rect = aea._pixmap_rect(main_window)
    if rect is None:
        return True
    pos = view.mapToScene(event.pos())
    stroke["points"].append((pos.x() - rect.left(), pos.y() - rect.top()))
    stroke["path"].lineTo(pos)
    stroke["item"].setPath(stroke["path"])
    return True


def handle_mouse_release(
    main_window: "MainWindow", view: QtWidgets.QGraphicsView, event: QtGui.QMouseEvent
) -> bool:
    stroke = getattr(main_window, _STROKE_ATTR, None)
    if stroke is None or event.button() != QtCore.Qt.MouseButton.LeftButton:
        return False
    setattr(main_window, _STROKE_ATTR, None)
    try:
        view.scene().removeItem(stroke["item"])
    except RuntimeError:
        pass
    frame_w, frame_h = stroke["size"]
    apply_stroke(
        main_window,
        stroke["frame"],
        stroke["face"],
        stroke["points"],
        frame_w,
        frame_h,
        stroke["erase"],
    )
    return True


# --- Context menu ----------------------------------------------------------
def add_context_menu_actions(
    main_window: "MainWindow", menu: QtWidgets.QMenu, scene_pos: QtCore.QPointF
) -> dict[QtGui.QAction, Callable[[], Any]]:
    handlers: dict[QtGui.QAction, Callable[[], Any]] = {}
    if not is_mask_paint_active(main_window):
        return handlers
    frame_number = aea.current_frame_number(main_window)
    rect = aea._pixmap_rect(main_window)
    point = (
        aea._to_normalized(rect, scene_pos.x(), scene_pos.y())
        if rect is not None
        else (0.5, 0.5)
    )
    face_index = aea.nearest_face_index(
        aea._frame_faces(main_window, frame_number), *point
    )
    if face_index is not None:
        set_active_face(main_window, frame_number, face_index)
        _sync_bar_buttons(main_window)
    return handlers
