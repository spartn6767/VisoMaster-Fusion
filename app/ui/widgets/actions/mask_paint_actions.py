"""Hand-painted face masks in the viewer ("Paint Mask" mode).

Strokes are stored on the face's alignment entry in canonical face space (see
``manual_masks``) and override the computed swap mask, or, with the bar's Hair
Mask layer, Hair Makeup's computed hair mask. A painted mask holds on the
same target face in later frames until the next frame painted for that face;
"Reset Mask to Settings" writes an empty keyframe so the computed mask applies again.
"""

from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from app.processors.video_utils import manual_masks
from app.processors.video_utils.alignment_store import SOURCE_MANUAL, _nearest
from app.ui.widgets.actions import alignment_edit_actions as aea

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

_OVERLAY_ATTR = "_mask_paint_items"
_STROKE_ATTR = "_mask_paint_stroke"
_BRUSH_BAR_ATTR = "_mask_brush_bar"
_CLIPBOARD_ATTR = "mask_clipboard"
_ACTIVE_FACE_ATTR = "_mask_active_face"
_BAR_BUTTONS_ATTR = "_mask_bar_buttons"
_PASTE_LABEL_ATTR = "_mask_paste_label"
_PASTE_RUN_ATTR = "_paste_run"
_CURSOR_ATTR = "_mask_brush_cursor"
_ERASE_BOX_ATTR = "_mask_erase_box"
_LAYER_BOX_ATTR = "_mask_layer_box"
_MIN_DRAG_PIXELS = 3  # screen pixels; a click without a drag paints nothing
_ADD_RGB = (0, 220, 120)
_ERASE_RGB = (255, 60, 60)
DEFAULT_BRUSH = {
    "size": 40,
    "feather": 6,
    "opacity": 45,
    "erase": False,
    "frame_only": False,
    "layer": "mask",
}
# Which mask the brush edits: the swap mask, or Hair Makeup's hair mask.
LAYERS = {"mask": manual_masks.SWAP_SPACE, "hair_mask": manual_masks.HAIR_SPACE}
LAYER_NAMES = {"mask": "Swap Mask", "hair_mask": "Hair Mask"}


def brush_settings(main_window: "MainWindow") -> dict[str, Any]:
    brush = getattr(main_window, "mask_brush", None)
    if not isinstance(brush, dict):
        brush = dict(DEFAULT_BRUSH)
        main_window.mask_brush = brush
    return brush


def set_erase(main_window: "MainWindow", erase: bool) -> None:
    """Brush erases (True) or paints (False); keeps the bar's Erase box in step."""
    brush_settings(main_window)["erase"] = bool(erase)
    box = getattr(main_window, _ERASE_BOX_ATTR, None)
    if box is None:
        return
    try:
        box.setChecked(bool(erase))
    except RuntimeError:  # bar already deleted
        setattr(main_window, _ERASE_BOX_ATTR, None)


def current_layer(main_window: "MainWindow") -> str:
    layer = brush_settings(main_window).get("layer")
    return layer if layer in LAYERS else "mask"


def _space(main_window: "MainWindow") -> manual_masks.MaskSpace:
    return LAYERS[current_layer(main_window)]


def _layer_fields(
    main_window: "MainWindow", mask: Optional[np.ndarray], skip: bool = False
) -> dict[str, Any]:
    layer = current_layer(main_window)
    return {layer: mask, f"{layer}_skip": skip}


def _clipboard_attr(main_window: "MainWindow") -> str:
    layer = current_layer(main_window)
    return _CLIPBOARD_ATTR if layer == "mask" else f"{layer}_clipboard"


def set_layer(main_window: "MainWindow", layer: str) -> None:
    """Brush edits the swap mask ("mask") or the hair mask ("hair_mask")."""
    if layer not in LAYERS:
        return
    brush_settings(main_window)["layer"] = layer
    setattr(main_window, _PASTE_RUN_ATTR, None)
    box = getattr(main_window, _LAYER_BOX_ATTR, None)
    if box is not None:
        try:
            box.setCurrentIndex(list(LAYERS).index(layer))
        except RuntimeError:  # bar already deleted
            setattr(main_window, _LAYER_BOX_ATTR, None)
    refresh_mask_overlay(main_window)


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
                "Settings > Detectors > Reuse Stored Alignments on, and Face Compare, "
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
    ("reset_all", "Reset All", "Remove every painted mask in this video."),
)
# Right-click menu on a face box (edit_toolbar_actions.face_menu_at).
FACE_ACTIONS = (
    ("copy", "Copy Mask", "Copy this face's mask (C)."),
    ("paste", "Paste Mask", "Paste the copied mask onto this face (V)."),
    (
        "previous",
        "< Paste",
        "Paste the mask from the nearest earlier stored frame (Z). "
        "Press again for the next earlier one.",
    ),
    (
        "next",
        "Paste >",
        "Paste the mask from the nearest later stored frame (X). "
        "Press again for the next later one.",
    ),
    ("reset_frame", "Reset Frame", "Computed mask on this frame only (M)."),
    ("reset_on", "Reset From Here", "Computed mask from this frame on."),
    (
        "range",
        "Apply to Range...",
        "Use this face's mask on every frame from one frame to another.",
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


def _source_face(
    main_window: "MainWindow", frame_number: int, center: tuple[float, float]
) -> Optional[int]:
    return aea.nearest_face_index(aea._frame_faces(main_window, frame_number), *center)


def _same_mask(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return bool(np.array_equal(a, b))


def _paste_run(
    main_window: "MainWindow", mode: str, frame_number: int, face_index: int
) -> Optional[dict]:
    """The < Paste / Paste > run still open on this frame and face, if nothing
    else changed since its last paste."""
    run = getattr(main_window, _PASTE_RUN_ATTR, None)
    if (
        run is not None
        and run["mode"] == mode
        and run["frame"] == frame_number
        and run["face"] == face_index
        and run["revision"] == main_window.alignments.revision
    ):
        return run
    return None


def paste_step(
    main_window: "MainWindow", mode: str, frame_number: int, face_index: int, step: int
) -> bool:
    """< Paste (step -1) / Paste > (step 1) for mode "align" or "mask".

    Repeated presses walk one row of stored frames, replacing the previous paste:
    Z goes one frame earlier, X one later, and reaching this frame puts the
    original back. The whole run is one undo step. In mask mode, frames whose mask
    matches the one shown are skipped."""
    store = main_window.alignments
    faces = aea._frame_faces(main_window, frame_number)
    run = _paste_run(main_window, mode, frame_number, face_index)
    if run is None and not 0 <= face_index < len(faces):
        return False
    if run is None:
        x1, y1, x2, y2 = faces[face_index]["bbox"]
        run = {
            "mode": mode,
            "frame": frame_number,
            "face": face_index,
            "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            "source": frame_number,
            "pasted": False,
        }
    frames = store.face_frames()
    target = _nearest(frames, run["source"], step)
    if mode == "mask" and not run.get("added"):
        layer = current_layer(main_window)
        shown = store.effective_mask(frame_number, face_index, layer=layer)

        def source_mask(n: int) -> Optional[np.ndarray]:
            index = _source_face(main_window, n, run["center"])
            return (
                None if index is None else store.effective_mask(n, index, layer=layer)
            )

        while (
            target is not None
            and target != frame_number
            and _same_mask(source_mask(target), shown)
        ):
            target = _nearest(frames, target, step)
    if target is None:
        return False
    source_index = None
    if target != frame_number:
        source_index = _source_face(main_window, target, run["center"])
        if source_index is None:
            return False
    if run["pasted"]:
        store.undo()
    before = store.revision
    if source_index is None:
        aea._refresh(main_window)
        if mode == "mask":
            refresh_mask_overlay(main_window)
    elif run.get("added"):
        _add_pasted_face(main_window, mode, target, source_index, frame_number)
    elif mode == "mask":
        paste_mask_from_frame(
            main_window, target, source_index, frame_number, face_index
        )
    else:
        aea.paste_from_frame(
            main_window, target, source_index, frame_number, face_index
        )
    run.update(
        source=target,
        pasted=source_index is not None and store.revision != before,
        revision=store.revision,
    )
    setattr(main_window, _PASTE_RUN_ATTR, run)
    return True


def open_run_face(
    main_window: "MainWindow", mode: str, frame_number: int
) -> Optional[int]:
    """Face of the run still open on this frame (it may be absent after a
    double-click run walked back to before the face was added)."""
    run = getattr(main_window, _PASTE_RUN_ATTR, None)
    if run is None:
        return None
    return (
        run["face"]
        if _paste_run(main_window, mode, frame_number, run["face"])
        else None
    )


DEFAULT_NEW_FACE_HEIGHT = 0.2  # of the frame height


def _is_near(bbox: Any, x: float, y: float) -> bool:
    """Within one box width / height of the box centre."""
    x1, y1, x2, y2 = bbox
    return abs(x - (x1 + x2) / 2.0) <= x2 - x1 and abs(y - (y1 + y2) / 2.0) <= y2 - y1


def _add_pasted_face(
    main_window: "MainWindow",
    mode: str,
    source_frame: int,
    source_index: int,
    frame_number: int,
) -> None:
    """Adds a copy of a face from another frame; in mask mode with its mask too."""
    store = main_window.alignments
    with store.edit([frame_number]):
        aea.paste_from_frame(main_window, source_frame, source_index, frame_number)
        mask = store.effective_mask(
            source_frame, source_index, layer=current_layer(main_window)
        )
        if mode == "mask" and mask is not None:
            index = len(aea._frame_faces(main_window, frame_number)) - 1
            set_mask(main_window, frame_number, index, mask.copy())
    if mode == "mask":
        refresh_mask_overlay(main_window)


def _select(
    main_window: "MainWindow", mode: str, frame_number: int, index: int
) -> None:
    if mode == "mask":
        set_active_face(main_window, frame_number, index)
    else:
        aea.select_face(main_window, frame_number, index)


def paste_new_face(
    main_window: "MainWindow",
    frame_number: int,
    point: tuple[float, float],
    aspect: float = 1.0,
    mode: str = "align",
) -> int:
    """Adds a face at a point where none is stored (normalised coords; aspect is
    frame height / width). Pastes the nearest earlier frame's face (in mask mode
    with its mask) if it is near the point and opens a Z / X run on it, else draws
    a box centred on the point sized like that face (or a default size) for the
    landmarks to be fixed by hand."""
    store = main_window.alignments
    x, y = point
    index = len(aea._frame_faces(main_window, frame_number))
    height = DEFAULT_NEW_FACE_HEIGHT
    width = height * aspect
    source = _nearest(store.face_frames(), frame_number, -1)
    source_index = None if source is None else _source_face(main_window, source, point)
    if source is not None and source_index is not None:
        bbox = aea._frame_faces(main_window, source)[source_index]["bbox"]
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if _is_near(bbox, x, y):
            before = store.revision
            _add_pasted_face(main_window, mode, source, source_index, frame_number)
            setattr(
                main_window,
                _PASTE_RUN_ATTR,
                {
                    "mode": mode,
                    "frame": frame_number,
                    "face": index,
                    "center": (x, y),
                    "source": source,
                    "pasted": store.revision != before,
                    "added": True,
                    "revision": store.revision,
                },
            )
            _select(main_window, mode, frame_number, index)
            return index
    box = np.array(
        [x - width / 2, y - height / 2, x + width / 2, y + height / 2], np.float32
    )
    index = aea.add_manual_face(main_window, frame_number, np.clip(box, 0.0, 1.0))
    _select(main_window, mode, frame_number, index)
    if mode == "mask":
        refresh_mask_overlay(main_window)
    return index


def paste_or_add_mask_at(
    main_window: "MainWindow",
    frame_number: int,
    x: float,
    y: float,
    aspect: float = 1.0,
) -> int:
    """Paint Mask double-click. On or near a face: paste the earlier frame's mask,
    as < Paste does. Elsewhere: add the earlier frame's face with its mask, or a
    box centred here if none is near."""
    faces = aea._frame_faces(main_window, frame_number)
    index = aea.face_index_at(faces, x, y) if faces else None
    if index is None:
        nearest = aea.nearest_face_index(faces, x, y)
        if nearest is not None and _is_near(faces[nearest]["bbox"], x, y):
            index = nearest
    if index is not None:
        set_active_face(main_window, frame_number, index)
        paste_step(main_window, "mask", frame_number, index, -1)
    else:
        index = paste_new_face(main_window, frame_number, (x, y), aspect, mode="mask")
    _sync_bar_buttons(main_window)
    return index


def paste_source_text(
    main_window: "MainWindow", mode: str, face_index: Optional[int]
) -> str:
    """ "From frame N" while the shown face holds a < Paste / Paste > result."""
    frame_number = aea.current_frame_number(main_window)
    run = getattr(main_window, _PASTE_RUN_ATTR, None)
    if run is not None and run["frame"] != frame_number:
        setattr(main_window, _PASTE_RUN_ATTR, None)
        return ""
    if face_index is None:
        return ""
    run = _paste_run(main_window, mode, frame_number, face_index)
    if run is None or run["source"] == frame_number:
        return ""
    return f"From frame {run['source']}"


def paste_source_label(parent: QtWidgets.QWidget) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(parent)
    label.setToolTip(
        "Frame the current < Paste / Paste > came from. Press Z / X again to try "
        "the next one; reaching this frame puts the original back."
    )
    return label


def toolbar_action(main_window: "MainWindow", key: str, *args: Any) -> Any:
    """Runs a brush-bar action on the selected face of the shown frame."""
    if key == "reset_all":
        return reset_all_masks(main_window)
    if key in ("undo", "redo"):
        return aea.undo_edit(main_window, redo=key == "redo")
    frame_number = aea.current_frame_number(main_window)
    face_index = active_face(main_window)
    if face_index is None and key in ("previous", "next"):
        face_index = open_run_face(main_window, "mask", frame_number)
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
        result = paste_step(
            main_window,
            "mask",
            frame_number,
            face_index,
            -1 if key == "previous" else 1,
        )
    _sync_bar_buttons(main_window)
    return result


def action_states(main_window: "MainWindow") -> dict[str, bool]:
    """Which brush-bar and face-menu actions apply to the selected face."""
    store = main_window.alignments
    frame_number = aea.current_frame_number(main_window)
    face_index = active_face(main_window)
    enabled = {
        "reset_all": store.mask_frame_count(current_layer(main_window)) > 0,
        "undo": store.can_undo(),
        "redo": store.can_redo(),
    }
    if face_index is not None:
        has_mask = (
            store.effective_mask(
                frame_number, face_index, layer=current_layer(main_window)
            )
            is not None
        )
        enabled.update(
            copy=has_mask,
            paste=getattr(main_window, _clipboard_attr(main_window), None) is not None,
            previous=_adjacent_source(main_window, frame_number, face_index, -1)
            is not None,
            next=_adjacent_source(main_window, frame_number, face_index, 1) is not None,
            reset_frame=has_mask,
            reset_on=has_mask,
            range=_face_id(main_window, frame_number, face_index) is not None,
        )
    return enabled


def _sync_bar_buttons(main_window: "MainWindow") -> None:
    buttons = getattr(main_window, _BAR_BUTTONS_ATTR, None)
    if not buttons:
        return
    enabled = action_states(main_window)
    for key, button in buttons.items():
        button.setEnabled(bool(enabled.get(key, False)))
    label = getattr(main_window, _PASTE_LABEL_ATTR, None)
    if label is not None:
        label.setText(paste_source_text(main_window, "mask", active_face(main_window)))


def default_bar_position(bar: QtWidgets.QWidget) -> tuple[int, int]:
    """Bottom-right corner of the viewer, until the user drags the bar."""
    parent = bar.parentWidget()
    if parent is None:
        return (8, 8)
    return (parent.width() - bar.width() - 8, parent.height() - bar.height() - 8)


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

    layer_box = QtWidgets.QComboBox(bar)
    layer_box.addItems(list(LAYER_NAMES.values()))
    layer_box.setCurrentIndex(list(LAYERS).index(current_layer(main_window)))
    layer_box.setToolTip(
        "Swap Mask: where the new face is pasted. Hair Mask: where Hair Makeup "
        "recolors (add missed hair, erase wrongly tinted areas)."
    )
    layer_box.currentIndexChanged.connect(
        lambda i: set_layer(main_window, list(LAYERS)[i])
    )
    layout.addWidget(layer_box)
    setattr(main_window, _LAYER_BOX_ATTR, layer_box)

    erase = QtWidgets.QCheckBox("Erase (Shift)", bar)
    erase.setChecked(bool(brush["erase"]))
    erase.toggled.connect(lambda checked: brush.__setitem__("erase", bool(checked)))
    layout.addWidget(erase)
    setattr(main_window, _ERASE_BOX_ATTR, erase)

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
    label = paste_source_label(bar)
    actions_row.addWidget(label)
    setattr(main_window, _PASTE_LABEL_ATTR, label)
    setattr(main_window, _BAR_BUTTONS_ATTR, bar_buttons)

    bar.adjustSize()
    x, y = brush.get("bar_pos") or default_bar_position(bar)
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
        if face.get(current_layer(main_window)) is None:
            held = store.effective_mask(
                frame_number, index, layer=current_layer(main_window)
            )
            store.update_face(
                frame_number,
                index,
                1,
                1,
                **_layer_fields(
                    main_window,
                    manual_masks.empty_mask(_space(main_window))
                    if held is None
                    else held.copy(),
                ),
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
            **_layer_fields(main_window, mask, skip),
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
        main_window.alignments.effective_mask(
            frame_number, face_index, layer=current_layer(main_window)
        ),
        points,
        _face_kps_px(faces[face_index], frame_w, frame_h),
        radius=float(brush["size"]) / 2.0,
        erase=erase,
        feather=float(brush["feather"]),
        space=_space(main_window),
    )
    return set_mask(main_window, frame_number, face_index, mask)


def reset_mask(main_window: "MainWindow", frame_number: int, face_index: int) -> bool:
    """Back to the computed mask for this face from this frame on."""
    return set_mask(
        main_window,
        frame_number,
        face_index,
        manual_masks.empty_mask(_space(main_window)),
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
        main_window,
        frame_number,
        face_index,
        faces[face_index].get(current_layer(main_window)),
        skip=True,
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
    mask = store.effective_mask(
        frame_number, face_index, layer=current_layer(main_window)
    )
    mask = manual_masks.empty_mask(_space(main_window)) if mask is None else mask.copy()
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
                    **_layer_fields(main_window, None if keyframe_set else mask),
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
        cleared = main_window.alignments.clear_masks(current_layer(main_window))
    if cleared:
        aea._refresh(main_window)
        refresh_mask_overlay(main_window)
    return cleared


def _confirm_reset_all(main_window: "MainWindow") -> bool:
    hair = current_layer(main_window) == "hair_mask"
    answer = QtWidgets.QMessageBox.question(
        main_window,
        "Reset All Hair Masks" if hair else "Reset All Masks",
        "Remove every painted hair mask in this video? Hair Makeup goes back to "
        "the hair mask its settings produce."
        if hair
        else "Remove every painted mask in this video? Faces go back to the masks "
        "your mask settings produce.",
    )
    return answer == QtWidgets.QMessageBox.StandardButton.Yes


def copy_mask(main_window: "MainWindow", frame_number: int, face_index: int) -> bool:
    mask = main_window.alignments.effective_mask(
        frame_number, face_index, layer=current_layer(main_window)
    )
    if mask is None:
        return False
    setattr(main_window, _clipboard_attr(main_window), mask.copy())
    return True


def paste_mask(main_window: "MainWindow", frame_number: int, face_index: int) -> bool:
    copied = getattr(main_window, _clipboard_attr(main_window), None)
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
    mask = main_window.alignments.effective_mask(
        source_frame, source_index, layer=current_layer(main_window)
    )
    return set_mask(
        main_window,
        frame_number,
        face_index,
        manual_masks.empty_mask(_space(main_window)) if mask is None else mask.copy(),
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
    """The worker's computed mask (swap mask, or Hair Makeup's hair mask in Hair
    Mask) for a face on the shown frame, if captured."""
    data = getattr(main_window, "computed_masks", None)
    if not isinstance(data, dict) or data.get("frame") != frame_number:
        return None
    if not main_window.alignments.matches_media(data.get("media_path")):
        return None
    key = "faces" if current_layer(main_window) == "mask" else "hair"
    return (data.get(key) or {}).get(face_index)


def overlay_rgba(
    main_window: "MainWindow", frame_number: int, frame_w: int, frame_h: int
) -> Optional[np.ndarray]:
    """Tinted RGBA preview of the final masks on a frame: the computed mask with
    the painted add/erase applied (green), erased areas in red. None if empty."""
    store = main_window.alignments
    layer, space = current_layer(main_window), _space(main_window)
    final = np.zeros((frame_h, frame_w), dtype=np.float32)
    erased = np.zeros((frame_h, frame_w), dtype=np.float32)
    found = False
    for face_index, face in enumerate(aea._frame_faces(main_window, frame_number)):
        painted = store.effective_mask(frame_number, face_index, layer=layer)
        computed = computed_mask(main_window, frame_number, face_index)
        if painted is None and computed is None:
            continue
        stack = np.zeros((3, space.size, space.size), np.uint8)
        if painted is not None:
            stack[:2] = painted
        if computed is not None:
            stack[2] = computed
        add, erase, base = manual_masks.mask_to_frame(
            stack, _face_kps_px(face, frame_w, frame_h), frame_w, frame_h, space
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
            "start": QtCore.QPoint(event.pos()),
            "moved": False,
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
    """Circle showing the brush size (solid) and where feathering starts (dashed);
    green to paint, red to erase (Erase ticked or Shift held)."""
    hide_brush_cursor(main_window)
    rect = aea._pixmap_rect(main_window)
    if not is_mask_paint_active(main_window) or rect is None or not rect.contains(pos):
        return
    brush = brush_settings(main_window)
    radius = float(brush["size"]) / 2.0
    inner = radius - float(brush["feather"])
    shift = bool(
        QtWidgets.QApplication.keyboardModifiers()
        & QtCore.Qt.KeyboardModifier.ShiftModifier
    )
    color = QtGui.QColor(*(_ERASE_RGB if bool(brush["erase"]) != shift else _ADD_RGB))
    items = []
    for r, style in (
        (radius, QtCore.Qt.PenStyle.SolidLine),
        (inner, QtCore.Qt.PenStyle.DashLine),
    ):
        if r <= 0 or (style == QtCore.Qt.PenStyle.DashLine and r >= radius):
            continue
        pen = QtGui.QPen(color, 1, style)
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
    if (event.pos() - stroke["start"]).manhattanLength() >= _MIN_DRAG_PIXELS:
        stroke["moved"] = True
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
    if not stroke["moved"]:
        return True
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


def handle_double_click(
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
    paste_or_add_mask_at(
        main_window,
        aea.current_frame_number(main_window),
        *aea._to_normalized(rect, pos.x(), pos.y()),
        aspect=rect.height() / rect.width(),
    )
    return True
