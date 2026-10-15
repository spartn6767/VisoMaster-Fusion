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
_ADD_RGB = (0, 220, 120)
_ERASE_RGB = (255, 60, 60)
DEFAULT_BRUSH = {"size": 40, "feather": 6, "opacity": 45, "erase": False}


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
            bar.raise_()
    refresh_mask_overlay(main_window)
    from app.ui.widgets.actions import edit_toolbar_actions

    edit_toolbar_actions.sync_edit_toolbar(main_window)


def _build_brush_bar(main_window: "MainWindow") -> QtWidgets.QFrame:
    brush = brush_settings(main_window)
    bar = QtWidgets.QFrame(main_window.graphicsViewFrame)
    bar.setObjectName("maskBrushBar")
    bar.setStyleSheet(
        "#maskBrushBar { background: rgba(20, 20, 20, 200); border-radius: 4px; }"
    )
    layout = QtWidgets.QHBoxLayout(bar)
    layout.setContentsMargins(6, 4, 6, 4)

    erase = QtWidgets.QCheckBox("Erase (Shift)", bar)
    erase.setChecked(bool(brush["erase"]))
    erase.toggled.connect(lambda checked: brush.__setitem__("erase", bool(checked)))
    layout.addWidget(erase)

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
    bar.adjustSize()
    bar.move(8, 8)
    setattr(main_window, _BRUSH_BAR_ATTR, bar)
    return bar


# --- Mask edits ------------------------------------------------------------
def _face_kps_px(face: dict, frame_w: float, frame_h: float) -> np.ndarray:
    return np.asarray(face["kps_5"], dtype=np.float64)[:5] * (frame_w, frame_h)


def set_mask(
    main_window: "MainWindow",
    frame_number: int,
    face_index: int,
    mask: Optional[np.ndarray],
    skip: bool = False,
) -> bool:
    """Stores a mask keyframe on a face; the frame becomes manual (pinned)."""
    store = main_window.alignments
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
    return set_mask(main_window, frame_number, face_index, manual_masks.empty_mask())


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


def reset_all_masks(main_window: "MainWindow") -> int:
    if not _confirm_reset_all(main_window):
        return 0
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


def handle_mouse_move(
    main_window: "MainWindow", view: QtWidgets.QGraphicsView, event: QtGui.QMouseEvent
) -> bool:
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
    toggle = menu.addAction("Paint Mask")
    toggle.setCheckable(True)
    toggle.setChecked(bool(getattr(main_window, "mask_paint_enabled", False)))
    handlers[toggle] = lambda: set_mask_paint_mode(main_window, toggle.isChecked())
    if not is_mask_paint_active(main_window):
        return handlers
    store = main_window.alignments
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
        has_mask = store.effective_mask(frame_number, face_index) is not None
        copy_action = menu.addAction("Copy Mask")
        copy_action.setEnabled(has_mask)
        handlers[copy_action] = partial(
            copy_mask, main_window, frame_number, face_index
        )
        paste_action = menu.addAction("Paste Mask")
        paste_action.setEnabled(getattr(main_window, _CLIPBOARD_ATTR, None) is not None)
        handlers[paste_action] = partial(
            paste_mask, main_window, frame_number, face_index
        )
        for label, other_frame in aea.adjacent_frame_labels(
            store, frame_number, "Paste Mask Previous", "Paste Mask Next"
        ):
            other_index = (
                None
                if other_frame is None
                else aea.nearest_face_index(
                    aea._frame_faces(main_window, other_frame), *point
                )
            )
            action = menu.addAction(label)
            action.setEnabled(other_index is not None)
            if other_frame is not None and other_index is not None:
                handlers[action] = partial(
                    paste_mask_from_frame,
                    main_window,
                    other_frame,
                    other_index,
                    frame_number,
                    face_index,
                )
        for label, reset in (
            ("Reset Mask (This Frame)", reset_mask_frame),
            ("Reset Mask (From Here On)", reset_mask),
        ):
            reset_action = menu.addAction(label)
            reset_action.setEnabled(has_mask)
            handlers[reset_action] = partial(
                reset, main_window, frame_number, face_index
            )
    reset_all_action = menu.addAction("Reset All Masks")
    reset_all_action.setEnabled(store.mask_frame_count() > 0)
    handlers[reset_all_action] = partial(reset_all_masks, main_window)
    return handlers
