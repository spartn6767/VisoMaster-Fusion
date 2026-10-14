"""Manual alignment corrections in the viewer ("Edit Alignments" mode).

Faces are shown over the paused frame: drag a landmark to override it, drag on an
empty area to add a missed face. Edits are stored as ``source: "manual"`` in
``main_window.alignments``; the frame worker fills in the rest (landmarks,
embedding) on the next refresh.
"""

from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from app.processors.video_utils.alignment_store import (
    SOURCE_MANUAL,
    alignments_cache_enabled,
)

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

# ArcFace 5-point template (112x112); starting points for a drawn bbox.
_ARCFACE_TEMPLATE = (
    np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    / 112.0
)
_MIN_BBOX_PIXELS = 8.0
_HANDLE_RADIUS = 5.0
_OVERLAY_Z = 10.0
_OVERLAY_ATTR = "_alignment_edit_items"
_RUBBER_BAND_ATTR = "_alignment_rubber_band"


def template_kps_for_bbox(bbox: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    pts = _ARCFACE_TEMPLATE.copy()
    pts[:, 0] = x1 + pts[:, 0] * (x2 - x1)
    pts[:, 1] = y1 + pts[:, 1] * (y2 - y1)
    return pts


def bbox_from_corners(p1: Sequence[float], p2: Sequence[float]) -> np.ndarray:
    return np.array(
        [min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]), max(p1[1], p2[1])],
        dtype=np.float32,
    )


def face_index_at(faces: Sequence[dict], x: float, y: float) -> Optional[int]:
    """Index of the smallest face bbox containing (x, y)."""
    best: Optional[int] = None
    best_area = 0.0
    for idx, face in enumerate(faces):
        x1, y1, x2, y2 = (float(v) for v in face["bbox"])
        if x1 <= x <= x2 and y1 <= y <= y2:
            area = (x2 - x1) * (y2 - y1)
            if best is None or area < best_area:
                best, best_area = idx, area
    return best


def current_frame_number(main_window: "MainWindow") -> int:
    return int(main_window.videoSeekSlider.value())


def can_edit_alignments(main_window: "MainWindow") -> bool:
    video_processor = main_window.video_processor
    return bool(
        getattr(main_window, "alignments", None) is not None
        and video_processor.file_type == "video"
        and video_processor.media_path
        and not video_processor.processing
        and alignments_cache_enabled(main_window.control)
        and not getattr(main_window, "view_face_compare_enabled", False)
        and not getattr(main_window, "view_face_mask_enabled", False)
    )


def is_alignment_edit_active(main_window: "MainWindow") -> bool:
    return bool(getattr(main_window, "alignment_edit_enabled", False)) and (
        can_edit_alignments(main_window)
    )


def set_alignment_edit_mode(main_window: "MainWindow", enabled: bool) -> None:
    main_window.alignment_edit_enabled = bool(enabled)
    if enabled and not can_edit_alignments(main_window):
        QtWidgets.QMessageBox.information(
            main_window,
            "Edit Alignments",
            "Alignments can be edited on a paused video frame with "
            "Settings > Detectors > Reuse Saved Alignments on, and Face Compare, "
            "Face Mask view, VR180 and Manual Rotation off.",
        )
    refresh_alignment_overlay(main_window)


def _refresh(main_window: "MainWindow") -> None:
    from app.ui.widgets.actions import common_actions

    QtCore.QTimer.singleShot(0, lambda: common_actions.refresh_frame(main_window))


def _pixmap_rect(main_window: "MainWindow") -> Optional[QtCore.QRectF]:
    for item in main_window.graphicsViewFrame.scene().items():
        if isinstance(item, QtWidgets.QGraphicsPixmapItem):
            rect = item.sceneBoundingRect()
            return rect if rect.width() > 0 and rect.height() > 0 else None
    return None


def _to_normalized(rect: QtCore.QRectF, x: float, y: float) -> tuple[float, float]:
    return (x - rect.left()) / rect.width(), (y - rect.top()) / rect.height()


def _frame_faces(main_window: "MainWindow", frame_number: int) -> list[dict]:
    return main_window.alignments.get(frame_number, 1, 1, None) or []


def add_manual_face(
    main_window: "MainWindow", frame_number: int, bbox: np.ndarray
) -> int:
    """Adds a face from a normalised bbox; the worker detects its landmarks."""
    store = main_window.alignments
    store.bind_media(main_window.video_processor.media_path)
    index = store.add_face(
        frame_number,
        {"bbox": bbox, "kps_5": template_kps_for_bbox(bbox), "refine": True},
        1,
        1,
    )
    _refresh(main_window)
    return index


def move_landmark(
    main_window: "MainWindow",
    frame_number: int,
    face_index: int,
    point_index: int,
    x: float,
    y: float,
) -> bool:
    """Overrides one kps_5 point (normalised coords) and pins the face as manual."""
    faces = _frame_faces(main_window, frame_number)
    if not 0 <= face_index < len(faces):
        return False
    kps = np.asarray(faces[face_index]["kps_5"], dtype=np.float32).copy()
    kps[point_index] = (x, y)
    updated = main_window.alignments.update_face(
        frame_number,
        face_index,
        1,
        1,
        kps_5=np.clip(kps, 0.0, 1.0),
        kps_all=None,
        kps_203=None,
        embedding=None,
        source=SOURCE_MANUAL,
    )
    if updated:
        _refresh(main_window)
    return updated


def delete_face(main_window: "MainWindow", frame_number: int, face_index: int) -> None:
    if main_window.alignments.remove_face(frame_number, face_index):
        _refresh(main_window)


def redetect_frame(main_window: "MainWindow", frame_number: int) -> None:
    main_window.alignments.remove_frame(frame_number)
    _refresh(main_window)


class _LandmarkHandle(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, on_moved: Callable[[QtCore.QPointF], Any]):
        r = _HANDLE_RADIUS
        super().__init__(-r, -r, 2 * r, 2 * r)
        self._on_moved = on_moved
        self._press_pos: Optional[QtCore.QPointF] = None
        self.setBrush(QtGui.QBrush(QtGui.QColor(255, 60, 60)))
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255))
        pen.setCosmetic(True)
        self.setPen(pen)
        flags = QtWidgets.QGraphicsItem.GraphicsItemFlag
        self.setFlag(flags.ItemIsMovable, True)
        self.setFlag(flags.ItemIgnoresTransformations, True)
        self.setZValue(_OVERLAY_Z + 1)
        self.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)

    def mousePressEvent(self, event):
        self._press_pos = self.pos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self._press_pos is not None and self.pos() != self._press_pos:
            self._on_moved(self.pos())
        self._press_pos = None


def _commit_handle(
    main_window: "MainWindow",
    frame_number: int,
    face_index: int,
    point_index: int,
    scene_pos: QtCore.QPointF,
) -> None:
    rect = _pixmap_rect(main_window)
    if rect is None:
        return
    x, y = _to_normalized(rect, scene_pos.x(), scene_pos.y())
    move_landmark(main_window, frame_number, face_index, point_index, x, y)


def clear_alignment_overlay(main_window: "MainWindow") -> None:
    scene = main_window.graphicsViewFrame.scene()
    for item in getattr(main_window, _OVERLAY_ATTR, []):
        try:
            if item.scene() is scene:
                scene.removeItem(item)
        except RuntimeError:  # already deleted by scene.clear()
            pass
    setattr(main_window, _OVERLAY_ATTR, [])


def refresh_alignment_overlay(main_window: "MainWindow") -> None:
    """Redraws the editable boxes and landmarks for the current frame."""
    clear_alignment_overlay(main_window)
    if not is_alignment_edit_active(main_window):
        return
    rect = _pixmap_rect(main_window)
    if rect is None:
        return
    scene = main_window.graphicsViewFrame.scene()
    frame_number = current_frame_number(main_window)
    items: list[QtWidgets.QGraphicsItem] = []
    for face_index, face in enumerate(_frame_faces(main_window, frame_number)):
        manual = face.get("source") == SOURCE_MANUAL
        pen = QtGui.QPen(
            QtGui.QColor(0, 220, 120) if manual else QtGui.QColor(255, 200, 0)
        )
        pen.setCosmetic(True)
        pen.setWidth(2)
        x1, y1, x2, y2 = (float(v) for v in face["bbox"])
        box = QtWidgets.QGraphicsRectItem(
            rect.left() + x1 * rect.width(),
            rect.top() + y1 * rect.height(),
            (x2 - x1) * rect.width(),
            (y2 - y1) * rect.height(),
        )
        box.setPen(pen)
        box.setZValue(_OVERLAY_Z)
        box.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        scene.addItem(box)
        items.append(box)
        for point_index, (px, py) in enumerate(
            np.asarray(face["kps_5"], dtype=np.float64)[:5]
        ):
            handle = _LandmarkHandle(
                partial(
                    _commit_handle, main_window, frame_number, face_index, point_index
                )
            )
            handle.setPos(
                rect.left() + px * rect.width(), rect.top() + py * rect.height()
            )
            scene.addItem(handle)
            items.append(handle)
    setattr(main_window, _OVERLAY_ATTR, items)


def handle_mouse_press(
    main_window: "MainWindow", view: QtWidgets.QGraphicsView, event: QtGui.QMouseEvent
) -> bool:
    """Starts drawing a bbox; returns True if the event was consumed."""
    if event.button() != QtCore.Qt.MouseButton.LeftButton:
        return False
    if not is_alignment_edit_active(main_window):
        return False
    if isinstance(view.itemAt(event.pos()), _LandmarkHandle):
        return False
    rect = _pixmap_rect(main_window)
    start = view.mapToScene(event.pos())
    if rect is None or not rect.contains(start):
        return True
    band = QtWidgets.QGraphicsRectItem(QtCore.QRectF(start, start))
    pen = QtGui.QPen(QtGui.QColor(0, 220, 120))
    pen.setCosmetic(True)
    pen.setStyle(QtCore.Qt.PenStyle.DashLine)
    band.setPen(pen)
    band.setZValue(_OVERLAY_Z)
    view.scene().addItem(band)
    setattr(main_window, _RUBBER_BAND_ATTR, (start, band))
    return True


def handle_mouse_move(
    main_window: "MainWindow", view: QtWidgets.QGraphicsView, event: QtGui.QMouseEvent
) -> bool:
    state = getattr(main_window, _RUBBER_BAND_ATTR, None)
    if state is None:
        return False
    start, band = state
    band.setRect(QtCore.QRectF(start, view.mapToScene(event.pos())).normalized())
    return True


def handle_mouse_release(
    main_window: "MainWindow", view: QtWidgets.QGraphicsView, event: QtGui.QMouseEvent
) -> bool:
    state = getattr(main_window, _RUBBER_BAND_ATTR, None)
    if state is None or event.button() != QtCore.Qt.MouseButton.LeftButton:
        return False
    setattr(main_window, _RUBBER_BAND_ATTR, None)
    start, band = state
    try:
        view.scene().removeItem(band)
    except RuntimeError:
        pass
    end = view.mapToScene(event.pos())
    rect = _pixmap_rect(main_window)
    if (
        rect is None
        or abs(end.x() - start.x()) < _MIN_BBOX_PIXELS
        or abs(end.y() - start.y()) < _MIN_BBOX_PIXELS
    ):
        return True
    bbox = bbox_from_corners(
        _to_normalized(rect, start.x(), start.y()),
        _to_normalized(rect, end.x(), end.y()),
    )
    add_manual_face(main_window, current_frame_number(main_window), np.clip(bbox, 0, 1))
    return True


def add_context_menu_actions(
    main_window: "MainWindow", menu: QtWidgets.QMenu, scene_pos: QtCore.QPointF
) -> dict[QtGui.QAction, Callable[[], None]]:
    """Adds the Edit Alignments entries; returns handlers keyed by action."""
    handlers: dict[QtGui.QAction, Callable[[], None]] = {}
    menu.addSeparator()
    toggle = menu.addAction("Edit Alignments")
    toggle.setCheckable(True)
    toggle.setChecked(bool(getattr(main_window, "alignment_edit_enabled", False)))
    handlers[toggle] = lambda: set_alignment_edit_mode(main_window, toggle.isChecked())
    if not is_alignment_edit_active(main_window):
        return handlers
    frame_number = current_frame_number(main_window)
    rect = _pixmap_rect(main_window)
    faces = _frame_faces(main_window, frame_number)
    if rect is not None and faces:
        face_index = face_index_at(
            faces, *_to_normalized(rect, scene_pos.x(), scene_pos.y())
        )
        if face_index is not None:
            delete_action = menu.addAction("Delete Face")
            handlers[delete_action] = partial(
                delete_face, main_window, frame_number, face_index
            )
    if frame_number in main_window.alignments:
        redetect_action = menu.addAction("Re-detect Frame")
        handlers[redetect_action] = partial(redetect_frame, main_window, frame_number)
    return handlers
