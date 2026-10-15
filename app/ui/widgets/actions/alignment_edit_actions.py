"""Manual alignment corrections in the viewer ("Edit Alignments" mode).

Faces are shown over the paused frame: drag a landmark to override it, drag on an
empty area to add a missed face. Edits are stored as ``source: "manual"`` in
``main_window.alignments``; the frame worker fills in the rest (landmarks,
embedding) on the next refresh.
"""

import contextlib
import os
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, Sequence

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from app.processors.video_utils.alignment_store import (
    SOURCE_MANUAL,
    AlignmentStore,
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
_CLIPBOARD_ATTR = "alignment_clipboard"
# Tag (item data key 0) for overlay pixmaps that must not be taken for the frame.
OVERLAY_ITEM_TAG = "viewer_overlay"


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


def nearest_face_index(faces: Sequence[dict], x: float, y: float) -> Optional[int]:
    """Face under (x, y), else the face whose bbox centre is closest."""
    if not faces:
        return None
    hit = face_index_at(faces, x, y)
    if hit is not None:
        return hit
    centres = np.array(
        [
            [(f["bbox"][0] + f["bbox"][2]) / 2.0, (f["bbox"][1] + f["bbox"][3]) / 2.0]
            for f in faces
        ],
        dtype=np.float64,
    )
    return int(np.argmin(np.sum((centres - (x, y)) ** 2, axis=1)))


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
    from app.ui.widgets.actions import mask_paint_actions

    main_window.alignment_edit_enabled = bool(enabled)
    if enabled:
        main_window.mask_paint_enabled = False
        mask_paint_actions.sync_mask_paint_ui(main_window)
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
        if (
            isinstance(item, QtWidgets.QGraphicsPixmapItem)
            and item.data(0) != OVERLAY_ITEM_TAG
        ):
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


def move_face(
    main_window: "MainWindow", frame_number: int, face_index: int, dx: float, dy: float
) -> bool:
    """Translates a face (bbox and landmarks, normalised offsets) and pins it as manual."""
    faces = _frame_faces(main_window, frame_number)
    if not 0 <= face_index < len(faces):
        return False
    face = faces[face_index]
    offset = np.array([dx, dy], dtype=np.float32)
    moved: dict[str, Any] = {
        "bbox": np.asarray(face["bbox"], dtype=np.float32) + np.tile(offset, 2),
        "embedding": None,
        "source": SOURCE_MANUAL,
    }
    for key in ("kps_5", "kps_all", "kps_203"):
        if face.get(key) is not None:
            moved[key] = np.asarray(face[key], dtype=np.float32) + offset
    updated = main_window.alignments.update_face(
        frame_number, face_index, 1, 1, **moved
    )
    if updated:
        _refresh(main_window)
    return updated


def resize_face(
    main_window: "MainWindow",
    frame_number: int,
    face_index: int,
    new_bbox: Sequence[float],
) -> bool:
    """Sets a new bbox (normalised), stretching the landmarks with it, and pins it."""
    faces = _frame_faces(main_window, frame_number)
    if not 0 <= face_index < len(faces):
        return False
    face = faces[face_index]
    old = np.asarray(face["bbox"], dtype=np.float32)
    new = np.asarray(new_bbox, dtype=np.float32).reshape(4)
    old_size = np.maximum(old[2:] - old[:2], 1e-6)
    scale = (new[2:] - new[:2]) / old_size
    updates: dict[str, Any] = {
        "bbox": new,
        "embedding": None,
        "source": SOURCE_MANUAL,
    }
    for key in ("kps_5", "kps_all", "kps_203"):
        if face.get(key) is not None:
            pts = np.asarray(face[key], dtype=np.float32)
            updates[key] = new[:2] + (pts - old[:2]) * scale
    updated = main_window.alignments.update_face(
        frame_number, face_index, 1, 1, **updates
    )
    if updated:
        _refresh(main_window)
    return updated


def delete_face(main_window: "MainWindow", frame_number: int, face_index: int) -> None:
    if main_window.alignments.remove_face(frame_number, face_index):
        _refresh(main_window)


def copy_face(main_window: "MainWindow", frame_number: int, face_index: int) -> bool:
    """Copies a stored face (normalised coords) to the alignment clipboard."""
    faces = _frame_faces(main_window, frame_number)
    if not 0 <= face_index < len(faces):
        return False
    setattr(main_window, _CLIPBOARD_ATTR, faces[face_index])
    return True


def paste_face(
    main_window: "MainWindow", frame_number: int, replace_index: Optional[int] = None
) -> bool:
    """Pastes the copied face as manual, replacing ``replace_index`` if given."""
    copied = getattr(main_window, _CLIPBOARD_ATTR, None)
    if copied is None:
        return False
    return _paste(main_window, frame_number, copied, replace_index)


def paste_from_frame(
    main_window: "MainWindow",
    source_frame: int,
    source_index: int,
    frame_number: int,
    replace_index: Optional[int] = None,
) -> bool:
    """Pastes a face stored on ``source_frame`` onto ``frame_number`` (clipboard untouched)."""
    faces = _frame_faces(main_window, source_frame)
    if not 0 <= source_index < len(faces):
        return False
    return _paste(main_window, frame_number, faces[source_index], replace_index)


def _paste(
    main_window: "MainWindow",
    frame_number: int,
    copied: dict,
    replace_index: Optional[int],
) -> bool:
    face = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in copied.items()
    }
    face["source"] = SOURCE_MANUAL
    store = main_window.alignments
    store.bind_media(main_window.video_processor.media_path)
    if replace_index is not None and store.update_face(
        frame_number, replace_index, 1, 1, **face
    ):
        _refresh(main_window)
        return True
    store.add_face(frame_number, face, 1, 1)
    _refresh(main_window)
    return True


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
            self._on_moved(self.scenePos())
        self._press_pos = None


class _CornerHandle(QtWidgets.QGraphicsRectItem):
    """Square on a box corner; dragging it resizes the parent ``_FaceBox``."""

    def __init__(self, box: "_FaceBox", corner: int):
        r = _HANDLE_RADIUS
        super().__init__(-r, -r, 2 * r, 2 * r, box)
        self._box = box
        self.corner = corner
        self._dragging = False
        self.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
        pen = QtGui.QPen(QtGui.QColor(0, 0, 0))
        pen.setCosmetic(True)
        self.setPen(pen)
        flags = QtWidgets.QGraphicsItem.GraphicsItemFlag
        self.setFlag(flags.ItemIsMovable, True)
        self.setFlag(flags.ItemIgnoresTransformations, True)
        self.setZValue(_OVERLAY_Z + 2)
        cursor = (
            QtCore.Qt.CursorShape.SizeFDiagCursor
            if corner in (0, 2)
            else QtCore.Qt.CursorShape.SizeBDiagCursor
        )
        self.setCursor(cursor)

    def mousePressEvent(self, event):
        self._dragging = True
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self._box.drag_corner(self)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self._dragging:
            self._dragging = False
            self._box.finish_resize()


class _FaceBox(QtWidgets.QGraphicsRectItem):
    """Draggable face bbox; landmark and corner handles are children and move with it."""

    def __init__(
        self,
        rect: QtCore.QRectF,
        on_moved: Callable[[QtCore.QPointF], Any],
        on_resized: Optional[Callable[[QtCore.QRectF], Any]] = None,
    ):
        super().__init__(rect)
        self._on_moved = on_moved
        self._on_resized = on_resized
        self._start_rect = QtCore.QRectF(rect)
        self._corners = [_CornerHandle(self, i) for i in range(4)]
        self._place_corners()
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton)
        self.setZValue(_OVERLAY_Z)
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)

    def _corner_points(self, rect: QtCore.QRectF) -> list[QtCore.QPointF]:
        return [rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft()]

    def _place_corners(self, skip: Optional[_CornerHandle] = None) -> None:
        for handle, point in zip(self._corners, self._corner_points(self.rect())):
            if handle is not skip:
                handle.setPos(point)

    def drag_corner(self, handle: _CornerHandle) -> None:
        opposite = self._corner_points(self._start_rect)[(handle.corner + 2) % 4]
        self.setRect(QtCore.QRectF(opposite, handle.pos()).normalized())
        self._place_corners(skip=handle)

    def finish_resize(self) -> None:
        rect = self.rect()
        if (
            rect.width() < _MIN_BBOX_PIXELS
            or rect.height() < _MIN_BBOX_PIXELS
            or self._on_resized is None
        ):
            self.setRect(self._start_rect)
            self._place_corners()
            return
        if rect != self._start_rect:
            self._on_resized(rect.translated(self.pos()))

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if not self.pos().isNull():
            self._on_moved(self.pos())


def _commit_box(
    main_window: "MainWindow",
    frame_number: int,
    face_index: int,
    offset: QtCore.QPointF,
) -> None:
    rect = _pixmap_rect(main_window)
    if rect is None:
        return
    move_face(
        main_window,
        frame_number,
        face_index,
        offset.x() / rect.width(),
        offset.y() / rect.height(),
    )


def _commit_resize(
    main_window: "MainWindow",
    frame_number: int,
    face_index: int,
    scene_rect: QtCore.QRectF,
) -> None:
    rect = _pixmap_rect(main_window)
    if rect is None:
        return
    bbox = bbox_from_corners(
        _to_normalized(rect, scene_rect.left(), scene_rect.top()),
        _to_normalized(rect, scene_rect.right(), scene_rect.bottom()),
    )
    resize_face(main_window, frame_number, face_index, np.clip(bbox, 0.0, 1.0))


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
        box = _FaceBox(
            QtCore.QRectF(
                rect.left() + x1 * rect.width(),
                rect.top() + y1 * rect.height(),
                (x2 - x1) * rect.width(),
                (y2 - y1) * rect.height(),
            ),
            partial(_commit_box, main_window, frame_number, face_index),
            partial(_commit_resize, main_window, frame_number, face_index),
        )
        box.setPen(pen)
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
            handle.setParentItem(box)
    setattr(main_window, _OVERLAY_ATTR, items)


def handle_mouse_press(
    main_window: "MainWindow", view: QtWidgets.QGraphicsView, event: QtGui.QMouseEvent
) -> bool:
    """Starts drawing a bbox; returns True if the event was consumed."""
    if event.button() != QtCore.Qt.MouseButton.LeftButton:
        return False
    if not is_alignment_edit_active(main_window):
        return False
    if isinstance(view.itemAt(event.pos()), (_LandmarkHandle, _CornerHandle, _FaceBox)):
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
) -> dict[QtGui.QAction, Callable[[], Any]]:
    """Adds the Edit Alignments entries; returns handlers keyed by action."""
    handlers: dict[QtGui.QAction, Callable[[], Any]] = {}
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
    point = (
        _to_normalized(rect, scene_pos.x(), scene_pos.y())
        if rect is not None
        else (0.5, 0.5)
    )
    face_index = face_index_at(faces, *point) if faces else None

    copy_action = menu.addAction("Copy Face")
    copy_action.setEnabled(face_index is not None)
    if face_index is not None:
        handlers[copy_action] = partial(
            copy_face, main_window, frame_number, face_index
        )
    for label, other_frame in (
        ("Paste Previous", frame_number - 1),
        ("Paste Next", frame_number + 1),
    ):
        other_index = nearest_face_index(_frame_faces(main_window, other_frame), *point)
        action = menu.addAction(label)
        action.setEnabled(other_index is not None)
        if other_index is not None:
            handlers[action] = partial(
                paste_from_frame,
                main_window,
                other_frame,
                other_index,
                frame_number,
                face_index,
            )
    paste_action = menu.addAction("Paste Face")
    paste_action.setEnabled(getattr(main_window, _CLIPBOARD_ATTR, None) is not None)
    handlers[paste_action] = partial(paste_face, main_window, frame_number, face_index)

    if face_index is not None:
        delete_action = menu.addAction("Delete Face")
        handlers[delete_action] = partial(
            delete_face, main_window, frame_number, face_index
        )
    if frame_number in main_window.alignments:
        redetect_action = menu.addAction("Re-detect Frame")
        handlers[redetect_action] = partial(redetect_frame, main_window, frame_number)
    return handlers


# --- Target media switching ----------------------------------------------
_SUPPRESS_SWITCH_PROMPT_ATTR = "_suppress_media_switch_prompt"


@contextlib.contextmanager
def suppress_media_switch_prompt(main_window: "MainWindow") -> Iterator[None]:
    """Skips the unsaved-alignments prompt for programmatic media selection."""
    previous = getattr(main_window, _SUPPRESS_SWITCH_PROMPT_ATTR, False)
    setattr(main_window, _SUPPRESS_SWITCH_PROMPT_ATTR, True)
    try:
        yield
    finally:
        setattr(main_window, _SUPPRESS_SWITCH_PROMPT_ATTR, previous)


def _ask_discard_alignments(
    main_window: "MainWindow",
    frame_count: int,
    name: str,
    verb: str,
    consequence: str,
) -> str:
    box = QtWidgets.QMessageBox(main_window)
    box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    box.setWindowTitle("Unsaved Alignments")
    box.setText(
        f"Stored alignments for {frame_count} frames of '{name}' haven't been saved."
    )
    box.setInformativeText(consequence)
    save_button = box.addButton(
        f"Save Workspace && {verb}", QtWidgets.QMessageBox.ButtonRole.AcceptRole
    )
    discard_button = box.addButton(
        f"{verb} Without Saving", QtWidgets.QMessageBox.ButtonRole.DestructiveRole
    )
    cancel_button = box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(cancel_button)
    box.exec()
    clicked = box.clickedButton()
    if clicked is save_button:
        return "save"
    if clicked is discard_button:
        return "discard"
    return "cancel"


def _save_workspace(main_window: "MainWindow") -> None:
    from app.ui.widgets.actions import save_load_actions

    save_load_actions.save_current_workspace(main_window)


def _unsaved_alignments(main_window: "MainWindow") -> Optional[AlignmentStore]:
    """The store if it holds unsaved alignments and prompting is allowed."""
    store = getattr(main_window, "alignments", None)
    if not isinstance(store, AlignmentStore):
        return None
    if getattr(main_window, _SUPPRESS_SWITCH_PROMPT_ATTR, False) or getattr(
        main_window, "is_batch_processing", False
    ):
        return None
    if store.media_path is None or not store.has_unsaved_changes():
        return None
    return store


def _confirm_discard(
    main_window: "MainWindow", store: AlignmentStore, verb: str, consequence: str
) -> bool:
    choice = _ask_discard_alignments(
        main_window,
        len(store),
        os.path.basename(store.media_path or ""),
        verb,
        consequence,
    )
    if choice == "save":
        _save_workspace(main_window)
        return not store.has_unsaved_changes()
    return choice == "discard"


def confirm_switch_target_media(
    main_window: "MainWindow", media_path: Optional[str]
) -> bool:
    """Asks before a media switch drops unsaved alignments; False means cancel."""
    store = _unsaved_alignments(main_window)
    if store is None or store.matches_media(media_path):
        return True
    return _confirm_discard(
        main_window,
        store,
        "Switch",
        "Switching target media discards them, including manual edits.",
    )


def confirm_load_workspace(main_window: "MainWindow") -> bool:
    """Asks before loading a workspace drops unsaved alignments; False means cancel."""
    store = _unsaved_alignments(main_window)
    if store is None:
        return True
    return _confirm_discard(
        main_window,
        store,
        "Load",
        "Loading a workspace discards them, including manual edits.",
    )
