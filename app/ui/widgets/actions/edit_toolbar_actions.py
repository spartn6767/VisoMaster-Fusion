"""Edit toolbar above the media buttons, the floating Edit Alignments bar and the
right-click menu on a face box in Edit Alignments / Paint Mask.

The bars carry frame-wide actions; actions on one face live in the face menu.
"""

import os
from typing import TYPE_CHECKING, Any, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.widgets.actions import alignment_edit_actions as aea
from app.ui.widgets.actions import mask_paint_actions as mpa

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

_BUTTONS_ATTR = "editToolbarButtons"


def _saved_workspace(main_window: "MainWindow") -> str:
    path = str(getattr(main_window, "current_workspace_path", "") or "")
    return path if path and os.path.isfile(path) else ""


def _save_alignments(main_window: "MainWindow") -> bool:
    """Saves next to the current workspace without asking; with no saved workspace,
    asks where to save it first. Confirms with a toast."""
    from app.ui.widgets.actions import common_actions, save_load_actions

    workspace = _saved_workspace(main_window)
    if workspace:
        save_load_actions.save_current_workspace(main_window, workspace)
    else:
        save_load_actions.save_current_workspace(main_window)
        workspace = _saved_workspace(main_window)
        if not workspace:
            return False  # cancelled
    common_actions.create_and_show_toast_message(
        main_window,
        "Alignments Saved",
        f"Saved alignments for {len(main_window.alignments)} frames next to "
        f"{os.path.basename(workspace)}.",
    )
    return True


def install_edit_toolbar(main_window: "MainWindow") -> QtWidgets.QWidget:
    """Adds the toolbar row directly above the media buttons row."""
    existing = getattr(main_window, "editToolbar", None)
    if existing is not None:
        return existing
    bar = QtWidgets.QWidget(main_window.mediaLayout)
    bar.setObjectName("editToolbar")
    layout = QtWidgets.QHBoxLayout(bar)
    layout.setContentsMargins(0, 2, 0, 2)
    layout.setSpacing(6)
    layout.addStretch(1)

    def add_button(key: str, text: str, tip: str, checkable: bool = False):
        button = QtWidgets.QPushButton(text, bar)
        button.setObjectName(f"editToolbar_{key}")
        button.setCheckable(checkable)
        button.setToolTip(tip)
        layout.addWidget(button)
        buttons[key] = button
        return button

    buttons: dict[str, QtWidgets.QPushButton] = {}
    add_button(
        "edit",
        "Edit Alignments",
        "Drag landmarks and boxes, draw missed faces (pauses playback).",
        checkable=True,
    ).toggled.connect(lambda on: aea.set_alignment_edit_mode(main_window, on))
    add_button(
        "paint",
        "Paint Mask",
        "Paint add/erase strokes on faces (pauses playback).",
        checkable=True,
    ).toggled.connect(lambda on: mpa.set_mask_paint_mode(main_window, on))
    add_button(
        "save",
        "Save Alignments",
        "Save the alignments (with masks) with the current workspace; asks where "
        "if the workspace hasn't been saved yet.",
    ).clicked.connect(lambda: _save_alignments(main_window))
    separator = QtWidgets.QFrame(bar)
    separator.setFrameShape(QtWidgets.QFrame.Shape.VLine)
    layout.addWidget(separator)
    add_button(
        "redetect", "Re-detect Frame", "Drop this frame's stored faces."
    ).clicked.connect(
        lambda: aea.redetect_frame(main_window, aea.current_frame_number(main_window))
    )
    add_button(
        "reset_masks", "Reset All Masks", "Remove every painted mask in this video."
    ).clicked.connect(lambda: mpa.reset_all_masks(main_window))
    add_button(
        "reset_alignments",
        "Reset All Alignments",
        "Discard every stored alignment, manual edits and masks included.",
    ).clicked.connect(lambda: aea.reset_all_alignments(main_window))
    layout.addStretch(1)

    parent_layout = main_window.verticalLayoutMediaControls
    index = parent_layout.indexOf(main_window.horizontalLayoutMediaButtons)
    parent_layout.insertWidget(index if index >= 0 else parent_layout.count(), bar)
    main_window.editToolbar = bar
    setattr(main_window, _BUTTONS_ATTR, buttons)
    sync_edit_toolbar(main_window)
    return bar


def sync_edit_toolbar(main_window: "MainWindow") -> None:
    """Matches the toggles to the current modes and greys out actions that don't apply."""
    sync_alignment_bar(main_window)
    face_compare = getattr(main_window, "actionView_FaceCompare", None)
    if isinstance(face_compare, QtGui.QAction):
        # X is "Paste >" while an edit mode is on.
        editing = getattr(main_window, "alignment_edit_enabled", False) or getattr(
            main_window, "mask_paint_enabled", False
        )
        face_compare.setShortcut(QtGui.QKeySequence() if editing else "X")
    buttons = getattr(main_window, _BUTTONS_ATTR, None)
    if not buttons:
        return
    for key, flag in (
        ("edit", "alignment_edit_enabled"),
        ("paint", "mask_paint_enabled"),
    ):
        button = buttons[key]
        button.blockSignals(True)
        button.setChecked(bool(getattr(main_window, flag, False)))
        button.blockSignals(False)
    store = getattr(main_window, "alignments", None)
    if store is None:
        for key in ("redetect", "reset_masks", "reset_alignments"):
            buttons[key].setEnabled(False)
        return
    buttons["redetect"].setEnabled(aea.current_frame_number(main_window) in store)
    buttons["reset_masks"].setEnabled(store.mask_frame_count() > 0)
    buttons["reset_alignments"].setEnabled(len(store) > 0)


# --- Floating Edit Alignments bar ---------------------------------------------
_ALIGN_BAR_ATTR = "_alignment_bar"
_ALIGN_BAR_BUTTONS_ATTR = "_alignment_bar_buttons"
_ALIGN_BAR_POS_ATTR = "alignment_bar_pos"
_ALIGN_PASTE_LABEL_ATTR = "_alignment_paste_label"
ALIGNMENT_FRAME_ACTIONS = (
    ("undo", "Undo", "Undo the last alignment or mask edit (Ctrl+Z)."),
    ("redo", "Redo", "Redo the last undone edit (Ctrl+Shift+Z)."),
    ("redetect", "Re-detect Frame", "Drop this frame's stored faces."),
    ("reset_all", "Reset All Alignments", "Discard every stored alignment."),
    (
        "check",
        "Find Frames to Check",
        "Mark frames where landmarks jump or a face briefly disappears "
        "(Shift+N / Shift+P to step through them).",
    ),
)
ALIGNMENT_FACE_ACTIONS = (
    ("copy", "Copy", "Copy the selected face (C)."),
    ("paste", "Paste", "Paste the copied face, replacing the selected face (V)."),
    (
        "previous",
        "< Paste",
        "Paste the face from the nearest earlier stored frame (Z). "
        "Press again for the next earlier one.",
    ),
    (
        "next",
        "Paste >",
        "Paste the face from the nearest later stored frame (X). "
        "Press again for the next later one.",
    ),
    ("assign", "Assign to Target", "Pin the selected face to a target face."),
    ("add_target", "Add as Target", "Create a target face from the selected face."),
    ("delete", "Delete Face", "Delete the selected face (M)."),
    (
        "fill",
        "Fill Between Edits",
        "Blend the selected face between the previous frame you edited it on "
        "and this frame.",
    ),
)


def active_alignment_face(main_window: "MainWindow") -> Optional[int]:
    """Last dragged/resized/right-clicked face on this frame, or the only face."""
    frame_number = aea.current_frame_number(main_window)
    faces = aea._frame_faces(main_window, frame_number)
    chosen = getattr(main_window, aea.ACTIVE_FACE_ATTR, None)
    if chosen is not None and chosen[0] == frame_number and chosen[1] < len(faces):
        return chosen[1]
    return 0 if len(faces) == 1 else None


def alignment_bar_action(main_window: "MainWindow", key: str) -> Any:
    frame_number = aea.current_frame_number(main_window)
    face_index = active_alignment_face(main_window)
    if face_index is None and key in ("previous", "next"):
        face_index = mpa.open_run_face(main_window, "align", frame_number)
    result: Any = False
    if key in ("undo", "redo"):
        result = aea.undo_edit(main_window, redo=key == "redo")
    elif key == "redetect":
        aea.redetect_frame(main_window, frame_number)
        result = True
    elif key == "reset_all":
        result = aea.reset_all_alignments(main_window)
    elif key == "check":
        result = aea.find_frames_to_check(main_window)
    elif key == "paste":
        result = aea.paste_face(main_window, frame_number, face_index)
    elif face_index is not None:
        if key == "copy":
            result = aea.copy_face(main_window, frame_number, face_index)
        elif key in ("previous", "next"):
            result = mpa.paste_step(
                main_window,
                "align",
                frame_number,
                face_index,
                -1 if key == "previous" else 1,
            )
        elif key == "add_target":
            result = aea.add_as_target_face(main_window, frame_number, face_index)
        elif key == "delete":
            aea.delete_face(main_window, frame_number, face_index)
            result = True
        elif key == "fill":
            result = aea.fill_between_edits(main_window, frame_number, face_index)
    sync_alignment_bar(main_window)
    return result


def populate_assign_menu(main_window: "MainWindow", menu: QtWidgets.QMenu) -> None:
    menu.clear()
    frame_number = aea.current_frame_number(main_window)
    face_index = active_alignment_face(main_window)
    if face_index is None:
        menu.addAction("Select a face first").setEnabled(False)
        return
    faces = aea._frame_faces(main_window, frame_number)
    assigned = faces[face_index].get("assigned_face_id")
    targets = getattr(main_window, "target_faces", None) or {}
    for n, (target_id, target) in enumerate(targets.items(), start=1):
        target_face_id = str(getattr(target, "face_id", target_id))
        action = menu.addAction(aea._target_label(target, n))
        action.setCheckable(True)
        action.setChecked(target_face_id == assigned)
        action.triggered.connect(
            lambda *_, f=target_face_id: aea.assign_target(
                main_window, frame_number, face_index, f
            )
        )
    if not targets:
        menu.addAction("No target faces").setEnabled(False)
    if assigned:
        menu.addSeparator()
        menu.addAction("Clear Assignment").triggered.connect(
            lambda *_: aea.assign_target(main_window, frame_number, face_index, None)
        )


def _build_alignment_bar(main_window: "MainWindow") -> QtWidgets.QFrame:
    bar = QtWidgets.QFrame(main_window.graphicsViewFrame)
    bar.setObjectName("alignmentEditBar")
    bar.setStyleSheet(
        "#alignmentEditBar { background: rgba(20, 20, 20, 200); border-radius: 4px; }"
    )
    top = QtWidgets.QHBoxLayout(bar)
    top.setContentsMargins(6, 4, 6, 4)
    top.setSpacing(4)

    def moved(point: QtCore.QPoint) -> None:
        bar.move(point)
        setattr(main_window, _ALIGN_BAR_POS_ATTR, (point.x(), point.y()))

    top.addWidget(mpa._DragHandle(bar, moved))
    top.addWidget(QtWidgets.QLabel("Alignments", bar))
    buttons: dict[str, QtWidgets.QToolButton] = {}
    for key, text, tip in ALIGNMENT_FRAME_ACTIONS:
        button = QtWidgets.QToolButton(bar)
        button.setText(text)
        button.setToolTip(tip)
        button.clicked.connect(lambda *_, k=key: alignment_bar_action(main_window, k))
        top.addWidget(button)
        buttons[key] = button
    label = mpa.paste_source_label(bar)
    top.addWidget(label)
    setattr(main_window, _ALIGN_PASTE_LABEL_ATTR, label)
    top.addStretch(1)
    top.addWidget(
        mpa.close_bar_button(
            bar,
            "Edit Alignments",
            lambda: aea.set_alignment_edit_mode(main_window, False),
        )
    )
    bar.adjustSize()
    x, y = getattr(main_window, _ALIGN_BAR_POS_ATTR, None) or mpa.default_bar_position(
        bar
    )
    bar.move(mpa.clamp_bar_position(bar, QtCore.QPoint(int(x), int(y))))
    setattr(main_window, _ALIGN_BAR_ATTR, bar)
    setattr(main_window, _ALIGN_BAR_BUTTONS_ATTR, buttons)
    return bar


def sync_alignment_bar(main_window: "MainWindow") -> None:
    """Shows the floating bar while Edit Alignments is active and updates its buttons."""
    active = aea.is_alignment_edit_active(main_window)
    bar = getattr(main_window, _ALIGN_BAR_ATTR, None)
    if active and bar is None and getattr(main_window, "graphicsViewFrame", None):
        bar = _build_alignment_bar(main_window)
    if bar is None:
        return
    bar.setVisible(active)
    if not active:
        return
    bar.move(mpa.clamp_bar_position(bar, bar.pos()))
    bar.raise_()
    enabled = alignment_action_states(main_window)
    for key, button in getattr(main_window, _ALIGN_BAR_BUTTONS_ATTR, {}).items():
        button.setEnabled(bool(enabled.get(key, False)))
    label = getattr(main_window, _ALIGN_PASTE_LABEL_ATTR, None)
    if label is not None:
        label.setText(
            mpa.paste_source_text(
                main_window, "align", active_alignment_face(main_window)
            )
        )


def alignment_action_states(main_window: "MainWindow") -> dict[str, bool]:
    """Which bar and face-menu actions apply to the selected face."""
    store = main_window.alignments
    frame_number = aea.current_frame_number(main_window)
    face_index = active_alignment_face(main_window)
    enabled = {
        "redetect": frame_number in store,
        "reset_all": len(store) > 0,
        "check": len(store) > 0,
        "undo": store.can_undo(),
        "redo": store.can_redo(),
        "paste": getattr(main_window, aea._CLIPBOARD_ATTR, None) is not None,
    }
    if face_index is not None:
        enabled.update(
            copy=True,
            previous=mpa._adjacent_source(main_window, frame_number, face_index, -1)
            is not None,
            next=mpa._adjacent_source(main_window, frame_number, face_index, 1)
            is not None,
            assign=True,
            add_target=True,
            delete=True,
            fill=bool(store.edited_frames())
            and store.edited_frames()[0] < frame_number - 1,
        )
    return enabled


# --- Right-click menu on a face box ---------------------------------------------
FACE_MENU_KEYS = {
    "copy": "C",
    "paste": "V",
    "previous": "Z",
    "next": "X",
    "delete": "M",
    "reset_frame": "M",
}


def face_menu_at(
    main_window: "MainWindow",
    parent: Optional[QtWidgets.QWidget],
    x: float,
    y: float,
) -> Optional[QtWidgets.QMenu]:
    """Menu of the actions for the face whose box contains (x, y) (normalised),
    which it selects; None outside the edit modes or outside every box."""
    actions: tuple[tuple[str, str, str], ...]
    if aea.is_alignment_edit_active(main_window):
        actions, run = ALIGNMENT_FACE_ACTIONS, alignment_bar_action
        select, states = aea.select_face, alignment_action_states
    elif mpa.is_mask_paint_active(main_window):
        actions, run = mpa.FACE_ACTIONS, mpa.toolbar_action
        select, states = mpa.set_active_face, mpa.action_states
    else:
        return None
    frame_number = aea.current_frame_number(main_window)
    face_index = aea.face_index_at(aea._frame_faces(main_window, frame_number), x, y)
    if face_index is None:
        return None
    select(main_window, frame_number, face_index)
    sync_alignment_bar(main_window)
    mpa._sync_bar_buttons(main_window)
    enabled = states(main_window)
    menu = QtWidgets.QMenu(parent)
    menu.setToolTipsVisible(True)
    for key, text, tip in actions:
        if key == "assign":
            action = menu.addMenu(text).menuAction()
            populate_assign_menu(main_window, action.menu())
        else:
            shortcut = FACE_MENU_KEYS.get(key)
            action = menu.addAction(f"{text}\t{shortcut}" if shortcut else text)
            action.triggered.connect(lambda *_, k=key: run(main_window, k))
        action.setToolTip(tip)
        action.setEnabled(bool(enabled.get(key, False)))
    paint = run is mpa.toolbar_action
    menu.addSeparator()
    if paint:
        erase = bool(mpa.brush_settings(main_window)["erase"])
        switch = menu.addAction("Switch to Paint\t3" if erase else "Switch to Erase\t3")
        switch.triggered.connect(lambda *_: mpa.set_erase(main_window, not erase))
    mode = menu.addAction(
        "Switch to Edit Alignments" if paint else "Switch to Paint Mask"
    )
    mode.triggered.connect(
        lambda *_: _switch_edit_mode(main_window, not paint, frame_number, face_index)
    )
    return menu


def _switch_edit_mode(
    main_window: "MainWindow", paint: bool, frame_number: int, face_index: int
) -> None:
    """Turns on the other edit mode with the same face selected."""
    if paint:
        mpa.set_mask_paint_mode(main_window, True)
        mpa.set_active_face(main_window, frame_number, face_index)
        mpa._sync_bar_buttons(main_window)
    else:
        aea.set_alignment_edit_mode(main_window, True)
        aea.select_face(main_window, frame_number, face_index)
        sync_alignment_bar(main_window)


def face_context_menu(
    main_window: "MainWindow", scene_pos: QtCore.QPointF
) -> Optional[QtWidgets.QMenu]:
    """face_menu_at for a viewer scene position."""
    rect = aea._pixmap_rect(main_window)
    if rect is None:
        return None
    return face_menu_at(
        main_window,
        main_window.graphicsViewFrame,
        *aea._to_normalized(rect, scene_pos.x(), scene_pos.y()),
    )
