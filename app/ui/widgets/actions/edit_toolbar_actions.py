"""Toolbar above the media buttons for the Edit Alignments / Paint Mask modes.

Mirrors the viewer's right-click menu for the frame-wide actions; per-face actions
stay in the menu because they need a face under the cursor.
"""

from typing import TYPE_CHECKING

from PySide6 import QtWidgets

from app.ui.widgets.actions import alignment_edit_actions as aea
from app.ui.widgets.actions import mask_paint_actions as mpa

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

_BUTTONS_ATTR = "editToolbarButtons"


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
