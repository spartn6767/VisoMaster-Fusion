"""Asks whether a recording gets the watermark (see app.helpers.watermark)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, cast

from PySide6 import QtWidgets

from app.helpers import watermark

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

ASK_KEY = "RecordingWatermarkAskToggle"
ALWAYS_KEY = "RecordingWatermarkAlwaysToggle"


def _prompt(main_window: Any, text: str) -> Optional[bool]:
    box = QtWidgets.QMessageBox(cast(QtWidgets.QWidget, main_window))
    box.setWindowTitle("Watermark")
    box.setIcon(QtWidgets.QMessageBox.Icon.Question)
    box.setText("Add the watermark to this recording?")
    box.setInformativeText(
        f'"{text}"\n\nChange the text, or stop asking, in '
        "Settings > Video Recording Settings."
    )
    add = box.addButton("Add Watermark", QtWidgets.QMessageBox.ButtonRole.YesRole)
    skip = box.addButton("No Watermark", QtWidgets.QMessageBox.ButtonRole.NoRole)
    box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(add)
    # QMessageBox sizes itself to its text; a minimum keeps the buttons readable.
    box.setStyleSheet(
        "QLabel#qt_msgbox_label, QLabel#qt_msgbox_informativelabel "
        "{ min-width: 440px; } "
        "QPushButton { min-width: 130px; padding: 6px 14px; }"
    )
    box.exec()
    clicked = box.clickedButton()
    return True if clicked is add else False if clicked is skip else None


def ask_recording_watermark(main_window: "MainWindow") -> Optional[bool]:
    """True/False for whether to watermark, None if the user cancelled."""
    control = main_window.control
    text = watermark.watermark_text(control)
    if not text:
        return False
    if control.get(ALWAYS_KEY, False):
        return True
    if not control.get(ASK_KEY, True):
        return False
    return _prompt(main_window, text)


def begin_recording_run(main_window: "MainWindow") -> bool:
    """Asks once for a recording, batch or job run; False if cancelled."""
    answer = ask_recording_watermark(main_window)
    if answer is None:
        return False
    main_window.recording_watermark = answer  # type: ignore[attr-defined]
    return True
