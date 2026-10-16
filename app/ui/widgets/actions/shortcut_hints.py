"""Adds each control's keyboard shortcut to its hover tooltip.

The key is appended when the tooltip is shown, so tooltips that change at runtime
(Play / Stop, Start / Stop Recording) keep their own text. Keep this list in step
with MainWindow.keyPressEvent and docs/wiki/keyboard-shortcuts.md.
"""

from typing import TYPE_CHECKING

from PySide6 import QtCore, QtWidgets

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

HINT_PROPERTY = "shortcutHint"
SHORTCUT_HINTS = {
    "buttonMediaPlay": "Space",
    "buttonMediaRecord": "R",
    "swapfacesButton": "S",
    "theatreModeButton": "T",
    "frameAdvanceButton": "V, or L in the edit modes",
    "frameRewindButton": "C, or K in the edit modes",
    "frameSkipStepSpinBox": "D / A step by this many frames; K / L in the edit modes",
    "addMarkerButton": "F",
    "removeMarkerButton": "Alt+F",
    "nextMarkerButton": "W",
    "previousMarkerButton": "Q",
}


class ShortcutHintFilter(QtCore.QObject):
    def eventFilter(self, obj, event):
        if event.type() != QtCore.QEvent.Type.ToolTip or not isinstance(
            obj, QtWidgets.QWidget
        ):
            return False
        hint = obj.property(HINT_PROPERTY)
        if not hint:
            return False
        text = obj.toolTip()
        QtWidgets.QToolTip.showText(
            event.globalPos(), f"{text} ({hint})" if text else str(hint), obj
        )
        return True


def install_shortcut_hints(main_window: "MainWindow") -> None:
    hint_filter = ShortcutHintFilter()  # kept alive by main_window below
    for name, hint in SHORTCUT_HINTS.items():
        widget = getattr(main_window, name, None)
        if isinstance(widget, QtWidgets.QWidget):
            widget.setProperty(HINT_PROPERTY, hint)
            widget.installEventFilter(hint_filter)
    main_window.shortcut_hint_filter = hint_filter
