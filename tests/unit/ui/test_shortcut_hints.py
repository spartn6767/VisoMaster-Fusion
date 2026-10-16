from types import SimpleNamespace

from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.widgets.actions import shortcut_hints


def test_tooltip_shows_the_shortcut_key(monkeypatch):
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    shown = []
    monkeypatch.setattr(
        QtWidgets.QToolTip,
        "showText",
        lambda pos, text, widget=None: shown.append(text),
    )
    play = QtWidgets.QPushButton()
    play.setToolTip("Play")
    window = SimpleNamespace(buttonMediaPlay=play)
    shortcut_hints.install_shortcut_hints(window)

    event = QtGui.QHelpEvent(
        QtCore.QEvent.Type.ToolTip, QtCore.QPoint(1, 1), QtCore.QPoint(1, 1)
    )
    assert window.shortcut_hint_filter.eventFilter(play, event)
    play.setToolTip("Stop")
    window.shortcut_hint_filter.eventFilter(play, event)
    assert shown == ["Play (Space)", "Stop (Space)"]
    assert not window.shortcut_hint_filter.eventFilter(QtWidgets.QPushButton(), event)
