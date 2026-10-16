from types import SimpleNamespace

from PySide6 import QtCore, QtGui, QtWidgets

import app.ui.main_ui  # noqa: F401  (import order: avoids circular imports)
from app.ui.widgets.actions import alignment_edit_actions as aea
from app.ui.widgets.event_filters import GraphicsViewEventFilter


def _key(key):
    return QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress, key, QtCore.Qt.KeyboardModifier.NoModifier
    )


def test_arrow_keys_never_scroll_the_preview(monkeypatch):
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    steps = []
    monkeypatch.setattr(
        aea,
        "step_frame_shortcut",
        lambda mw, forward: steps.append(forward) or False,
    )
    event_filter = GraphicsViewEventFilter(SimpleNamespace())
    view = QtWidgets.QGraphicsView()
    assert event_filter.eventFilter(view, _key(QtCore.Qt.Key.Key_Left))
    assert event_filter.eventFilter(view, _key(QtCore.Qt.Key.Key_Right))
    assert not event_filter.eventFilter(view, _key(QtCore.Qt.Key.Key_Up))
    assert steps == [False, True]
