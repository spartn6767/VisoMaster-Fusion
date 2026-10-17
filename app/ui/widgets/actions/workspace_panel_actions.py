"""Workspace section above Target Videos: the current workspace, Open / Save /
Save As, recently used workspaces, and the Ctrl+S / Ctrl+Shift+S shortcuts."""

from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from PySide6 import QtCore, QtGui, QtWidgets

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

RECENT_LIMIT = 10
_MEDIA = Path(__file__).resolve().parents[2] / "core" / "media"
_PANEL_ATTR = "workspacePanel"
_LABEL_ATTR = "_workspace_panel_label"


def recent_file(main_window: Any) -> Path:
    return Path(main_window.last_workspace_path).with_name("recent_workspaces.json")


def recent_workspaces(main_window: Any) -> list[str]:
    try:
        data = json.loads(recent_file(main_window).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [str(p) for p in data if isinstance(p, str)][:RECENT_LIMIT]


def _write_recent(main_window: Any, paths: list[str]) -> None:
    try:
        recent_file(main_window).write_text(
            json.dumps(paths[:RECENT_LIMIT], indent=2), encoding="utf-8"
        )
    except OSError as e:
        print(f"[WARN] Could not save the recent workspaces list: {e}")


def remember_workspace(main_window: Any, path: str) -> None:
    """Moves ``path`` to the top of the recent workspaces."""
    if not path or not getattr(main_window, "last_workspace_path", None):
        return
    if Path(path).name == "last_workspace.json":
        return
    full = os.path.normpath(os.path.abspath(path))
    paths = [
        p
        for p in recent_workspaces(main_window)
        if os.path.normcase(p) != os.path.normcase(full)
    ]
    _write_recent(main_window, [full, *paths])


def forget_missing(main_window: Any) -> None:
    _write_recent(
        main_window, [p for p in recent_workspaces(main_window) if os.path.isfile(p)]
    )


def clear_recent(main_window: Any) -> None:
    _write_recent(main_window, [])


def workspace_title(main_window: Any) -> str:
    from app.ui.widgets.actions import unsaved_changes_actions

    path = str(getattr(main_window, "current_workspace_path", "") or "")
    name = os.path.basename(path) if path else "Unsaved workspace"
    if unsaved_changes_actions.has_unsaved_changes(main_window):
        return f"{name} \u2022"
    return name


def refresh_workspace_panel(main_window: Any) -> None:
    label = getattr(main_window, _LABEL_ATTR, None)
    if label is None:
        return
    title = workspace_title(main_window)
    if label.text() != title:
        label.setText(title)
    path = str(getattr(main_window, "current_workspace_path", "") or "")
    label.setToolTip(path or "Not saved to a file yet")


def open_workspace(main_window: "MainWindow", path: Union[str, bool] = False) -> None:
    from app.ui.widgets.actions import save_load_actions

    save_load_actions.load_saved_workspace(main_window, path)
    refresh_workspace_panel(main_window)


def save_workspace(main_window: "MainWindow", save_as: bool = False) -> None:
    """Saves in place (asking where if there's no file yet) with the alignments."""
    from app.ui.widgets.actions import save_load_actions

    save_load_actions.save_current_workspace(main_window, save_as=save_as)
    refresh_workspace_panel(main_window)


def _populate_recent_menu(main_window: Any, menu: QtWidgets.QMenu) -> None:
    menu.clear()
    paths = recent_workspaces(main_window)
    if not paths:
        menu.addAction("No recent workspaces").setEnabled(False)
        return
    for path in paths:
        missing = not os.path.isfile(path)
        action = menu.addAction(
            f"{os.path.basename(path)}{'  (missing)' if missing else ''}"
        )
        action.setToolTip(path)
        action.setEnabled(not missing)
        action.triggered.connect(lambda *_, p=path: open_workspace(main_window, p))
    menu.addSeparator()
    menu.addAction("Remove Missing").triggered.connect(
        lambda *_: forget_missing(main_window)
    )
    menu.addAction("Clear List").triggered.connect(lambda *_: clear_recent(main_window))


def install_workspace_panel(main_window: Any) -> QtWidgets.QWidget:
    existing = getattr(main_window, _PANEL_ATTR, None)
    if existing is not None:
        return existing
    panel = QtWidgets.QFrame(main_window.dockWidgetContents)
    panel.setObjectName("workspacePanel")
    outer = QtWidgets.QVBoxLayout(panel)
    outer.setContentsMargins(0, 0, 0, 4)
    outer.setSpacing(2)
    header = QtWidgets.QHBoxLayout()
    toggle = QtWidgets.QToolButton(panel)
    toggle.setText("Workspace")
    toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    toggle.setArrowType(QtCore.Qt.ArrowType.DownArrow)
    toggle.setCheckable(True)
    toggle.setChecked(True)
    toggle.setAutoRaise(True)
    label = QtWidgets.QLabel(panel)
    header.addWidget(toggle)
    header.addWidget(label, 1)
    outer.addLayout(header)
    body = QtWidgets.QWidget(panel)
    row = QtWidgets.QHBoxLayout(body)
    row.setContentsMargins(0, 0, 0, 0)
    for key, icon, tip, slot in (
        (
            "open",
            "open_file.png",
            "Open Workspace...",
            partial(open_workspace, main_window),
        ),
        (
            "save",
            "save_file.png",
            "Save Workspace, with its alignments and masks (Ctrl+S)",
            partial(save_workspace, main_window, False),
        ),
        (
            "save_as",
            "save_file_as.png",
            "Save Workspace As... (Ctrl+Shift+S)",
            partial(save_workspace, main_window, True),
        ),
    ):
        button = QtWidgets.QToolButton(body)
        button.setObjectName(f"workspacePanel_{key}")
        button.setIcon(QtGui.QIcon(str(_MEDIA / icon)))
        button.setIconSize(QtCore.QSize(18, 18))
        button.setAutoRaise(True)
        button.setToolTip(tip)
        button.clicked.connect(lambda *_, s=slot: s())
        row.addWidget(button)
    recent = QtWidgets.QToolButton(body)
    recent.setObjectName("workspacePanel_recent")
    recent.setIcon(
        body.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_FileDialogDetailedView
        )
    )
    recent.setIconSize(QtCore.QSize(18, 18))
    recent.setAutoRaise(True)
    recent.setToolTip("Recent Workspaces")
    recent.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
    menu = QtWidgets.QMenu(recent)
    menu.aboutToShow.connect(lambda: _populate_recent_menu(main_window, menu))
    recent.setMenu(menu)
    row.addWidget(recent)
    row.addStretch(1)
    outer.addWidget(body)

    def toggled(on: bool) -> None:
        body.setVisible(on)
        toggle.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if on else QtCore.Qt.ArrowType.RightArrow
        )

    toggle.toggled.connect(toggled)
    dock = getattr(main_window, "input_Target_DockWidget", None)
    if isinstance(dock, QtWidgets.QDockWidget):
        # Above the dock's "Target Videos" title, so Target Videos sits below it.
        title_bar = QtWidgets.QWidget(dock)
        column = QtWidgets.QVBoxLayout(title_bar)
        column.setContentsMargins(6, 4, 6, 2)
        column.setSpacing(2)
        panel.setParent(title_bar)
        column.addWidget(panel)
        column.addWidget(QtWidgets.QLabel(dock.windowTitle(), title_bar))
        dock.setTitleBarWidget(title_bar)
    else:
        main_window.vboxLayout.insertWidget(0, panel)
    for keys, save_as in (("Ctrl+S", False), ("Ctrl+Shift+S", True)):
        shortcut = QtGui.QShortcut(QtGui.QKeySequence(keys), main_window)
        shortcut.activated.connect(partial(save_workspace, main_window, save_as))
    timer = QtCore.QTimer(panel)
    timer.setInterval(2000)
    timer.timeout.connect(lambda: refresh_workspace_panel(main_window))
    timer.start()
    setattr(main_window, _PANEL_ATTR, panel)
    setattr(main_window, _LABEL_ATTR, label)
    refresh_workspace_panel(main_window)
    return panel
