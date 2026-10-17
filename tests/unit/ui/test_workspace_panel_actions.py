import os

from PySide6 import QtGui, QtWidgets

from app.ui.widgets.actions import workspace_panel_actions as wpa


def _window(tmp_path):
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    mw = QtWidgets.QWidget()
    mw.dockWidgetContents = QtWidgets.QWidget(mw)
    mw.vboxLayout = QtWidgets.QVBoxLayout(mw.dockWidgetContents)
    mw.last_workspace_path = tmp_path / "last_workspace.json"
    mw.current_workspace_path = ""
    mw.alignments = None
    return mw


def test_recent_workspaces_order_limit_and_missing(tmp_path):
    mw = _window(tmp_path)
    files = []
    for i in range(12):
        f = tmp_path / f"w{i}.json"
        f.write_text("{}")
        files.append(str(f))
        wpa.remember_workspace(mw, str(f))
    wpa.remember_workspace(mw, str(mw.last_workspace_path))
    wpa.remember_workspace(mw, files[5])
    recent = wpa.recent_workspaces(mw)
    assert len(recent) == wpa.RECENT_LIMIT
    assert recent[0] == os.path.normpath(files[5]) and recent[1].endswith("w11.json")
    os.remove(files[11])
    wpa.forget_missing(mw)
    assert not any(p.endswith("w11.json") for p in wpa.recent_workspaces(mw))
    wpa.clear_recent(mw)
    assert wpa.recent_workspaces(mw) == []


def test_panel_buttons_and_ctrl_s(tmp_path, monkeypatch):
    mw = _window(tmp_path)
    saves = []
    monkeypatch.setattr(
        "app.ui.widgets.actions.save_load_actions.save_current_workspace",
        lambda main_window, save_as=False: saves.append(save_as),
    )
    panel = wpa.install_workspace_panel(mw)
    assert wpa.install_workspace_panel(mw) is panel
    assert mw.vboxLayout.itemAt(0).widget() is panel
    assert mw._workspace_panel_label.text() == "Unsaved workspace"
    mw.current_workspace_path = str(tmp_path / "proj.json")
    wpa.refresh_workspace_panel(mw)
    assert mw._workspace_panel_label.text() == "proj.json"

    buttons = {b.objectName(): b for b in panel.findChildren(QtWidgets.QToolButton)}
    assert "Ctrl+S" in buttons["workspacePanel_save"].toolTip()
    buttons["workspacePanel_save"].click()
    buttons["workspacePanel_save_as"].click()
    shortcuts = {s.key().toString(): s for s in mw.findChildren(QtGui.QShortcut)}
    shortcuts["Ctrl+S"].activated.emit()
    shortcuts["Ctrl+Shift+S"].activated.emit()
    assert saves == [False, True, False, True]


def test_new_button_and_menu(tmp_path, monkeypatch):
    from app.ui.widgets.actions import save_load_actions

    mw = _window(tmp_path)
    mw.project_root_path = tmp_path
    mw.actionRecover_Alignments = QtGui.QAction("Recover Autosaved Alignments...")
    calls = []
    monkeypatch.setattr(
        save_load_actions, "new_workspace", lambda *a, **k: calls.append("new")
    )
    monkeypatch.setattr(
        save_load_actions,
        "load_saved_workspace",
        lambda *a, **k: calls.append(("load", a[1:], k)),
    )
    monkeypatch.setattr(
        save_load_actions,
        "save_current_workspace",
        lambda *a, **k: calls.append(("save", a[1:], k)),
    )
    panel = wpa.install_workspace_panel(mw)
    buttons = {b.objectName(): b for b in panel.findChildren(QtWidgets.QToolButton)}
    assert mw.workspacePanelNewButton is buttons["workspacePanel_new"]
    assert mw.workspacePanelOpenButton is buttons["workspacePanel_open"]
    buttons["workspacePanel_new"].click()
    menu = buttons["workspacePanel_menu"].menu()
    wpa._populate_menu(mw, menu)
    actions = {a.text(): a for a in menu.actions()}
    assert "No recent workspaces" in actions
    assert actions["Recover Autosaved Alignments..."].isEnabled() is False
    actions["Reset to Last Session"].trigger()
    actions["Save as Last Session"].trigger()
    last = str(mw.last_workspace_path)
    assert calls[0] == "new"
    assert calls[1][0] == "load" and last in repr(calls[1])
    assert calls[2][0] == "save" and last in repr(calls[2])
