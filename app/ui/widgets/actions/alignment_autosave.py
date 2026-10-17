"""Periodic autosave of unsaved alignments, recoverable from the Workspace section's menu after a crash.

The autosave never marks the store as saved, so the save/discard prompts still
appear. A clean exit (which saves the last workspace) deletes it.
"""

import json
import os
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

CHECK_INTERVAL_MS = 60_000
_TIMER_ATTR = "_alignments_autosave_timer"
_REVISION_ATTR = "_alignments_autosave_revision"
_LAST_ATTR = "_alignments_autosave_time"
ACTION_ATTR = "actionRecover_Alignments"


def autosave_path(main_window: "MainWindow") -> Path:
    return Path(main_window.project_root_path) / ".autosave" / "alignments.npz"


def install_autosave(main_window: "MainWindow") -> None:
    """Starts the autosave timer and creates the Recover Autosaved Alignments action.

    The Workspace section's menu shows the action (workspace_panel_actions).
    """
    if getattr(main_window, _TIMER_ATTR, None) is not None:
        return
    timer = QtCore.QTimer()
    timer.setInterval(CHECK_INTERVAL_MS)
    setattr(main_window, _LAST_ATTR, time.monotonic())
    timer.timeout.connect(partial(autosave_now, main_window))
    timer.start()
    setattr(main_window, _TIMER_ATTR, timer)
    action = QtGui.QAction("Recover Autosaved Alignments...")
    action.triggered.connect(partial(recover_autosave, main_window))
    setattr(main_window, ACTION_ATTR, action)
    _sync_action(main_window)


def _sync_action(main_window: "MainWindow") -> None:
    action = getattr(main_window, ACTION_ATTR, None)
    if action is not None:
        action.setEnabled(autosave_path(main_window).is_file())


def autosave_now(main_window: "MainWindow") -> bool:
    """Writes unsaved alignments to the autosave file when Autosave Alignments is
    on and its interval has passed; True if it wrote."""
    control = getattr(main_window, "control", None) or {}
    if not control.get("AlignmentsAutosaveToggle", True):
        return False
    minutes = max(1, int(control.get("AlignmentsAutosaveIntervalSlider", 2)))
    if time.monotonic() - getattr(main_window, _LAST_ATTR, 0.0) < minutes * 60 - 1:
        return False
    store = getattr(main_window, "alignments", None)
    if store is None or store.media_path is None or not store.has_unsaved_changes():
        return False
    revision = store._revision
    if getattr(main_window, _REVISION_ATTR, None) == revision:
        return False
    path = autosave_path(main_window)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        store.save(str(path), mark_saved=False)
    except Exception as e:
        print(f"[WARN] Alignments autosave failed: {e}")
        return False
    setattr(main_window, _REVISION_ATTR, revision)
    setattr(main_window, _LAST_ATTR, time.monotonic())
    _sync_action(main_window)
    return True


def discard_autosave(main_window: "MainWindow") -> None:
    try:
        autosave_path(main_window).unlink(missing_ok=True)
    except OSError as e:
        print(f"[WARN] Could not delete the alignments autosave: {e}")
    setattr(main_window, _REVISION_ATTR, None)
    _sync_action(main_window)


def on_close(main_window: "MainWindow") -> None:
    """Called after the last workspace is saved on exit."""
    store = getattr(main_window, "alignments", None)
    if store is not None and not store.has_unsaved_changes():
        discard_autosave(main_window)


def autosave_info(path: Path) -> Optional[dict]:
    """media_path, frame count and time of an autosave file, or None."""
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            frames = (
                int(len(data["frame_numbers"])) if "frame_numbers" in data else None
            )
    except Exception:
        return None
    return {
        "media_path": meta.get("media_path"),
        "frames": frames,
        "time": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path))),
    }


def recover_autosave(main_window: "MainWindow") -> bool:
    path = autosave_path(main_window)
    info = autosave_info(path) if path.is_file() else None
    title = "Recover Autosaved Alignments"
    if info is None:
        QtWidgets.QMessageBox.information(main_window, title, "No autosave found.")
        _sync_action(main_window)
        return False
    store = main_window.alignments
    video = os.path.basename(info["media_path"] or "")
    if not _same_path(info["media_path"], main_window.video_processor.media_path):
        QtWidgets.QMessageBox.information(
            main_window,
            title,
            f"The autosave from {info['time']} is for {video}. "
            "Load that video first, then try again.",
        )
        return False
    answer = QtWidgets.QMessageBox.question(
        main_window,
        title,
        f"Replace the current alignments of {video} with the autosave from "
        f"{info['time']}? This includes manual edits and painted masks.",
    )
    if answer != QtWidgets.QMessageBox.StandardButton.Yes:
        return False
    store.load(str(path), mark_saved=False)
    from app.ui.widgets.actions import alignment_edit_actions, mask_paint_actions

    alignment_edit_actions._refresh(main_window)
    alignment_edit_actions.refresh_alignment_overlay(main_window)
    mask_paint_actions.refresh_mask_overlay(main_window)
    return True


def _same_path(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))
