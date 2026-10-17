"""Asks before closing the app, starting a new workspace or loading another one
throws away unsaved changes or a recording that is still running."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Optional, cast

from PySide6 import QtWidgets

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

_BASELINE_ATTR = "_workspace_saved_fingerprint"
# Workspace fields that change without an edit worth saving, or that fill in
# asynchronously after a load (media lists load in background threads).
_IGNORED_KEYS = (
    "target_medias_data",
    "input_faces_data",
    "issue_frames_by_face",
    "dropped_frames",
    "scan_tools_expanded",
    "parameter_section_states",
    "selected_media_id",
    "selected_target_face_id",
    "last_target_media_folder_path",
    "last_input_media_folder_path",
)


def workspace_fingerprint(main_window: Any) -> Optional[str]:
    """The saveable workspace state as a string, or None if it can't be read."""
    from app.ui.widgets.actions import job_manager_actions

    try:
        data = job_manager_actions._serialize_job_data(main_window)
    except Exception as e:
        print(f"[WARN] Could not snapshot the workspace for unsaved changes: {e}")
        return None
    for key in _IGNORED_KEYS:
        data.pop(key, None)
    return json.dumps(data, sort_keys=True, default=str)


def mark_saved(main_window: Any) -> None:
    """Call after the workspace was saved, loaded or started fresh."""
    setattr(main_window, _BASELINE_ATTR, workspace_fingerprint(main_window))


def has_unsaved_changes(main_window: Any) -> bool:
    store = getattr(main_window, "alignments", None)
    if (
        store is not None
        and getattr(store, "media_path", None)
        and store.has_unsaved_changes()
    ):
        return True
    baseline = getattr(main_window, _BASELINE_ATTR, None)
    if baseline is None:
        return False
    current = workspace_fingerprint(main_window)
    return current is not None and current != baseline


def _ask(
    main_window: Any, title: str, text: str, info: str, accept: str, discard: str
) -> str:
    """ "accept", "discard" or "cancel"."""
    box = QtWidgets.QMessageBox(cast(QtWidgets.QWidget, main_window))
    box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    box.setWindowTitle(title)
    box.setText(text)
    box.setInformativeText(info)
    accept_button = box.addButton(accept, QtWidgets.QMessageBox.ButtonRole.AcceptRole)
    discard_button = box.addButton(
        discard, QtWidgets.QMessageBox.ButtonRole.DestructiveRole
    )
    box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(accept_button)
    # QMessageBox sizes itself to its text; a minimum keeps the buttons readable.
    box.setStyleSheet(
        "QLabel#qt_msgbox_label, QLabel#qt_msgbox_informativelabel "
        "{ min-width: 460px; } "
        "QPushButton { min-width: 130px; padding: 6px 14px; }"
    )
    box.exec()
    clicked = box.clickedButton()
    if clicked is accept_button:
        return "accept"
    return "discard" if clicked is discard_button else "cancel"


def confirm_discard_changes(main_window: "MainWindow", action: str) -> bool:
    """Save / Don't Save / Cancel before ``action`` (e.g. "closing") drops unsaved
    changes. False means the user cancelled."""
    if getattr(main_window, "is_batch_processing", False) or not has_unsaved_changes(
        main_window
    ):
        return True
    path = str(getattr(main_window, "current_workspace_path", "") or "")
    name = f"'{os.path.basename(path)}'" if path else "this workspace"
    choice = _ask(
        main_window,
        "Unsaved Changes",
        f"Save changes to {name} before {action}?",
        "Your changes (faces, settings, markers, alignments and masks) are lost "
        "if you don't save them.",
        "Save",
        "Don't Save",
    )
    if choice == "accept":
        from app.ui.widgets.actions import save_load_actions

        save_load_actions.save_current_workspace(main_window)
        return not has_unsaved_changes(main_window)
    return choice == "discard"


def processing_is_active(main_window: Any) -> bool:
    processor = main_window.video_processor
    job = getattr(main_window, "job_processor", None)
    return bool(
        getattr(processor, "recording", False)
        or getattr(processor, "is_processing_segments", False)
        or getattr(main_window, "is_batch_processing", False)
        or (job is not None and hasattr(job, "isRunning") and job.isRunning())
    )


def save_partial_recording(main_window: Any) -> None:
    """Stops like the Record button: keeps what's recorded so far as the output."""
    processor = main_window.video_processor
    job = getattr(main_window, "job_processor", None)
    if job is not None and hasattr(job, "isRunning") and job.isRunning():
        try:
            job.request_cancel("Cancelled because the app is closing.")
        except Exception as e:
            print(f"[WARN] Failed to cancel the job queue on close: {e}")
    if processor.is_processing_segments:
        processor.finalize_segment_concatenation()
    elif processor.recording:
        processor._finalize_default_style_recording()


def confirm_close(main_window: "MainWindow", ask_unsaved: bool = True) -> bool:
    """Run before the main window closes; False means keep it open."""
    if processing_is_active(main_window):
        choice = _ask(
            main_window,
            "Recording in Progress",
            "A recording or batch run is still in progress.",
            "Save what's recorded so far as a normal output file, or discard it.",
            "Save Recording && Close",
            "Discard && Close",
        )
        if choice == "cancel":
            return False
        if choice == "accept":
            save_partial_recording(main_window)
    return not ask_unsaved or confirm_discard_changes(main_window, "closing")
