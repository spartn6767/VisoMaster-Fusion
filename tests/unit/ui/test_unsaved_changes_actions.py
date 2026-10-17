from types import SimpleNamespace

from app.ui.widgets.actions import job_manager_actions
from app.ui.widgets.actions import unsaved_changes_actions as uca


def _window(**kw):
    return SimpleNamespace(
        alignments=None,
        current_workspace_path="",
        video_processor=SimpleNamespace(recording=False, is_processing_segments=False),
        job_processor=None,
        **kw,
    )


def test_unsaved_changes_ignore_volatile_fields(monkeypatch):
    state = {"control": {"a": 1}, "dropped_frames": [1], "target_medias_data": {}}
    monkeypatch.setattr(
        job_manager_actions, "_serialize_job_data", lambda mw: dict(state)
    )
    mw = _window()
    assert not uca.has_unsaved_changes(mw)  # no baseline yet
    uca.mark_saved(mw)
    state["dropped_frames"] = [1, 2]
    state["target_medias_data"] = {"m": 1}
    assert not uca.has_unsaved_changes(mw)
    state["control"] = {"a": 2}
    assert uca.has_unsaved_changes(mw)
    mw.alignments = SimpleNamespace(
        media_path="/v.mp4", has_unsaved_changes=lambda: True
    )
    uca.mark_saved(mw)
    assert uca.has_unsaved_changes(mw)


def test_confirm_discard_changes(monkeypatch):
    dirty = {"value": True}
    monkeypatch.setattr(uca, "has_unsaved_changes", lambda mw: dirty["value"])
    answers = []
    monkeypatch.setattr(uca, "_ask", lambda *a: answers.pop(0))
    saved = []

    def save(mw):
        saved.append(mw)
        dirty["value"] = not save_works

    monkeypatch.setattr(
        "app.ui.widgets.actions.save_load_actions.save_current_workspace", save
    )
    mw = _window()
    answers[:] = ["discard", "cancel"]
    assert uca.confirm_discard_changes(mw, "closing")
    assert not uca.confirm_discard_changes(mw, "closing")
    save_works = False  # e.g. the user cancelled the Save As dialog
    answers[:] = ["accept"]
    assert not uca.confirm_discard_changes(mw, "closing") and saved
    save_works = True
    dirty["value"] = True
    answers[:] = ["accept"]
    assert uca.confirm_discard_changes(mw, "closing")
    dirty["value"] = False
    assert uca.confirm_discard_changes(mw, "closing")  # nothing to ask


def test_confirm_close_while_recording(monkeypatch):
    finalized = []
    mw = _window()
    mw.video_processor = SimpleNamespace(
        recording=True,
        is_processing_segments=False,
        _finalize_default_style_recording=lambda: finalized.append("default"),
    )
    monkeypatch.setattr(uca, "confirm_discard_changes", lambda mw, action: True)
    answers = ["cancel", "discard", "accept"]
    monkeypatch.setattr(uca, "_ask", lambda *a: answers.pop(0))
    assert not uca.confirm_close(mw)
    assert uca.confirm_close(mw) and finalized == []
    assert uca.confirm_close(mw) and finalized == ["default"]
    mw.video_processor.recording = False
    monkeypatch.setattr(uca, "confirm_discard_changes", lambda mw, action: False)
    assert not uca.confirm_close(mw)
    assert uca.confirm_close(mw, ask_unsaved=False)
