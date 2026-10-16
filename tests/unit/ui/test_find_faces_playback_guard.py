from types import SimpleNamespace

import pytest

from app.ui.widgets.actions import card_actions, video_control_actions


@pytest.fixture(autouse=True)
def _no_ui_refresh(monkeypatch):
    for name in ("refresh_frame", "update_gpu_memory_progressbar"):
        monkeypatch.setattr(card_actions.common_widget_actions, name, lambda *_: None)
    monkeypatch.setattr(
        video_control_actions, "update_scan_review_button_states", lambda *_: None
    )


class _FakeVideoProcessor:
    def __init__(self, processing=False):
        self.file_type = "video"
        self.processing = processing
        self.media_path = "clip.mp4"
        self.media_capture = None
        self.current_frame_number = 0
        self.media_rotation = 0
        self.calls: list[str] = []

    def stop_processing(self):
        self.calls.append("stop")
        self.processing = False
        # The capture is reopened only after the playback threads are joined.
        self.media_capture = SimpleNamespace(
            set=lambda *_args: self.calls.append("seek")
        )
        return True


def _main_window(video_processor):
    return SimpleNamespace(
        video_processor=video_processor, control={}, scan_issue_worker=None
    )


def test_find_faces_stops_playback_before_reading_frame(monkeypatch):
    vp = _FakeVideoProcessor(processing=True)
    monkeypatch.setattr(
        card_actions.misc_helpers,
        "read_frame",
        lambda *_args: vp.calls.append("read") or (False, None),
    )

    card_actions.find_target_faces(_main_window(vp))

    assert vp.calls == ["stop", "seek", "read"]
    assert vp.processing is False


def test_find_faces_does_not_stop_idle_video(monkeypatch):
    vp = _FakeVideoProcessor()
    vp.media_capture = SimpleNamespace(set=lambda *_args: vp.calls.append("seek"))
    monkeypatch.setattr(
        card_actions.misc_helpers,
        "read_frame",
        lambda *_args: vp.calls.append("read") or (False, None),
    )

    card_actions.find_target_faces(_main_window(vp))

    assert vp.calls == ["seek", "read"]
