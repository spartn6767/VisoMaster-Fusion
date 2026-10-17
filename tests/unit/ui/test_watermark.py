from types import SimpleNamespace

import numpy as np

from app.helpers import watermark
from app.ui.widgets.actions import watermark_actions


def test_draw_watermark_marks_only_the_bottom_right_corner():
    frame = np.zeros((720, 1280, 3), np.uint8)
    out = watermark.draw_watermark(frame, watermark.DEFAULT_TEXT)
    assert out is frame
    assert frame[600:, 900:].max() > 150
    assert frame[:500, :].max() == 0 and frame[:, :400].max() == 0


def test_draw_watermark_wraps_and_handles_odd_frames():
    lines = watermark._wrap(watermark.DEFAULT_TEXT * 3, 1.0, 2, 300)
    assert len(lines) > 1
    tall = np.zeros((1080, 200, 3), np.uint8)
    watermark.draw_watermark(tall, watermark.DEFAULT_TEXT)
    assert tall.max() > 0
    for shape in ((4, 4, 3), (60, 80, 3)):
        watermark.draw_watermark(np.zeros(shape, np.uint8), "x y z")
    blank = np.zeros((100, 100, 3), np.uint8)
    watermark.draw_watermark(blank, "   ")
    assert blank.max() == 0
    frozen = np.zeros((200, 300, 3), np.uint8)
    frozen.flags.writeable = False
    assert watermark.draw_watermark(frozen, "AI").max() > 0


def test_should_draw():
    preview_on = {watermark.PREVIEW_KEY: True}
    assert watermark.should_draw({}, True, True, False)
    assert not watermark.should_draw(preview_on, True, False, False)
    assert watermark.should_draw(preview_on, False, False, True)
    assert not watermark.should_draw({}, False, True, True)


def test_tag_video_file(tmp_path, monkeypatch):
    video = tmp_path / "out.mp4"
    video.write_bytes(b"x")
    calls = []

    def run(args, check):
        calls.append(args)
        open(args[-1], "wb").write(b"tagged")

    monkeypatch.setattr(watermark.subprocess, "run", run)
    assert watermark.tag_video_file(str(video), "Made with AI")
    assert "comment=Made with AI" in calls[0] and video.read_bytes() == b"tagged"
    assert not watermark.tag_video_file(str(tmp_path / "missing.mp4"), "x")
    assert not watermark.tag_video_file(str(video), "")


def test_begin_recording_run(monkeypatch):
    asked = []
    answer = {"value": True}
    monkeypatch.setattr(
        watermark_actions,
        "_prompt",
        lambda mw, text: asked.append(text) or answer["value"],
    )
    mw = SimpleNamespace(control={})
    assert watermark_actions.begin_recording_run(mw)
    assert mw.recording_watermark is True and asked == [watermark.DEFAULT_TEXT]

    answer["value"] = None
    assert not watermark_actions.begin_recording_run(mw)
    assert mw.recording_watermark is True

    mw.control = {watermark_actions.ALWAYS_KEY: True}
    assert watermark_actions.begin_recording_run(mw) and mw.recording_watermark
    mw.control = {watermark_actions.ASK_KEY: False}
    assert watermark_actions.begin_recording_run(mw) and not mw.recording_watermark
    mw.control = {watermark.TEXT_KEY: "  "}
    assert watermark_actions.begin_recording_run(mw) and not mw.recording_watermark
    assert len(asked) == 2
