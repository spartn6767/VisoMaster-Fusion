from PySide6 import QtCore, QtWidgets

from app.helpers.miscellaneous import MediaMetadata
from app.ui.widgets import sortable_widgets


def _video_item(tmp_path):
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    item = sortable_widgets.SortableListWidgetItem()
    item.setFileType("video")
    item.setFileInfo(QtCore.QFileInfo(str(video)))
    return item


def test_format_duration():
    assert sortable_widgets.format_duration(4.4) == "0:04"
    assert sortable_widgets.format_duration(83) == "1:23"
    assert sortable_widgets.format_duration(3723) == "1:02:03"


def test_ensure_media_metadata_probes_once_and_fills_tooltip(tmp_path, monkeypatch):
    item = _video_item(tmp_path)
    calls = []

    def fake_probe(path, file_type):
        calls.append((path, file_type))
        return MediaMetadata(
            width=1920, height=1080, total_frames=2490, frame_rate=29.97
        )

    monkeypatch.setattr(sortable_widgets, "probe_media_metadata", fake_probe)
    assert item.ensureMediaMetadata() is True
    assert item.ensureMediaMetadata() is False
    assert len(calls) == 1

    tooltip = item.to_tooltip()
    assert "Resolution: 1920x1080" in tooltip
    assert "Duration: 1:23" in tooltip
    assert "Length: 2490 frames" in tooltip
    assert "Frame Rate: 29.970 fps" in tooltip


def test_ensure_media_metadata_does_not_retry_failed_probe(tmp_path, monkeypatch):
    item = _video_item(tmp_path)
    calls = []

    def fake_probe(path, file_type):
        calls.append(path)
        return None

    monkeypatch.setattr(sortable_widgets, "probe_media_metadata", fake_probe)
    assert item.ensureMediaMetadata() is False
    assert item.ensureMediaMetadata() is False
    assert len(calls) == 1
    assert "Duration" not in item.to_tooltip()
