import os

import pytest

from app.helpers.miscellaneous import rename_media_file


def _video(tmp_path, name="clip.mp4"):
    path = tmp_path / name
    path.write_bytes(b"data")
    return str(path)


def test_rename_keeps_extension(tmp_path):
    old = _video(tmp_path)
    new = rename_media_file(old, "  holiday ")
    assert new == str(tmp_path / "holiday.mp4")
    assert os.path.exists(new) and not os.path.exists(old)


def test_rename_does_not_double_extension(tmp_path):
    old = _video(tmp_path)
    assert rename_media_file(old, "holiday.MP4") == str(tmp_path / "holiday.MP4")


def test_rename_appends_extension_when_another_is_typed(tmp_path):
    old = _video(tmp_path)
    assert rename_media_file(old, "take.v2") == str(tmp_path / "take.v2.mp4")


def test_rename_same_name_is_noop(tmp_path):
    old = _video(tmp_path)
    assert rename_media_file(old, "clip") == old
    assert os.path.exists(old)


@pytest.mark.parametrize("name", ["", "   ", "a/b", "a\\b", "what?", ".."])
def test_rename_rejects_invalid_names(tmp_path, name):
    old = _video(tmp_path)
    with pytest.raises(ValueError):
        rename_media_file(old, name)
    assert os.path.exists(old)


def test_rename_refuses_to_overwrite(tmp_path):
    old = _video(tmp_path)
    other = _video(tmp_path, "taken.mp4")
    with pytest.raises(ValueError, match="already exists"):
        rename_media_file(old, "taken")
    assert open(other, "rb").read() == b"data" and os.path.exists(old)
