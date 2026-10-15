from types import SimpleNamespace

import numpy as np
import pytest

from app.processors.video_utils import manual_masks as mm
from app.processors.video_utils.alignment_store import SOURCE_MANUAL, AlignmentStore
from app.ui.widgets.actions import alignment_edit_actions as aea
from app.ui.widgets.actions import mask_paint_actions as mpa

W, H = 1000, 1000
KPS = mm.CANONICAL_KPS * 2 + [100, 50]
NOSE = tuple(KPS[2])


@pytest.fixture
def main_window(monkeypatch):
    monkeypatch.setattr(aea, "_refresh", lambda mw: None)
    store = AlignmentStore()
    store.bind_media("/v.mp4")
    for n in range(3):
        store.put(
            n,
            [{"bbox": [0.1, 0.05, 0.6, 0.6], "kps_5": KPS / W, "face_id": "A"}],
            1,
            1,
            "sig",
        )
    return SimpleNamespace(
        alignments=store, video_processor=SimpleNamespace(media_path="/v.mp4")
    )


def test_stroke_pins_frame_and_holds_on_later_frames(main_window):
    store = main_window.alignments
    assert mpa.apply_stroke(main_window, 1, 0, [NOSE], W, H, erase=False)
    assert store.get(1, 1, 1, "other")[0]["source"] == SOURCE_MANUAL
    assert store.effective_mask(0, 0) is None
    assert store.effective_mask(2, 0) is not None

    rgba = mpa.overlay_rgba(main_window, 2, 200, 200)
    assert rgba is not None and rgba[..., 3].max() > 0
    assert mpa.overlay_rgba(main_window, 0, 200, 200) is None


def test_reset_mask_returns_to_settings_from_that_frame(main_window):
    mpa.apply_stroke(main_window, 0, 0, [NOSE], W, H, erase=False)
    assert mpa.reset_mask(main_window, 1, 0)
    store = main_window.alignments
    assert store.effective_mask(0, 0) is not None
    assert store.effective_mask(1, 0) is None
    assert store.effective_mask(2, 0) is None


def test_reset_all_masks_needs_confirmation(main_window, monkeypatch):
    mpa.apply_stroke(main_window, 0, 0, [NOSE], W, H, erase=False)
    monkeypatch.setattr(mpa, "_confirm_reset_all", lambda mw: False)
    assert mpa.reset_all_masks(main_window) == 0
    monkeypatch.setattr(mpa, "_confirm_reset_all", lambda mw: True)
    assert mpa.reset_all_masks(main_window) == 1
    assert main_window.alignments.mask_frame_count() == 0


def test_copy_paste_and_paste_from_frame(main_window):
    store = main_window.alignments
    assert not mpa.copy_mask(main_window, 0, 0)
    mpa.apply_stroke(main_window, 0, 0, [NOSE], W, H, erase=True)
    assert mpa.copy_mask(main_window, 0, 0)
    mpa.reset_mask(main_window, 1, 0)
    assert mpa.paste_mask(main_window, 2, 0)
    assert np.array_equal(store.effective_mask(2, 0), store.effective_mask(0, 0))

    assert mpa.paste_mask_from_frame(main_window, 1, 0, 2, 0)
    assert store.effective_mask(2, 0) is None


def test_reset_mask_frame_only_affects_that_frame(main_window):
    store = main_window.alignments
    mpa.apply_stroke(main_window, 0, 0, [NOSE], W, H, erase=False)
    assert mpa.reset_mask_frame(main_window, 1, 0)
    assert [store.effective_mask(n, 0) is not None for n in range(3)] == [
        True,
        False,
        True,
    ]

    mpa.apply_stroke(main_window, 1, 0, [NOSE], W, H, erase=True)
    painted = store.effective_mask(1, 0)
    assert mpa.reset_mask_frame(main_window, 1, 0)
    assert store.effective_mask(1, 0) is None
    assert np.array_equal(store.effective_mask(2, 0), painted)
