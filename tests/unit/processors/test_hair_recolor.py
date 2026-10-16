"""Tests for the luminance-preserving hair recolor (no models needed)."""

from __future__ import annotations

import kornia.color as kc
import torch

from app.processors.hair_recolor import (
    HAIR_LABEL,
    feather_mask,
    recolor_hair_lab,
    soft_hair_mask,
)


def _logits_with_hair_prob(prob: torch.Tensor) -> torch.Tensor:
    """Build [1,19,H,W] logits whose softmax gives ``prob`` for the hair class."""
    h, w = prob.shape
    logits = torch.zeros(1, 19, h, w)
    other = (1.0 - prob) / 18.0
    logits[:] = torch.log(other.clamp(min=1e-6))
    logits[0, HAIR_LABEL] = torch.log(prob.clamp(min=1e-6))
    return logits


def _striped_dark_hair(size: int = 32) -> torch.Tensor:
    img = torch.zeros(3, size, size, dtype=torch.uint8)
    img[0] = 40
    img[1] = 30
    img[2] = 25
    img[:, ::2, :] += 20  # strands: alternating brighter rows
    return img


def test_soft_mask_ramps_around_threshold():
    prob = torch.tensor([[0.1, 0.45, 0.5, 0.55, 0.9]])
    mask = soft_hair_mask(_logits_with_hair_prob(prob), threshold=0.5)
    assert mask.shape == (1, 1, 5)
    values = mask[0, 0].tolist()
    assert values[0] == 0.0
    assert values[-1] == 1.0
    assert 0.0 < values[1] < values[2] < values[3] < 1.0
    assert abs(values[2] - 0.5) < 1e-3


def test_higher_threshold_shrinks_mask():
    prob = torch.linspace(0.0, 1.0, 50).view(1, 50)
    logits = _logits_with_hair_prob(prob)
    assert (
        soft_hair_mask(logits, threshold=0.7).sum()
        < soft_hair_mask(logits, threshold=0.3).sum()
    )


def test_missing_parser_logits_give_empty_mask():
    mask = soft_hair_mask(torch.zeros(1, 19, 8, 8))
    assert mask.sum().item() == 0.0


def test_feather_softens_edge_and_stays_in_range():
    mask = torch.zeros(1, 32, 32)
    mask[:, :, 16:] = 1.0
    out = feather_mask(mask, 4)
    assert out.shape == mask.shape
    assert 0.0 < out[0, 0, 15].item() < 1.0
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert torch.equal(feather_mask(mask, 0), mask)


def test_empty_mask_or_zero_strength_leaves_image_unchanged():
    img = _striped_dark_hair()
    full = torch.ones(1, 32, 32)
    assert torch.equal(recolor_hair_lab(img, torch.zeros(1, 32, 32), [255, 0, 0]), img)
    assert torch.equal(recolor_hair_lab(img, full, [255, 0, 0], strength=0.0), img)


def test_recolor_moves_hue_and_keeps_strand_contrast():
    img = _striped_dark_hair()
    full = torch.ones(1, 32, 32)
    out = recolor_hair_lab(img, full, [200, 30, 30], strength=1.0, lightness=0.0)

    lab_in = kc.rgb_to_lab(img.float().div(255).unsqueeze(0))[0]
    lab_out = kc.rgb_to_lab(out.float().div(255).unsqueeze(0))[0]
    # Redder: a* goes up clearly.
    assert lab_out[1].mean() > lab_in[1].mean() + 10
    # Lightness 0: brightness of each pixel is preserved, so strands survive.
    assert torch.allclose(lab_out[0], lab_in[0], atol=2.0)
    strand_contrast_in = lab_in[0, 0].mean() - lab_in[0, 1].mean()
    strand_contrast_out = lab_out[0, 0].mean() - lab_out[0, 1].mean()
    assert abs(strand_contrast_out - strand_contrast_in) < 2.0


def test_lightness_brightens_dark_hair_toward_blonde_target():
    img = _striped_dark_hair()
    full = torch.ones(1, 32, 32)
    keep = recolor_hair_lab(img, full, [230, 200, 140], lightness=0.0)
    match = recolor_hair_lab(img, full, [230, 200, 140], lightness=1.0)
    l_keep = kc.rgb_to_lab(keep.float().div(255).unsqueeze(0))[0, 0]
    l_match = kc.rgb_to_lab(match.float().div(255).unsqueeze(0))[0, 0]
    assert l_match.mean() > l_keep.mean() + 20
    # Strand contrast is kept while brightening.
    assert (l_match[0].mean() - l_match[1].mean()) > 2.0


def test_recolor_only_touches_masked_pixels():
    img = _striped_dark_hair()
    mask = torch.zeros(1, 32, 32)
    mask[:, :16] = 1.0
    out = recolor_hair_lab(img, mask, [200, 30, 30], strength=1.0)
    assert torch.equal(out[:, 16:], img[:, 16:])
    assert not torch.equal(out[:, :16], img[:, :16])


# ---------------------------------------------------------------------------
# Hair mask stabilization cache
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

from app.processors.hair_recolor import (  # noqa: E402
    HairMaskCache,
    prob_from_cache,
    prob_to_cache,
)
from app.processors.video_utils.alignment_store import (  # noqa: E402
    HAIR_PROB_SIZE,
    AlignmentStore,
)

CROP = (2.5, -0.125)


def _store_with_frames(frames: range, face_id: str = "t1") -> AlignmentStore:
    store = AlignmentStore()
    face = {
        "bbox": np.array([10, 10, 50, 50], dtype=np.float32),
        "kps_5": np.full((5, 2), 30, dtype=np.float32),
        "face_id": face_id,
    }
    for n in frames:
        assert store.put(n, [face], 100, 100, "sig")
    return store


def _prob(value: int) -> np.ndarray:
    return np.full((HAIR_PROB_SIZE, HAIR_PROB_SIZE), value, dtype=np.uint8)


def test_prob_cache_round_trip():
    prob = torch.zeros(1, 512, 512)
    prob[:, :256] = 1.0
    cached = prob_to_cache(prob)
    assert cached.shape == (HAIR_PROB_SIZE, HAIR_PROB_SIZE)
    assert cached.dtype == np.uint8
    back = prob_from_cache(cached.astype(np.float32) / 255.0, (512, 512), "cpu")
    assert back.shape == (1, 512, 512)
    assert back[0, 10, 10] > 0.99 and back[0, 500, 10] < 0.01


def test_smoothed_hair_prob_averages_neighbours_of_same_face():
    store = _store_with_frames(range(0, 5))
    assert store.smoothed_hair_prob(2, 0, "t1", CROP, 2) is None
    for n, v in zip(range(5), (0, 0, 255, 0, 0)):
        assert store.set_hair_prob(n, 0, _prob(v), CROP)
    smoothed = store.smoothed_hair_prob(2, 0, "t1", CROP, 2)
    assert smoothed is not None
    assert 0.0 < float(smoothed.mean()) < 1.0
    single = store.smoothed_hair_prob(2, 0, "t1", CROP, 0)
    assert single is not None and np.allclose(single, 1.0)
    assert store.hair_frame_count() == 5


def test_smoothing_ignores_other_faces():
    store = _store_with_frames(range(0, 3))
    other = _store_with_frames(range(0, 3), face_id="t2")
    store.set_hair_prob(1, 0, _prob(255), CROP)
    other.set_hair_prob(0, 0, _prob(0), CROP)
    store.restore(store)  # no-op sanity
    smoothed = store.smoothed_hair_prob(1, 0, "t2", CROP, 1)
    # Neighbouring frames have no cached map for t2, so only this frame counts.
    assert smoothed is not None and np.allclose(smoothed, 1.0)


def test_changing_crop_drops_cached_maps():
    store = _store_with_frames(range(0, 2))
    store.set_hair_prob(0, 0, _prob(200), CROP)
    assert store.smoothed_hair_prob(0, 0, "t1", (3.0, 0.0), 1) is None
    store.set_hair_prob(1, 0, _prob(100), (3.0, 0.0))
    assert store.smoothed_hair_prob(0, 0, "t1", (3.0, 0.0), 0) is None
    assert store.hair_frame_count() == 1


def test_hair_probs_persist_in_sidecar(tmp_path):
    store = _store_with_frames(range(0, 3))
    store.set_hair_prob(1, 0, _prob(77), CROP)
    path = str(tmp_path / "v.alignments.npz")
    store.save(path)
    loaded = AlignmentStore()
    loaded.load(path)
    assert loaded.hair_crop == CROP
    smoothed = loaded.smoothed_hair_prob(1, 0, "t1", CROP, 0)
    assert smoothed is not None and np.allclose(smoothed, 77 / 255.0)


def test_cache_hit_skips_face_parser():
    from types import SimpleNamespace

    from app.processors.face_editors import FaceEditors

    calls: list[str] = []

    def parser_logits(img):
        calls.append("logits")
        return torch.zeros(1, 19, 512, 512)

    def parser_labels(img):
        calls.append("labels")
        return torch.zeros(512, 512, dtype=torch.long)

    editors = FaceEditors.__new__(FaceEditors)
    editors.function_worker = SimpleNamespace(
        get_faceparser_logits=parser_logits, get_faceparser_labels=parser_labels
    )
    params = {
        "HairMakeupEnableToggle": True,
        "HairMakeupRedSlider": 200,
        "HairMakeupGreenSlider": 30,
        "HairMakeupBlueSlider": 30,
        "FaceEditorCropScaleDecimalSlider": CROP[0],
        "FaceEditorVYRatioDecimalSlider": CROP[1],
    }
    saved: list[tuple] = []
    store_map: dict[tuple, np.ndarray] = {}
    cache = HairMaskCache(
        load=lambda crop: store_map.get(crop),
        save=lambda prob, crop: saved.append(crop),
    )
    img = torch.full((3, 512, 512), 40, dtype=torch.uint8)

    editors.apply_face_makeup(img, params, hair_cache=cache)
    assert calls == ["logits"] and saved == [CROP]

    calls.clear()
    store_map[CROP] = np.ones((HAIR_PROB_SIZE, HAIR_PROB_SIZE), dtype=np.float32)
    out, mask = editors.apply_face_makeup(img, params, hair_cache=cache)
    assert calls == []
    assert not torch.equal(out, img)
    assert mask.max() > 0.9


# ---------------------------------------------------------------------------
# Use Input's Hair Color
# ---------------------------------------------------------------------------


def test_sample_hair_color_uses_median_of_confident_hair():
    from app.processors.hair_recolor import sample_hair_color

    img = torch.zeros(3, 64, 64, dtype=torch.uint8)
    img[:, :32] = torch.tensor([150, 90, 40], dtype=torch.uint8).view(3, 1, 1)
    img[:, 0, :4] = 255  # a few highlight pixels should not skew the median
    img[:, 32:] = torch.tensor([220, 180, 160], dtype=torch.uint8).view(3, 1, 1)
    prob = torch.zeros(1, 64, 64)
    prob[:, :32] = 0.9
    color = sample_hair_color(img, prob)
    assert color is not None
    assert all(abs(c - t) <= 3 for c, t in zip(color, (150, 90, 40)))


def test_sample_hair_color_none_without_enough_hair():
    from app.processors.hair_recolor import sample_hair_color

    img = torch.full((3, 64, 64), 100, dtype=torch.uint8)
    prob = torch.zeros(1, 64, 64)
    prob[:, :5, :5] = 1.0
    assert sample_hair_color(img, prob) is None


def test_sample_input_hair_color_is_cached_per_image_and_crop():
    import threading

    from app.processors.workers.function_worker import FunctionWorker

    worker = FunctionWorker.__new__(FunctionWorker)
    worker._input_hair_colors = {}
    worker._input_hair_lock = threading.Lock()
    calls: list[tuple] = []

    def compute(path, scale, vy, control):
        calls.append((path, scale, vy))
        return (10.0, 20.0, 30.0) if path == "a.png" else None

    worker._compute_input_hair_color = compute
    params = {
        "FaceEditorCropScaleDecimalSlider": 2.5,
        "FaceEditorVYRatioDecimalSlider": -0.125,
    }
    assert worker.sample_input_hair_color("a.png", params, {}) == (10.0, 20.0, 30.0)
    assert worker.sample_input_hair_color("a.png", params, {}) == (10.0, 20.0, 30.0)
    assert worker.sample_input_hair_color("b.png", params, {}) is None
    assert worker.sample_input_hair_color("b.png", params, {}) is None
    assert len(calls) == 2
    params["FaceEditorCropScaleDecimalSlider"] = 3.0
    worker.sample_input_hair_color("a.png", params, {})
    assert len(calls) == 3
