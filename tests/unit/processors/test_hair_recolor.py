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
    segmenter_box,
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
    from app.processors.hair_recolor import hair_prob_for_crop, hair_probability

    calls: list[str] = []

    def parser_logits(img):
        calls.append("logits")
        return torch.zeros(1, 19, 512, 512)

    saved: list[tuple] = []
    store_map: dict[tuple, np.ndarray] = {}
    cache = HairMaskCache(
        load=lambda crop: store_map.get(crop),
        save=lambda prob, crop: saved.append(crop),
    )
    img = torch.full((3, 512, 512), 40, dtype=torch.uint8)

    def prob_fn(crop):
        return hair_probability(parser_logits(crop))

    hair_prob_for_crop(img, prob_fn, cache, CROP)
    assert calls == ["logits"] and saved == [CROP]

    calls.clear()
    store_map[CROP] = np.ones((HAIR_PROB_SIZE, HAIR_PROB_SIZE), dtype=np.float32)
    prob = hair_prob_for_crop(img, prob_fn, cache, CROP)
    assert calls == []
    assert prob.shape == (1, 512, 512) and prob.min() > 0.99


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

    def compute(path, parameters, control):
        calls.append((path, dict(parameters)))
        return (10.0, 20.0, 30.0) if path == "a.png" else None

    worker._compute_input_hair_color = compute
    params = {
        "HairMakeupAreaSizeDecimalSlider": 2.5,
        "HairMakeupAreaShiftDecimalSlider": -0.125,
    }
    assert worker.sample_input_hair_color("a.png", params, {}) == (10.0, 20.0, 30.0)
    assert worker.sample_input_hair_color("a.png", params, {}) == (10.0, 20.0, 30.0)
    assert worker.sample_input_hair_color("b.png", params, {}) is None
    assert worker.sample_input_hair_color("b.png", params, {}) is None
    assert len(calls) == 2
    params["HairMakeupAreaSizeDecimalSlider"] = 3.0
    worker.sample_input_hair_color("a.png", params, {})
    assert len(calls) == 3
    params["HairMakeupMaskModelSelection"] = "Hair Segmenter"
    worker.sample_input_hair_color("a.png", params, {})
    assert len(calls) == 4


def test_guided_refine_pulls_spilled_mask_back_to_hair_edge():
    from app.processors.hair_recolor import guided_refine_mask

    img = torch.full((3, 64, 64), 220, dtype=torch.uint8)  # bright background
    img[:, 16:48, 16:48] = 30  # dark hair
    mask = torch.zeros(1, 64, 64)
    mask[:, 10:54, 10:54] = 1.0  # parser mask spilling 6px onto the background
    refined = guided_refine_mask(img, mask, radius=6, eps=1e-3)
    assert refined.shape == (1, 64, 64)
    spill = refined[0, 24:40, 10:14].mean()
    hair = refined[0, 24:40, 20:44].mean()
    assert hair > 0.9
    assert spill < 0.7
    assert guided_refine_mask(img, mask, radius=0) is mask


def test_compute_input_hair_color_end_to_end(monkeypatch):
    import threading
    from types import SimpleNamespace

    import numpy as np

    from app.processors.workers import function_worker as fw_mod

    bgr = np.zeros((300, 300, 3), dtype=np.uint8)
    bgr[:] = (40, 90, 150)  # BGR -> RGB (150, 90, 40)
    monkeypatch.setattr(fw_mod.misc_helpers, "read_image_file", lambda p: bgr)

    worker = fw_mod.FunctionWorker.__new__(fw_mod.FunctionWorker)
    worker.mp = SimpleNamespace(device="cpu")
    worker._input_hair_colors = {}
    worker._input_hair_lock = threading.Lock()
    kps = np.array(
        [[120, 130], [180, 130], [150, 160], [125, 190], [175, 190]], dtype=np.float32
    )
    worker.run_detect = lambda *a, **k: (None, np.array([kps]), None)

    def logits(face):
        assert face.shape == (3, 512, 512) and face.dtype == torch.uint8
        out = torch.zeros(1, 19, 512, 512)
        out[:, 17, :200] = 10.0
        return out

    worker.get_faceparser_logits = logits
    params = {
        "HairMakeupAreaSizeDecimalSlider": 2.5,
        "HairMakeupAreaShiftDecimalSlider": -0.125,
    }
    color = worker.sample_input_hair_color("in.png", params, {})
    assert color is not None
    assert all(abs(c - t) <= 3 for c, t in zip(color, (150, 90, 40)))


def test_neutral_shadows_under_warm_hair_do_not_turn_blue():
    img = torch.zeros(3, 32, 32, dtype=torch.uint8)
    img[:, :, :16] = torch.tensor([110, 80, 55], dtype=torch.uint8).view(3, 1, 1)
    img[:, :, 16:] = torch.tensor([22, 20, 20], dtype=torch.uint8).view(3, 1, 1)
    full = torch.ones(1, 32, 32)

    for color in ([0, 0, 0], [150, 70, 40], [220, 190, 130]):
        out = recolor_hair_lab(img, full, color, strength=1.0, lightness=0.5)
        lab = kc.rgb_to_lab(out.float().div(255).unsqueeze(0))[0]
        assert lab[2].min() > -1.0, color


# ---------------------------------------------------------------------------
# Full-frame hair recolor (Hair Area)
# ---------------------------------------------------------------------------


def test_box_mean_matches_avg_pool():
    import torch.nn.functional as F

    from app.processors.hair_recolor import _box_mean

    x = torch.rand(1, 1, 23, 17)
    for r in (1, 3, 12):
        ref = F.avg_pool2d(x, 2 * r + 1, stride=1, padding=r, count_include_pad=False)
        assert torch.allclose(_box_mean(x, r), ref, atol=1e-5)


def test_mask_to_frame_roi_places_crop_mask_in_frame():
    from app.processors.hair_recolor import mask_to_frame_roi

    mask = torch.ones(1, 64, 64)
    m_c2o = np.array([[0.5, 0.0, 10.0], [0.0, 0.5, 20.0]], dtype=np.float32)
    placed = mask_to_frame_roi(mask, m_c2o, (100, 100))
    assert placed is not None
    roi, (top, bottom, left, right) = placed
    assert (top, left) == (20, 10)
    assert roi.shape == (1, bottom - top, right - left)
    assert roi[0, 2:28, 2:28].min() > 0.99
    assert mask_to_frame_roi(mask, m_c2o, (10, 5)) is None


def _frame_with_face():
    img = torch.zeros(3, 400, 300, dtype=torch.uint8)
    img[0], img[1], img[2] = 90, 60, 40
    kps = np.array(
        [[130, 75], [170, 75], [150, 95], [135, 115], [165, 115]], dtype=np.float32
    )
    return img, kps


def _hair_everywhere(crop):
    assert crop.shape == (3, 512, 512) and crop.dtype == torch.uint8
    logits = torch.zeros(1, 19, 512, 512)
    logits[:, 17] = 10.0
    return logits


def test_hair_area_reaches_long_hair_past_face_crop():
    from app.processors.hair_recolor import recolor_hair_in_frame

    img, kps = _frame_with_face()
    params = {
        "HairMakeupRedSlider": 30,
        "HairMakeupGreenSlider": 60,
        "HairMakeupBlueSlider": 200,
        "HairMakeupStrengthDecimalSlider": 1.0,
        "HairMakeupLightnessDecimalSlider": 0.0,
    }
    default = recolor_hair_in_frame(img, kps, params, _hair_everywhere)
    assert default.dtype == torch.uint8 and default.shape == img.shape
    assert not torch.equal(default[:, 90, 150], img[:, 90, 150])
    assert torch.equal(default[:, 300, 150], img[:, 300, 150])

    wide = dict(
        params,
        HairMakeupAreaSizeDecimalSlider=6.0,
        HairMakeupAreaShiftDecimalSlider=0.5,
        HairMakeupStrandRefineSlider=4,
    )
    out = recolor_hair_in_frame(img, kps, wide, _hair_everywhere)
    assert not torch.equal(out[:, 300, 150], img[:, 300, 150])
    assert int(out[2, 300, 150]) > int(img[2, 300, 150])


def test_hair_recolor_in_frame_keeps_float_frames_float():
    from app.processors.hair_recolor import recolor_hair_in_frame

    img, kps = _frame_with_face()
    out = recolor_hair_in_frame(
        img.float(), kps, {"HairMakeupBlueSlider": 200}, _hair_everywhere
    )
    assert out.dtype == torch.float32
    assert out[2, 90, 150] > img[2, 90, 150].float()


# ---------------------------------------------------------------------------
# Hair Segmenter (MediaPipe selfie multiclass)
# ---------------------------------------------------------------------------

SEGMENTER = {"HairMakeupMaskModelSelection": "Hair Segmenter"}


def _segmenter_all_hair(crop):
    assert crop.shape == (3, 256, 256) and crop.dtype == torch.uint8
    logits = torch.zeros(1, 6, 256, 256)
    logits[:, 1] = 10.0
    return logits


def _parser_must_not_run(crop):
    raise AssertionError("face parser called in Hair Segmenter mode")


def test_segmenter_box_covers_top_of_head_and_long_hair():
    from app.processors.hair_recolor import segmenter_box

    _, kps = _frame_with_face()  # eyes at y=75, face unit 40px
    left, top, right, bottom = segmenter_box(kps, 7.0, 8.0)
    assert top <= 75 - 2.5 * 40 and bottom >= 75 + 8 * 40
    assert (left + right) / 2 == 150 and right - left >= 7 * 40
    _, _, _, lower = segmenter_box(kps, 7.0, 12.0)
    assert lower > bottom
    assert segmenter_box(kps, 7.0, 12.0)[1] == top  # reaching down keeps the top


def test_crop_box_pads_off_frame_with_black():
    from app.processors.hair_recolor import crop_box

    img = torch.full((3, 100, 100), 200, dtype=torch.uint8)
    crop = crop_box(img, (-100, 0, 100, 100), size=64)
    assert crop.shape == (3, 64, 64) and crop.dtype == torch.uint8
    assert int(crop[:, :, :20].max()) == 0
    assert int(crop[:, :, 44:].min()) == 200


def test_hair_segmenter_recolors_long_hair_without_parser():
    from app.processors.hair_recolor import recolor_hair_in_frame

    img, kps = _frame_with_face()
    params = dict(
        SEGMENTER,
        HairMakeupRedSlider=30,
        HairMakeupGreenSlider=60,
        HairMakeupBlueSlider=200,
        HairMakeupStrengthDecimalSlider=1.0,
        HairMakeupLightnessDecimalSlider=0.0,
        HairMakeupStrandRefineSlider=4,
    )
    out = recolor_hair_in_frame(
        img, kps, params, _parser_must_not_run, segmenter_logits=_segmenter_all_hair
    )
    assert out.dtype == torch.uint8 and out.shape == img.shape
    for y in (5, 90, 390):  # crown, face height, far below the shoulders
        assert int(out[2, y, 150]) > int(img[2, y, 150]), y
    assert torch.equal(out[:, 200, 5], img[:, 200, 5])  # outside the area width


def test_hair_segmenter_without_hair_leaves_frame_untouched():
    from app.processors.hair_recolor import recolor_hair_in_frame

    img, kps = _frame_with_face()
    out = recolor_hair_in_frame(
        img,
        kps,
        dict(SEGMENTER, HairMakeupBlueSlider=200),
        _parser_must_not_run,
        segmenter_logits=lambda crop: torch.zeros(1, 6, 256, 256),
    )
    assert torch.equal(out, img)


def test_segmenter_masks_use_their_own_cache_key():
    from app.processors.hair_recolor import hair_area_key, recolor_hair_in_frame

    assert hair_area_key({}) == (2.5, -0.125)
    seg_key = hair_area_key(SEGMENTER)
    assert seg_key == (0.0, 7.0, 8.0, 1.0) and seg_key != hair_area_key({})

    img, kps = _frame_with_face()
    saved: list[tuple] = []
    cache = HairMaskCache(
        load=lambda crop: None, save=lambda prob, crop: saved.append(crop)
    )
    recolor_hair_in_frame(
        img,
        kps,
        dict(SEGMENTER),
        _parser_must_not_run,
        hair_cache=cache,
        segmenter_logits=_segmenter_all_hair,
    )
    assert saved == [seg_key]


def test_input_hair_color_samples_with_segmenter(monkeypatch):
    import threading
    from types import SimpleNamespace

    from app.processors.workers import function_worker as fw_mod

    bgr = np.zeros((400, 300, 3), dtype=np.uint8)
    bgr[:] = (40, 90, 150)
    monkeypatch.setattr(fw_mod.misc_helpers, "read_image_file", lambda p: bgr)
    worker = fw_mod.FunctionWorker.__new__(fw_mod.FunctionWorker)
    worker.mp = SimpleNamespace(device="cpu")
    worker._input_hair_colors = {}
    worker._input_hair_lock = threading.Lock()
    _, kps = _frame_with_face()
    worker.run_detect = lambda *a, **k: (None, np.array([kps]), None)
    worker.get_faceparser_logits = _parser_must_not_run
    worker.get_hair_segmenter_logits = _segmenter_all_hair

    color = worker.sample_input_hair_color("in.png", dict(SEGMENTER), {})
    assert color is not None
    assert all(abs(c - t) <= 3 for c, t in zip(color, (150, 90, 40)))
    assert ("in.png", 0.0, 7.0, 8.0, 1.0, "RetinaFace") in worker._input_hair_colors


def test_segmenter_box_accepts_203_point_landmarks():
    kps5 = np.array(
        [[270, 222], [318, 220], [296, 242], [283, 258], [318, 258]], np.float32
    )
    kps203 = np.zeros((203, 2), np.float32)
    kps203[[197, 198, 201, 48, 66]] = kps5
    assert segmenter_box(kps203, 7.0, 8.0) == segmenter_box(kps5, 7.0, 8.0)


def _two_people_logits():
    """Background everywhere, plus two separate people each with hair on top."""
    logits = torch.zeros(1, 6, 256, 256)
    logits[:, 0] = 10.0
    for x0 in (20, 160):
        logits[:, 0, 60:250, x0 : x0 + 70] = 0.0
        logits[:, 1, 60:120, x0 : x0 + 70] = 10.0  # hair
        logits[:, 3, 120:250, x0 : x0 + 70] = 10.0  # face/body
    return logits


def test_segmenter_keeps_only_hair_reaching_this_head():
    from app.processors.hair_recolor import segmenter_hair_probability

    logits = _two_people_logits()
    both = segmenter_hair_probability(logits)[0]
    assert both[90, 50] > 0.9 and both[90, 190] > 0.9
    left = segmenter_hair_probability(logits, (55.0, 110.0, 30.0, 40.0))[0]
    assert left[90, 50] > 0.9
    assert float(left[:, 150:].max()) == 0.0


def test_segmenter_without_hair_on_head_recolors_nothing():
    from app.processors.hair_recolor import segmenter_hair_probability

    prob = segmenter_hair_probability(_two_people_logits(), (128.0, 10.0, 8.0, 8.0))
    assert float(prob.max()) == 0.0


def test_segmenter_head_is_above_the_eyes_in_crop_pixels():
    from app.processors.hair_recolor import segmenter_box, segmenter_head

    kps = np.array(
        [[270, 222], [318, 220], [296, 242], [283, 258], [318, 258]], np.float32
    )
    box = segmenter_box(kps, 7.0, 8.0)
    cx, cy, rx, ry = segmenter_head(kps, box)
    eye_y = (221 - box[1]) * 256 / (box[3] - box[1])
    assert abs(cx - 128) < 3 and cy < eye_y and rx > 0 and ry > 0


def test_isolate_person_toggle_changes_cache_key():
    from app.processors.hair_recolor import hair_area_key

    off = {**SEGMENTER, "HairMakeupIsolatePersonToggle": False}
    assert hair_area_key(off) == (0.0, 7.0, 8.0, 0.0)
    assert hair_area_key(off) != hair_area_key(SEGMENTER)


def _touching_hair_logits():
    """Two people whose hair joins into one region, left one with long hair."""
    logits = torch.zeros(1, 6, 256, 256)
    logits[:, 0] = 10.0
    logits[:, 0, 60:120, 20:230] = 0.0
    logits[:, 1, 60:120, 20:230] = 10.0
    logits[:, 0, 120:250, 20:35] = 0.0
    logits[:, 1, 120:250, 20:35] = 10.0
    return logits


def test_segmenter_splits_shared_hair_by_nearest_head():
    from app.processors.hair_recolor import segmenter_hair_probability

    left_head, right_head = (55.0, 110.0, 30.0, 40.0), (195.0, 110.0, 30.0, 40.0)
    logits = _touching_hair_logits()
    alone = segmenter_hair_probability(logits, left_head)[0]
    assert alone[90, 200] > 0.9  # without the other face, the joined hair is kept
    split = segmenter_hair_probability(logits, left_head, [right_head])[0]
    assert split[90, 50] > 0.9 and split[200, 27] > 0.9
    assert float(split[:, 140:].max()) == 0.0
    right = segmenter_hair_probability(logits, right_head, [left_head])[0]
    assert right[90, 200] > 0.9 and float(right[:, :110].max()) == 0.0


def test_other_faces_only_split_hair_they_touch():
    from app.processors.hair_recolor import segmenter_hair_probability

    logits = _two_people_logits()
    head = (55.0, 110.0, 30.0, 40.0)
    far = (200.0, 30.0, 10.0, 10.0)  # touches no hair of this person
    assert torch.equal(
        segmenter_hair_probability(logits, head, [far]),
        segmenter_hair_probability(logits, head),
    )


def test_painted_hair_edit_adds_outside_and_erases_inside():
    import numpy as np

    from app.processors.hair_recolor import apply_hair_edit, capture_hair_mask
    from app.processors.video_utils import manual_masks as mm

    kps = mm.HAIR_SPACE.kps * (200 / mm.HAIR_MASK_SIZE) + 20  # canvas: 20..220
    size = mm.HAIR_MASK_SIZE
    edit = mm.empty_mask(mm.HAIR_SPACE)
    edit[mm.ADD_LAYER, : size // 4] = 255  # add along the top
    edit[mm.ERASE_LAYER, int(size * 0.45) :] = 255  # erase from frame y 110 down
    computed = torch.ones(1, 40, 40)
    placed = (computed, (100, 140, 100, 140), 1.0)
    assert capture_hair_mask(placed, kps).max() == 255

    out, (top, bottom, left, right), _ = apply_hair_edit(
        placed, edit, kps, (300, 300), torch.device("cpu")
    )
    assert abs(top - 20) <= 1 and abs(left - 20) <= 1
    assert bottom >= 219
    assert float(out[0, 30 - top, 100 - left]) > 0.9  # painted addition
    assert float(out[0, 104 - top, 110 - left]) > 0.9  # computed hair kept
    assert float(out[0, 139 - top, 110 - left]) < 0.1  # erased

    out, roi, _ = apply_hair_edit(None, edit, kps, (300, 300), torch.device("cpu"))
    assert float(out.max()) > 0.9 and abs(roi[0] - 20) <= 1
    assert (
        apply_hair_edit(
            None, mm.empty_mask(mm.HAIR_SPACE), kps, (300, 300), torch.device("cpu")
        )
        is None
    )
    assert np.isfinite(out.numpy()).all()
