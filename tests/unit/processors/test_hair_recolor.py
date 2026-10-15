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
