# Idea: whole-frame skin tone adjustment (skin mask + grading, auto-match to swapped face)

**Status:** idea / not started. Related to the natural hair recolor idea.

## Goal
Adjust skin tone across the whole frame (face, neck, arms, hands), not just inside the face crop. In particular, make body skin match the swapped face, so the neck and hands don't look off when the source face has a different skin tone.

## Skin mask options
1. **Fixed color thresholds** (YCrCb/HSV skin range): instant and needs no model, but catches skin-colored backgrounds or clothing and struggles in odd lighting.
2. **Face parser plus skin sampling (recommended first):** use the face parser's skin region (already computed) to sample this person's skin color statistics, then threshold the rest of the frame around them. More accurate, still no new model.
3. **Human-parsing model** (body-part segmentation): most accurate, but adds an ONNX model and per-frame GPU cost.

Soften the mask (feathering), and smooth it over time so it doesn't flicker.

## Adjustments
- Hue, Saturation, Brightness, Temperature/Tint sliders, plus Strength.
- **Auto-match:** move body-skin color statistics (mean and spread in Lab) toward the swapped face's skin, so the color is consistent.
- Applied as a post-swap step in the frame pipeline, before final output.

## Effort / risks
- Option 2 with sliders and auto-match: about one PR the size of the alignments cache.
- Needs GPU and models to tune visually; temporal stability is the main risk.
