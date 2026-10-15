# Idea: natural hair recolor (luminance-preserving, tone remap, soft mask)

**Status:** idea / not started.

## Problem
Hair Makeup (`face_parser_makeup_direct_rgb_masked` in `app/processors/face_editors.py`) applies a fixed RGB color with a per-channel Photoshop **Overlay** blend over a hard face-parser hair mask (label 17, argmax, 5px blur), inside the 512px face crop. It looks unnatural because:
- Overlay barely lightens dark pixels, so dark-to-blonde or red gives a muddy tint.
- Per-channel RGB blending shifts luminance, flattening strand highlights and shadows.
- The hard mask cuts out edges and flyaways.
- Hair outside the face crop (long hair) is not recolored.

## Proposed (in-crop, small PR)
1. **Recolor in Lab:** keep L (brightness and texture), replace a/b with the target chroma, weighted by Strength.
2. **Tone remap:** a Lightness slider maps the hair's own L range (for example its 5th to 95th percentile) toward the target color's L while keeping contrast, so dark-to-blonde and blonde-to-dark look natural.
3. **Soft mask:** use parser softmax confidence for hair instead of the argmax map, plus edge feathering.
4. **Controls:** color (picker or R/G/B), Strength, Lightness, optionally Saturation and Edge softness. Keep the old Overlay mode as a fallback option for existing workspaces.

## Later / separate
- Full-frame hair mask to recolor hair outside the face crop (needs a hair or human-parsing model). Related to the whole-frame skin-tone idea: a skin mask from face-parser skin sampling or a human-parsing model, with HSV/temperature sliders and auto-match to the swapped face.

## Notes
- Torch-only and runs on the existing parser output; no new model for steps 1 to 3.
- Needs GPU and models to tune visually.
