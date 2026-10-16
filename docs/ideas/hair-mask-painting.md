# Idea: paint the hair mask like the swap mask

**Status:** idea / not started. Builds on Paint Mask (#5), natural hair recolor / Hair Segmenter (#7) and the Paint Mask improvements in #18.

## Problem
Hair Makeup recolors whatever the hair mask covers. That mask is computed fresh on every frame by Face Parser or Hair Segmenter. When it misses hair (thin ends, highlights, random holes) or tints the wrong thing (shoulders, background, a neighbour's hair), the only fixes are the sliders (Mask Threshold, Edge Softness, Strand Refinement, area size). Sliders act on every frame at once. Paint Mask edits only the swap mask, not the hair mask.

## Proposed
- A **Swap / Hair** switch on the Paint Mask bar. In Hair mode, strokes add hair the model missed or erase wrongly tinted areas, and the overlay shows the hair mask instead of the swap mask.
- Same behavior as painted swap masks:
  - A painted frame is a keyframe; later frames of the same person use it until the next painted frame.
  - This Frame Only, Apply to Range, Copy/Paste, < Paste / Paste >, Reset Frame / From Here / All, the C/V/Z/X/M keys, Undo/Redo and Autosave all work the same.
- The final hair mask = computed hair mask + painted additions − painted erasures, then the existing edge softness/refinement.

## Notes / design points
- Store the edit as a separate add/erase layer on `AlignmentFace` (e.g. `hair_mask`, like `mask`), persisted in `<name>.alignments.npz`. The swap mask and hair edits never affect each other.
- Coordinate space: the swap mask uses canonical face space (`manual_masks`), which only covers the face. Hair needs a larger head-relative space: the Hair Segmenter's tall area, defined from the 5 landmarks (`faceutil.convert_face_landmark_x_to_5`, never `kps[:5]`) and the Hair Area Width / Reach Down settings. Pick one fixed head-relative frame, independent of those sliders, so a painted edit doesn't move when a slider changes.
- Apply point: wherever the hair mask is finalised before recolor (both Face Parser and Hair Segmenter paths), and after the cached `hair_prob` from Stabilize Hair Mask is smoothed, so painted edits aren't averaged away.
- Per-person keyframe lookup can reuse `effective_mask`'s `face_id` hold logic (generalise it to a field name).
- Limit: long hair swings independently of the face landmarks, so a carried edit can drift on later frames. Moving hair needs more frequent keyframes. The docs should say so.

## Effort
Medium to large: new store field and persistence, a head-relative warp, overlay and bar mode switch, hooking into the hair mask pipeline, plus tests. Needs GPU/models to check the look end-to-end.
