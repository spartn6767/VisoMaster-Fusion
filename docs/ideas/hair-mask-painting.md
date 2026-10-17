# Idea: paint the hair mask like the swap mask

**Status:** hair-mask painting built (Paint Mask > Hair Mask, see `docs/wiki/paint-mask.md`); the Back of Head box type is not started. Builds on Paint Mask (#5), natural hair recolor / Hair Segmenter (#7) and the Paint Mask improvements in #18.

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

## Back of the head (no face visible)
When a person turns away there's no face, so nothing is detected and Hair Makeup doesn't run.

**Workaround that works today** (tested by the user on their machine; it fails when the 5 points are misplaced):
1. In Edit Alignments, draw a box around the head.
2. Drag the 5 points to where the eyes, nose and mouth would be if the person faced the camera: eyes level, about a third down the box; nose in the middle; mouth corners lower. The hair search area is placed from these points.
3. Assign to Target: the target that has Hair Makeup on. A drawn box won't match any target by similarity.
4. In Paint Mask, erase over the whole face area. Otherwise a face is swapped onto the back of the head; Hair Makeup recolors the frame independently of the painted swap mask.
5. Alignment edits don't carry forward: set up two frames and use Fill Between Edits for the frames in between.

**Proposed "Back of Head" box type** to automate this:
- A flag on the manual `AlignmentFace` (e.g. `hair_only`), set from a draw option or a bar button. Persisted.
- Skip `_refine_manual_alignments` for it: keep `template_kps_for_bbox` points instead of the landmark detector's guess, and don't compute an embedding.
- Require an assigned target (or prompt for one when drawing it).
- In `frame_worker_standard`, skip `swap_core` for it and run only `swap_edit_face_core_makeup` with that target's parameters (Hair Makeup only; face/lips/brows makeup make no sense there).
- Optional: a sampled hair color (from a frame where the face is visible) to filter the segmenter result and fill gaps.
- Open question: the Hair Segmenter is trained on selfies, so check its quality on backs of heads (CPU test on a few photos) before building.

## Effort
Medium to large: new store field and persistence, a head-relative warp, overlay and bar mode switch, hooking into the hair mask pipeline, plus tests. Needs GPU/models to check the look end-to-end.
