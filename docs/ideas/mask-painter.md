> **Status:** implemented in #5 (Paint Mask mode). Kept for reference.

# Idea: manual mask painter with a persistent masks file

**Status:** idea / not started. Follows the alignments cache (#1) and the Edit Alignments mode (#3).

## Goal
Let the user paint mask corrections directly in the viewer, the same way faces can now be corrected in Edit Alignments. Keep those edits in a sidecar file next to the workspace, as the alignments are.

## Proposed design
**Paint in face space, not frame space**
- Strokes are stored in the aligned face crop the swapper uses (e.g. 256×256), warped through each face's `kps_5`.
- A painted correction therefore follows the face as it moves, turns or scales.
- A frame-space mask would slide off as soon as the head moves.

**Storage**
- One small 8-bit mask per face per painted frame, stored in `<name>.masks.npz` beside the workspace or job JSON (the same pattern as `<name>.alignments.npz`).
- A compressed hand-painted mask is a few KB.
- Only painted frames are stored.
- The workspace JSON gets a small `"masks"` header entry pointing to the sidecar.

**Applying it**
- Two layers per mask, **add** and **erase**, combined with the computed mask (occluder / XSeg / parser etc.) where the final swap mask is built.
- Manual strokes override the automatic result.
- The existing Face Mask view shows the combined result live.

**Avoiding painting every frame** (main open question)
- *Hold:* a painted mask applies to the following frames until the next painted keyframe. Because it is in face space, it tracks the face.
- *Blend:* fade between painted keyframes.
- *Marker range:* apply across a marker's frame range.

**UI**
- A "Paint Mask" mode next to Edit Alignments in the viewer's right-click menu.
- Brush and eraser, with size and feather controls.
- Painted strokes are drawn as a **semi-transparent coloured overlay** on the face previewer, with an **opacity slider** (e.g. 0–100%, default about 40–50%). You can see the face underneath while painting.
- Add and erase strokes use different tints. Display opacity only affects the preview; the stored mask values are unchanged.
- The same Copy / Copy Previous / Copy Next / Paste options as alignments.

## Effort / risks
- Larger than the alignment editor. A Qt painting brush is more work than draggable points.
- The mask has to plug into the swap/mask pipeline at the right point.
- Roughly one or two PRs, each about the size of Phase 2.
- Needs GPU and models to verify visually.

## First step when picked up
Read the mask pipeline (where `swap_mask` and the mask options are combined in `frame_worker_standard.py` and the mask processors). Then write a short plan, as was done for alignments.
