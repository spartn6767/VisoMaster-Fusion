# Idea: assign manual faces to a target (Assign to Target / Add as Target Face)

**Status:** idea / not started. Follows Edit Alignments (#3) and the mask painter (#5).

## Problem
A face drawn by hand in Edit Alignments gets an embedding (`run_detect_landmark` + `run_recognize_direct`). It is then matched against Target Faces by similarity, like detected faces. It is swapped only if:
- the person is already in Target Faces (Find Faces on a frame where the detector sees them), and
- the similarity clears that target's threshold. Profile views, blur or occlusion often fail this, and lowering the threshold affects every frame for that target.

There is no way to link a manual face to a target or input face by hand.

## Proposed
Two new right-click items in Edit Alignments, on a face:
- **Assign to Target ›** a submenu listing the current target faces. Picking one pins that face to the target: the worker skips embedding matching and uses that target's input faces and parameters.
- **Add as Target Face:** creates a new target face from the manual face's crop and embedding, so input faces can be assigned to it directly.

## Notes / design points
- Store the assignment on the alignment entry, e.g. a persisted `assigned_face_id` next to `face_id`, so it is saved in `<name>.alignments.npz`. The frame stays pinned, like other manual edits.
- `frame_worker_standard`: if the cached face has `assigned_face_id` and that target still exists, use it as `matched_target` instead of `_find_best_target_match`.
- The painted-mask Hold (#5) keys on `face_id`, so it would follow assigned faces automatically.
- Open questions: what happens when the assigned target is removed (fall back to matching? clear?), and should Paste Previous/Next carry the assignment (probably yes)?
- Visual cue: maybe a label with the target's name on the box in Edit Alignments.

## Effort
Small to medium: context menu and store field, a worker match override, and Add as Target reusing the existing target-face card creation. Needs GPU/models to verify end-to-end.
