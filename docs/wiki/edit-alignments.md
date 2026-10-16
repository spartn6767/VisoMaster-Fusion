# Edit Alignments

Edit Alignments lets you fix face detection by hand on a paused frame: move a face's box or landmarks, add a face the detector missed, delete a wrong one, or pin a face to a target face.

## Before you start

Edit Alignments works on a paused video frame. It needs **Settings > Detectors > Reuse Saved Alignments** on, and Face Compare, Face Mask view, VR180 and Manual Rotation off. Turning Edit Alignments on pauses playback.

## In the viewer

- Click **Edit Alignments** on the edit toolbar. Each stored face shows its box, landmark points and corner squares, and the floating Alignments bar appears. Close it with its ✕ or by clicking Edit Alignments again.
- **Drag a landmark point** to move it.
- **Drag inside a box** to move the whole face.
- **Drag a corner square** to resize the box.
- **Drag on an empty area** to draw a box around a face the detector missed. The app fills in its landmarks on the next refresh.
- **Right-click a face** to select it for the bar's buttons.

## Bar buttons

| Button | What it does |
|---|---|
| Undo / Redo | Steps back / forward through your alignment and mask edits (Ctrl+Z / Ctrl+Shift+Z). See below. |
| Re-detect Frame | Drops this frame's stored faces, including manual edits and painted masks, so the frame is detected again. |
| Reset All Alignments | Discards every stored alignment in the video (asks first). |
| Copy | Copies the selected face. |
| Paste | Pastes the copied face, replacing the selected face. |
| < Paste / Paste > | Pastes the face from the nearest earlier / later frame with stored alignments. |
| Assign to Target | Pins the selected face to a target face, skipping the usual look-alike matching for it on this frame. |
| Add as Target | Creates a target face from the selected face. |
| Delete Face | Deletes the selected face from this frame. |
| Find Frames to Check | Marks frames worth a look in pink on the seek bar. See below. |
| Fill Between Edits | Blends the selected face between the previous frame you edited it on and this frame. See below. |

## Autosave

Every 2 minutes, unsaved alignments (including masks) are written to an autosave file. If the app closes unexpectedly, load the same video and use **File > Recover Autosaved Alignments...** to get them back. The autosave doesn't count as saving, so you're still asked to save before switching videos. A normal exit deletes it once everything is saved.

## Moving between frames

While this mode is on, **K** / **L** step back / forward by the frame-step value (default 1), as they do outside the mode. (C and V do Copy and Paste here.)

**Keys for the bar buttons:** **C** Copy, **V** Paste, **Z** < Paste, **X** Paste >, **M** Delete Face. While the mode is on, these replace the keys' usual actions (Z jump to start, X Face Compare).

## Undo and redo

**Undo** and **Redo** on the bar, or **Ctrl+Z** and **Ctrl+Shift+Z**, step back and forward through your edits.

- **One step per edit:** each drag, drawn face, paste, deletion, target assignment, Re-detect Frame and Reset All Alignments.
- **Jumps to the edit:** if the step was on another frame, the viewer goes to that frame first.
- **Shared with Paint Mask:** alignment and mask edits are one history, so Undo on either bar undoes the last edit of either kind.
- **A new edit after undoing** drops the steps you could still redo.
- **History limits:** the last 100 steps are kept. The history is cleared when you switch videos, load a workspace or job, or click Clear or turn off Reuse Saved Alignments in Settings.
- **Add as Target:** Undo removes the assignment, but the new target face stays.

## Finding and fixing runs of frames

- **Edited frames are marked:** frames with manual alignment or mask edits show a cyan mark under the seek bar. Press **N** / **P** to jump to the next / previous one.
- **Find Frames to Check** marks frames in pink above the seek bar where a face's landmarks suddenly jump from the frame before, or where a face disappears for 1 to 3 frames. Press **Shift+N** / **Shift+P** to step through them. Only faces matched to a target face are checked, and only frames that already have stored alignments. Run it again after fixing frames to update the marks.
- **Fill Between Edits:** fix the face by hand on the first and last frame of a bad run, then select it on the last frame and click Fill Between Edits. Every frame in between gets a face blended between the two fixes (box and landmarks), and is marked as edited. Faces on those frames are replaced; painted masks on them are kept. One Undo reverts the whole fill.

## What an edit affects

- **Only the frame you're on.** Unlike painted masks, alignments don't carry forward: every frame stores its own faces. To fix a run of frames, go through them and use **< Paste** or **Paste >** to bring the face over from the neighbouring frame, then adjust it.
- **An edited frame is kept.** It's marked as manually edited, so face detection won't replace it, and it stays in use even if you change detector settings. Only Re-detect Frame or Reset All Alignments removes it.
- **Painted masks move with the landmarks.** A mask is stored relative to the face's landmarks, so fixing them on a frame moves that frame's painted mask too. See [Paint Mask](paint-mask.md) for how masks carry to the next frames.
- **Saving:** alignments are saved with the workspace. If you switch videos or load a workspace with unsaved alignments, the app asks first.
