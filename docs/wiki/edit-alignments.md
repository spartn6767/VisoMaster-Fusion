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
| Re-detect Frame | Drops this frame's stored faces, including manual edits and painted masks, so the frame is detected again. |
| Reset All Alignments | Discards every stored alignment in the video (asks first). |
| Copy | Copies the selected face. |
| Paste | Pastes the copied face, replacing the selected face. |
| < Paste / Paste > | Pastes the face from the nearest earlier / later frame with stored alignments. |
| Assign to Target | Pins the selected face to a target face, skipping the usual look-alike matching for it on this frame. |
| Add as Target | Creates a target face from the selected face. |
| Delete Face | Deletes the selected face from this frame. |

## What an edit affects

- **Only the frame you're on.** Unlike painted masks, alignments don't carry forward: every frame stores its own faces. To fix a run of frames, go through them and use **< Paste** or **Paste >** to bring the face over from the neighbouring frame, then adjust it.
- **An edited frame is kept.** It's marked as manually edited, so face detection won't replace it, and it stays in use even if you change detector settings. Only Re-detect Frame or Reset All Alignments removes it.
- **Painted masks move with the landmarks.** A mask is stored relative to the face's landmarks, so fixing them on a frame moves that frame's painted mask too. See [Paint Mask](paint-mask.md) for how masks carry to the next frames.
- **Saving:** alignments are saved with the workspace. If you switch videos or load a workspace with unsaved alignments, the app asks first.
