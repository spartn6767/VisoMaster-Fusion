# Edit Alignments

Edit Alignments lets you fix face detection by hand on a paused frame: move a face's box or landmarks, add a face the detector missed, delete a wrong one, or pin a face to a target face.

## Before you start

Edit Alignments works on a paused video frame. It needs **Settings > Detectors > Reuse Stored Alignments** on, and Face Compare, Face Mask view, VR180 and Manual Rotation off. Turning Edit Alignments on pauses playback.

## In the viewer

- Click **Edit Alignments** on the edit toolbar. Each stored face shows its box, landmark points and corner squares, and the floating Alignments bar appears. Close it with its ✕ or by clicking Edit Alignments again.
- **Drag a landmark point** to move it.
- **Drag inside a box** to move the whole face.
- **Drag a corner square** to resize the box.
- **Drag on an empty area** to draw a box around a face the detector missed. The app fills in its landmarks on the next refresh.
- **Double-click** where the detector missed a face to copy the face from the nearest earlier frame with stored faces, at the position it had there. If no earlier face is near the click, a box is drawn centred on the click (sized like the earlier face, or a default size) for you to adjust; its landmarks are filled in on the next refresh. Double-clicking an existing box replaces it with the earlier frame's face, like < Paste. Either way, **Z** / **X** then try other frames as described below, and one Ctrl+Z undoes the whole thing.
- **Right-click inside a face's box** for the actions on that face (see below). Right-clicking outside every box shows the usual viewer menu (Fit to View, Zoom, Save Image...).

## Face menu

Right-click inside a face's box. The menu only has actions for that face; the keys are shown next to them.

| Item | What it does |
|---|---|
| Copy (C) | Copies the face. |
| Paste (V) | Pastes the copied face, replacing this face. Greyed out until you copy one. |
| < Paste / Paste > (Z / X) | Pastes the face from the nearest earlier / later frame with stored alignments. Press the key again to try the next frame (see below). |
| Assign to Target | Pins the face to a target face, skipping the usual look-alike matching for it on this frame. |
| Add as Target | Creates a target face from the face. |
| Back of Head | Turns the box into a Back of Head box (ticked), or back into a normal face. See below. |
| Delete Face (M) | Deletes the face from this frame. |
| Fill Between Edits | Blends the face between the previous frame you edited it on and this frame. See below. |
| Switch to Paint Mask | Turns on Paint Mask with this face selected. |

The keys act on the selected face: the face you last right-clicked or dragged, or the only face on the frame.

## Bar buttons

The floating bar has the actions that aren't about one face.

| Button | What it does |
|---|---|
| Undo / Redo | Steps back / forward through your alignment and mask edits (Ctrl+Z / Ctrl+Shift+Z). See below. |
| Re-detect Frame | Drops this frame's stored faces, including manual edits and painted masks, so the frame is detected again. |
| Reset All Alignments | Discards every stored alignment in the video (asks first). |
| Find Frames to Check | Marks frames worth a look in pink on the seek bar. See below. |
| Draw Back of Head | While it's on, boxes you draw are Back of Head boxes. See below. |

## Back of the head

When someone turns away from the camera there's no face to detect, so Hair Makeup stops on those frames. A **Back of Head** box keeps the hair colored:

- **Hair only:** no face is swapped onto the box. Only Hair Makeup runs, with its target face's settings. Face, eyebrow and lip makeup are skipped.
- **Points from the box:** its 5 points are placed from the box shape, as if a face looked at the camera (the landmark detector would guess wrong on the back of a head). The Hair Area is placed from these points, so draw the box around the whole head. You can still drag the points.
- **Target face:** the box needs one, because there's no face to match. With a single target face it's used automatically; otherwise you're asked which one (the first with Hair Makeup on is suggested). With no target face, add one first.
- **How to make one:** turn on **Draw Back of Head** on the bar and drag around the head, or right-click an existing box and choose **Back of Head**. Choosing it again turns the box back into a normal face, whose landmarks are then detected again.
- **In the viewer** the box is purple and labelled "Back of Head -> target".
- **Paint Mask > Hair Mask** works on it like on any face.

Like other alignment edits, a Back of Head box is stored on its own frame only. For a turn that lasts several frames, add one on the first and last frame, then use **Fill Between Edits** on the last one. For example, if the person faces away from frame 300 to 360: draw a Back of Head box on 300 and on 360, then right-click the box on 360 and choose Fill Between Edits.

**When the head fills most of the picture,** the Hair Segmenter can miss the hair: the area it searches reaches far beyond the picture, so the head looks tiny to it. Face Parser or a tighter box may do better.

## Autosave

With **Settings > Detectors > Autosave Alignments** on (the default), unsaved alignments (including masks) are written to an autosave file every **Autosave Every** minutes (default 2). The **Save Alignments** button, right of Paint Mask, saves them with the workspace right away (it asks where if the workspace hasn't been saved yet). If the app closes unexpectedly, load the same video and use **Recover Autosaved Alignments...** in the Workspace section's menu to get them back. The autosave doesn't count as saving, so you're still asked to save before switching videos. A normal exit deletes it once everything is saved.

## Moving between frames

While this mode is on, **K** / **L** step back / forward by the frame-step value (default 1), as they do outside the mode. (C and V do Copy and Paste here.)

**Trying several frames with < Paste / Paste >:** if the pasted face doesn't fit, press **Z** (or < Paste) again: the paste is replaced by the face from the next earlier frame. **X** (Paste >) moves the other way, one frame at a time, through this frame (which puts your original face back) and on to later frames. For example, on frame 120: Z, Z, Z pastes from 119, 118, 117; X then goes back to 118. The bar shows "From frame N" while a pasted face is shown. The whole run is one undo step, so Ctrl+Z gives you back the face you had before the first press. Changing frame, selecting another face or making any other edit ends the run; the next Z / X starts from the nearest frame again.

**Keys for the face actions:** **C** Copy, **V** Paste, **Z** < Paste, **X** Paste >, **M** Delete Face. While the mode is on, these replace the keys' usual actions (Z jump to start, X Face Compare).

## Undo and redo

**Undo** and **Redo** on the bar, or **Ctrl+Z** and **Ctrl+Shift+Z**, step back and forward through your edits.

- **One step per edit:** each drag, drawn face, paste, deletion, target assignment, Re-detect Frame and Reset All Alignments.
- **Jumps to the edit:** if the step was on another frame, the viewer goes to that frame first.
- **Shared with Paint Mask:** alignment and mask edits are one history, so Undo on either bar undoes the last edit of either kind.
- **A new edit after undoing** drops the steps you could still redo.
- **History limits:** the last 100 steps are kept. The history is cleared when you switch videos, load a workspace or job, or click Clear or turn off Reuse Stored Alignments in Settings.
- **Add as Target:** Undo removes the assignment, but the new target face stays.

## Finding and fixing runs of frames

- **Edited frames are marked:** frames with manual alignment or mask edits show a cyan mark under the seek bar. Press **N** / **P** to jump to the next / previous one.
- **Find Frames to Check** marks frames in pink above the seek bar where a face's landmarks suddenly jump from the frame before, or where a face disappears for 1 to 3 frames. Press **Shift+N** / **Shift+P** to step through them. Only faces matched to a target face are checked, and only frames that already have stored alignments. Run it again after fixing frames to update the marks.
- **Fill Between Edits:** fix the face by hand on the first and last frame of a bad run, then right-click it on the last frame and choose Fill Between Edits. Every frame in between gets a face blended between the two fixes (box and landmarks), and is marked as edited. Faces on those frames are replaced; painted masks on them are kept. One Undo reverts the whole fill.

## What an edit affects

- **Only the frame you're on.** Unlike painted masks, alignments don't carry forward: every frame stores its own faces. To fix a run of frames, go through them and use **< Paste** or **Paste >** to bring the face over from the neighbouring frame, then adjust it.
- **An edited frame is kept.** It's marked as manually edited, so face detection won't replace it, and it stays in use even if you change detector settings. Only Re-detect Frame or Reset All Alignments removes it.
- **Painted masks move with the landmarks.** A mask is stored relative to the face's landmarks, so fixing them on a frame moves that frame's painted mask too. See [Paint Mask](paint-mask.md) for how masks carry to the next frames.
- **Saving:** alignments are saved with the workspace. If you switch videos with unsaved alignments, or close the app, start a new workspace or load another one with any unsaved changes, the app asks first.
