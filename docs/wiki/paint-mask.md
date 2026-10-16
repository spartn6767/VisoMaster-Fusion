# Paint Mask

Paint Mask lets you fix the swap mask by hand: add areas the swap should cover, or erase areas it shouldn't (hair, hands, glasses). Painted masks are saved with the alignments.

## Before you start

Paint Mask works on a paused video frame. It needs **Settings > Detectors > Reuse Saved Alignments** on, and Face Compare, Face Mask view, VR180 and Manual Rotation off. Turning Paint Mask on pauses playback.

## Painting

- Click **Paint Mask** on the edit toolbar. The floating Paint Mask bar appears. Drag it by its edge to move it. Close it with its ✕ or by clicking Paint Mask again.
- Left-drag over a face to paint. The stroke goes to the face nearest to where you start.
- Strokes **add** to the mask. Hold **Shift** while painting, or turn on **Erase** on the bar, to **erase** instead.
- **Size** and **Feather** are in frame pixels. **Opacity** only changes how see-through the overlay is, not the mask itself.
- The mask is stored relative to the face's landmarks, so it moves, turns and scales with the face and survives a swapper model change.
- How it combines with the computed mask: everything the computed mask covers, plus your painted additions, minus your painted erasures. Erasing always wins.

## How a mask carries to the next frames

When you paint on a frame, that frame gets a **keyframe**: a stored mask for that face. Later frames that don't have a keyframe of their own **use the most recent earlier keyframe** of the same person, up to the next keyframe.

Example with keyframes at frames 100 and 200:

| Frames | Mask used |
|---|---|
| 0–99 | Computed mask only |
| 100–199 | Frame 100's painted mask |
| 200 onward | Frame 200's painted mask |

What an edit changes:

- **Painting on a frame that shows a carried mask** (say frame 150): your stroke is added to the carried mask from frame 100 and stored as a new keyframe at 150. Frames 150–199 now use the new mask. Frames 100–149 keep frame 100's mask, and frame 200 onward is unchanged because frame 200 has its own keyframe.
- **Painting on a keyframe itself** (frame 100): every frame that carries it changes too, so frames 100–199.
- **Edits never flow backwards.** Frames before the frame you paint on are not changed.
- **To fix only one frame** (say frame 150), first go to frame 151 and click **< Paste**. That stores the mask frame 151 currently shows as its own keyframe. Then go back to frame 150 and paint. Your fix stops at frame 150.
- **Masks belong to a person.** A mask follows the target face the face was matched to, so a mask painted on one person never carries onto someone else. A face that isn't matched to any target face doesn't carry its mask to later frames.
- **Each frame places the mask on its own landmarks.** If you fix a face's landmarks with Edit Alignments, the painted mask on that frame moves with them.

## Bar buttons

The buttons act on the selected face: the face you last painted or right-clicked, or the only face on the frame.

| Button | What it does | Effect on later frames |
|---|---|---|
| Copy | Copies the selected face's mask (its own or the carried one). | None |
| Paste | Pastes the copied mask onto the selected face as a keyframe. | Carries forward like a painted mask |
| < Paste / Paste > | Copies the mask from the nearest earlier / later frame with stored alignments onto this frame as a keyframe. | Carries forward like a painted mask |
| Reset Frame | Uses the computed mask on this frame only. | None: later frames keep the mask they had |
| Reset From Here | Stores an empty keyframe, so this frame uses the computed mask. | Later frames use the computed mask too, up to the next painted keyframe |
| Reset All | Removes every painted mask in the video (asks first). Also on the edit toolbar as Reset All Masks. | All frames go back to the computed mask |

## Saving

- Masks are stored with the alignments, so save the workspace to keep them. If you switch videos or load a workspace with unsaved alignments, the app asks first.
- Painting marks the frame as manually edited, so face detection won't replace that frame's stored faces. **Re-detect Frame** (Edit Alignments) drops the frame's stored faces, including their painted masks.
