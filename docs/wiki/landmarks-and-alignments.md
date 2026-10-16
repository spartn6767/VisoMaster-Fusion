# Landmarks and alignments

How the landmark detection setting (Settings > Detectors) affects swaps and the stored alignments. Based on how the code works; the quality difference hasn't been measured.

## Effect on swaps

- **Landmark detection off:** the swap is placed from the 5 rough points (eyes, nose, mouth corners) the face detector returns with each face box.
- **Landmark detection on:** a separate landmark model places those points more precisely on each detected face. Better points mean the new face is aligned more accurately, wobbles less between frames and holds up better in profile views. The cost is one more model run per face per frame.
- **Features that need detailed landmarks** (expression restore, auto mouth, face editor, makeup and Hair Makeup) run the 203-point model on their own when they're on, whatever this setting is.

Usually worth turning on if you can afford the speed.

## Effect on stored alignments

- **Stored faces keep the points that were in use when they were detected.** Landmark settings are part of the check that decides whether a stored frame is still valid.
- **Changing a landmark setting** (on/off, model, score) makes every automatically found frame re-detect and re-store on the next play, seek or render. The first pass after the change is slower.
- **Frames you edited or painted are kept as they are.** They aren't re-detected, so their points don't pick up the new setting. Re-detect Frame redoes one, but also drops that frame's manual edits and painted masks.

## Recommendation

Choose the landmark setting before fixing alignments and painting masks. Changing it afterwards re-detects everything except your edited frames, which can then look different from their neighbours.

<!-- Signature keys: SIGNATURE_CONTROL_KEYS / detector_signature in app/processors/video_utils/alignment_store.py. Not shown in the Help menu. -->
