# Recording watermark

The app can burn a short notice, by default "AI Modified Video", into the bottom-right corner of recorded videos. It's off unless you choose it.

## When you record

With **Ask About Watermark When Recording** on (the default), pressing Record asks "Add the watermark to this recording?":

- **Add Watermark:** the text is drawn on every recorded frame, and also stored in the video file's comment metadata.
- **No Watermark:** the recording has no watermark.
- **Cancel:** recording doesn't start.

The answer only applies to that recording; the next one asks again. A **Batch Process** or **Job Manager** run asks once at the start and uses the answer for every video in the run. Recordings with job markers ask once for the whole recording.

Images (Save Image, batch images) don't get the watermark.

## Settings

In **Settings > Video Recording Settings**:

| Setting | What it does |
|---|---|
| Ask About Watermark When Recording | Asks before each recording (default on). Turn it off to record without the watermark and without asking. |
| Always Add Watermark | Adds the watermark to every recording without asking (default off). |
| Watermark Text | The text to draw. Press Enter to apply. Empty text means no watermark. |
| Show Watermark in Preview | Shows the watermark on the preview, to check how it looks (default off). While it's on, Save Image also includes it. |

## How it looks

Solid cyan text with a thin dark outline, in the bottom-right corner. Its height is about 3% of the video height, so it looks the same at 720p and 4K. Long text wraps onto more lines.

## Limits

- **The visible watermark** stays through re-uploads and re-encoding, but it can be cropped or blurred out. It's a clear label, not tamper-proof protection.
- **The metadata comment** is invisible in players (see it in file properties or MediaInfo). Editors, converters and most social sites usually drop it.
