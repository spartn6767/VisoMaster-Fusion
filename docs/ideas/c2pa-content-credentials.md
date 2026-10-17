# Idea: C2PA / Content Credentials for recordings

**Status:** not started. Follows the recording watermark (visible text plus a metadata comment, see `docs/wiki/recording-watermark.md`).

## Problem
The visible watermark can be cropped, and the metadata comment is easy to strip: most editors and social sites drop it. There's no reliable, machine-readable record that a video was modified with AI.

## Proposed
Attach a signed C2PA manifest ("Content Credentials") to watermarked recordings:
- An assertion saying the video was edited with AI (C2PA's AI-generated / AI-edited actions), the app name and version, and optionally which features ran (face swap, makeup).
- Signed with a certificate, so tampering is detectable. Tools like the Content Credentials verify site, and platforms that support C2PA, can read and show it.

## Notes / design points
- Library: the C2PA SDK (`c2pa-python`, Apache 2.0 / MIT) can sign MP4 and JPEG after the file is written, in the same place the metadata comment is added now.
- Certificate: signing needs a certificate. A self-signed one works technically but shows as untrusted; trusted ones come from a C2PA-conformant certificate authority. Decide whether users bring their own or the app ships a test certificate with a warning.
- Re-encoding (most uploads) drops the manifest unless the platform preserves C2PA, so the visible watermark stays the main label.
- Could also cover Save Image and batch images.

## Effort
Medium: a dependency, signing settings (certificate path, key), a post-recording signing step, docs, and tests with a test certificate.
