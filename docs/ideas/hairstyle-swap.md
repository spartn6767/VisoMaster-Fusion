# Idea: hairstyle swap

**Status:** parked. The user tried some hairstyle-transfer frameworks on their own and found them painful to set up, with disappointing results. Revisit only if a model clearly handles video.

## What it would be
Replace a person's hairstyle (shape, length, texture), not just its color, from a reference hairstyle image or a text description. Today VisoMaster only recolors hair (Hair Makeup, natural hair recolor in #7).

## How published frameworks work
- Examples: Barbershop, HairCLIP, HairFastGAN, Stable-Hair and similar research projects (GAN- or diffusion-based).
- They take a photo plus a reference hairstyle (or text) and regenerate the hair, usually on an aligned head crop.
- They're built and tested on still, mostly front-facing portraits, not video.

## What fits VisoMaster
- The pipeline already crops each face, runs a model and pastes the result back. A hair model could plug in at the same point as Hair Makeup.
- Existing pieces to build on: hair masks (Face Parser, Hair Segmenter), stored alignments, per-person settings, cached hair masks from Stabilize Hair Mask.

## Hard parts
- **Flicker (main blocker):** each frame is generated separately, so strands and texture change from frame to frame. Face swaps get away with this because faces are tightly aligned. Hair is larger and moves on its own.
- **Shape changes:** longer hair must cover background and shoulders. Shorter hair leaves gaps where the old hair was, which need background inpainting. Recoloring never has this problem.
- **Crop size:** long hair goes well beyond the current face crop, so it would need a larger head crop.
- **Speed:** these models are much slower than the swapper, so expect render-only use, not live preview.
- **Setup:** many StyleGAN-based projects compile CUDA-only extensions and need multi-GB weights. Diffusion-based ones run on CPU, but take minutes per image.
- **Licenses:** many research models are for non-commercial use only. Check before bundling one.

## Possible path if revisited
1. Single image / single frame style swap as an experiment, in a standalone script outside VisoMaster.
2. Video: generate hair on a few keyframes and blend/warp between them (like Fill Between Edits). Similar lengths (short to short, long to long) are the most likely to look good.
3. Smooth over time with the cached hair masks.
