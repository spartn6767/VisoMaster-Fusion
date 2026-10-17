# Idea: Presets that flow better

**Status:** not started. Comes from a review of `app/ui/widgets/actions/preset_actions.py`.

## How presets work today
- A preset is two files in `presets/`: `name.json` holds the selected face's Face Swap, Face Editor and Restorers values; `name_ctl.json` holds everything in Denoiser and Settings (`main_window.control`).
- **Apply** (or double-click) replaces the selected face's values. Denoiser and Settings are only applied when the **Apply Settings** toggle is on.

## Problems
1. **Saving depends on a hidden selection.** If a preset is highlighted, "Save Current as Preset" overwrites it instead of asking for a name; to make a new one you first have to deselect. Selecting two presets just shows an error.
2. **Apply only reaches one face.** It changes the selected face card, so with several faces you repeat it per face; there's no "apply to all faces". With no face selected it changes `current_widget_parameters`, which you can't see, so nothing seems to happen.
3. **Apply Settings carries machine settings.** With it on, a preset also changes the theme, the execution provider (which reloads models), the output folder (`OutputMediaFolder`), watermark options and so on.
4. **Output folder bug.** The code tries to keep the output folder by removing `InputFolder` / `OutputFolder` from the face parameters, but the output folder is the `OutputMediaFolder` control, so it still gets saved and applied.
5. **"Apply Settings" looks like an action but is a toggle.** It's off on every start, and its state only shows in a tooltip.
6. **Inconsistent confirmations.** Double-click asks "Apply preset?" but the Apply button doesn't. Delete doesn't ask (the files go to the Recycle Bin).
7. **No feedback.** You can't see which preset a face uses, whether you changed it since, or what a preset contains. Names aren't checked for characters Windows doesn't allow in file names.
8. **Docs are out of date.** `docs/user_manual.md` says presets cover "all control panel settings" and live in the top toolbar.

## Proposed
- **Separate New and Update:** compact icon buttons with hover hints, like the Workspace section. **New Preset...** always asks for a name; **Update** overwrites the selected preset after a confirmation.
- **Apply to:** **Selected Face** or **All Faces**. With no face selected, Apply is disabled with a hint.
- **"Also apply Denoiser & Settings" checkbox** instead of the toggle button, remembered between sessions.
- **Never save or apply machine settings:** theme, execution provider, output folder and other paths are left out of presets. This also fixes the output folder bug.
- **Same behavior for Apply and double-click:** both apply without a prompt. The toast offers **Undo**, which restores the values from before.
- **Active preset:** the preset a face card uses shows in bold in the list, with "•" when its values changed since applying it.
- **Hover summary:** swap model, restorer, key strengths and when the preset was saved.
- **Small fixes:** confirm before Delete, check names, add Duplicate to the right-click menu, update the manual.

## Notes / design points
- Existing preset files keep working; machine settings in old files are just ignored.
- The active preset name would be stored per face card (and saved in workspaces) so the marker survives reloads.
- Undo could reuse the unsaved-changes fingerprinting or keep a copy of the replaced `ParametersDict` / controls.

## Effort
Small to medium: mostly `preset_actions.py` and the Presets tab layout, plus tests and the manual. The active-preset marker and hover summary could be a follow-up.
