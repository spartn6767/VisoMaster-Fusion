# Keyboard and mouse shortcuts

Single-letter shortcuts don't work while a text box (such as search) has focus. Hovering over a button shows its key in the tooltip.

## Main window

| Key | Action |
|---|---|
| Space | Play / Stop |
| R | Record start/stop |
| S | Swap Faces toggle |
| T | Theatre Mode |
| F11 | Fullscreen |
| Esc | Leave fullscreen / Theatre Mode |
| L / K | Forward / back by the frame-step value (default 1) |
| Z | Jump to start (< Paste in the edit modes) |
| Ctrl+Z / Ctrl+Shift+Z | Undo / redo an alignment or mask edit (while Edit Alignments or Paint Mask is on) |
| F / Alt+F | Add / remove marker |
| W / Q | Next / previous marker |
| N / P | Next / previous frame with manual alignment or mask edits (cyan marks on the seek bar) |
| Shift+N / Shift+P | Next / previous frame found by Find Frames to Check (pink marks) |
| X | Face Compare (Paste > in the edit modes) |
| Ctrl+0 | Fit to View |
| Ctrl+1 | 100% zoom |

<!-- Tooltip keys: SHORTCUT_HINTS in app/ui/widgets/actions/shortcut_hints.py. -->
<!-- Defined in `MainWindow.keyPressEvent` (`app/ui/main_ui.py`), except X, Ctrl+0 and Ctrl+1, which are View menu actions in the same file. -->

## Edit Alignments and Paint Mask

While either mode is on, these keys replace their usual action and work on the selected face (the last one you right-clicked, dragged or painted), like the right-click face menu. Turning a mode on gives the preview keyboard focus; if a key doesn't respond (for example after typing in a number box), click the preview.

| Key | Edit Alignments | Paint Mask |
|---|---|---|
| C | Copy | Copy |
| V | Paste | Paste |
| Z | < Paste (from the earlier frame; press again for the next earlier one) | < Paste (same) |
| X | Paste > (from the later frame; after Z, steps back toward the original) | Paste > (same) |
| M | Delete Face | Reset Frame |
| 3 | (nothing) | Switch between paint and erase (the bar's Erase box follows) |
| K / L | Back / forward by the frame-step value (as everywhere) | Back / forward by the frame-step value |
| Ctrl+Z / Ctrl+Shift+Z | Undo / redo | Undo / redo |

<!-- Edit-mode keys: EDIT_MODE_KEYS in app/ui/widgets/actions/alignment_edit_actions.py; Face Compare's X shortcut is switched off in edit_toolbar_actions.sync_edit_toolbar. -->

## Mouse with keys

- **Viewer:** the wheel zooms; Shift+wheel switches to the next/previous embedding; middle-drag pans; right-click opens the viewport menu. In Edit Alignments and Paint Mask, right-clicking inside a face box opens that face's menu (Copy, Paste, Delete Face / Reset Frame...) instead.
- **Edit Alignments:** drag on an empty area to draw a face box; double-click to copy the earlier frame's face there (or onto the box you double-click), or draw a box centred on the click if no earlier face is near.
- **Paint Mask:** drag to paint (a click alone paints nothing); press 3 to switch between painting and erasing, or hold Shift while painting to flip it for that stroke. Double-click to paste the earlier frame's mask onto the face there, or the earlier face with its mask where the frame has none. A circle around the pointer shows the brush size; the dashed inner circle shows where the feathered edge starts.
- **Input face and merged embedding cards:** Ctrl+click assigns several to a face; Shift+click selects a range of input faces.
- **Sliders and dropdowns:** the wheel only changes them with Ctrl held, unless the setting that turns on mouse-wheel control for sliders (`SliderMouseWheelControlToggle`) is enabled.
- **Seek bar:** Left/Right move 1 frame; the wheel moves by the frame step. Enter in the frame-number box jumps to that frame.
- **Other sliders:** Left/Right step the value when the slider has focus.

## Embedding editor window

| Key | Action |
|---|---|
| Ctrl+Z | Undo |
| Ctrl+Shift+Z | Redo |
| Ctrl+V | Paste |
| Ctrl+A | Select all |
| Ctrl+D | Deselect all |
| Delete | Delete selected entries |

## Free keys

Not used in the main window: A, B, D, E, G, H, I, J, O, U, Y, the number keys (except 3 in Paint Mask), and most Ctrl/Shift combinations.

<!-- Help > View Shortcuts shows this page (HTML comments like this one are hidden), so update it whenever a shortcut is added or changed. -->
