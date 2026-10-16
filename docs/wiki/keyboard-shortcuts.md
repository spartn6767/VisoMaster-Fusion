# Keyboard and mouse shortcuts

Single-letter shortcuts don't work while a text box (such as search) has focus.

## Main window

| Key | Action |
|---|---|
| Space | Play / Stop |
| R | Record start/stop |
| S | Swap Faces toggle |
| T | Theatre Mode |
| F11 | Fullscreen |
| Esc | Leave fullscreen / Theatre Mode |
| V / C | Forward / back 1 frame |
| Right / Left | Forward / back 1 frame while Edit Alignments or Paint Mask is on (also when the seek bar has focus) |
| D / A | Forward / back by the frame-step value |
| Z | Jump to start |
| Ctrl+Z / Ctrl+Shift+Z | Undo / redo an alignment or mask edit (while Edit Alignments or Paint Mask is on) |
| F / Alt+F | Add / remove marker |
| W / Q | Next / previous marker |
| N / P | Next / previous frame with manual alignment or mask edits (cyan marks on the seek bar) |
| Shift+N / Shift+P | Next / previous frame found by Find Frames to Check (pink marks) |
| X | Face Compare |
| Ctrl+0 | Fit to View |
| Ctrl+1 | 100% zoom |

<!-- Defined in `MainWindow.keyPressEvent` (`app/ui/main_ui.py`), except X, Ctrl+0 and Ctrl+1, which are View menu actions in the same file. -->

## Mouse with keys

- **Viewer:** the wheel zooms; Shift+wheel switches to the next/previous embedding; middle-drag pans; right-click opens the viewport menu.
- **Paint Mask:** hold Shift while painting to erase. A circle around the pointer shows the brush size; the dashed inner circle shows where the feathered edge starts.
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

Not used in the main window: B, E, G, H, I, J, K, L, M, O, U, Y, the number keys, and most Ctrl/Shift combinations.

<!-- Help > View Shortcuts shows this page (HTML comments like this one are hidden), so update it whenever a shortcut is added or changed. -->
