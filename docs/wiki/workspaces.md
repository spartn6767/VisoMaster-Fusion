# Workspaces

A workspace file (.json) holds your target media, input faces, embeddings, found faces and their settings, markers and the app settings. Its alignments and painted masks are saved next to it in a matching `.alignments.npz` file.

## Workspace section

The **Workspace** section sits above Target Videos. Click its title to fold it away.

- **Name:** the current workspace file, or "Unsaved workspace" if it has never been saved. A **•** after the name means there are unsaved changes. Hover it for the full path.
- **Open...** opens a saved workspace (File > Load Saved Workspace).
- **Save** saves in place, with the alignments and masks. It asks where if the workspace has no file yet. Shortcut: **Ctrl+S**.
- **Save As...** saves to a new file. Shortcut: **Ctrl+Shift+S**.
- **Recent** lists the last 10 workspaces you opened or saved. Missing files are greyed out; **Remove Missing** and **Clear List** tidy the list.

Ctrl+S works anywhere in the main window, including in Edit Alignments and Paint Mask and while typing in a box.

## Unsaved changes

Closing the app, starting a new workspace or opening another one asks first if there are unsaved changes: **Save**, **Don't Save** or **Cancel**. Adding or removing media files alone doesn't count as a change.
