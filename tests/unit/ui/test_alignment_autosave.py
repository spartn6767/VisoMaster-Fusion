from types import SimpleNamespace

import numpy as np
from PySide6 import QtWidgets

from app.processors.video_utils.alignment_store import AlignmentStore
from app.ui.widgets.actions import alignment_autosave as aas
from app.ui.widgets.actions import alignment_edit_actions as aea
from app.ui.widgets.actions import mask_paint_actions as mpa


def test_autosave_keeps_unsaved_state_and_recovers(tmp_path, monkeypatch):
    monkeypatch.setattr(aea, "_refresh", lambda mw: None)
    monkeypatch.setattr(aea, "refresh_alignment_overlay", lambda mw: None)
    monkeypatch.setattr(mpa, "refresh_mask_overlay", lambda mw: None)
    store = AlignmentStore()
    store.bind_media("/v.mp4")
    store.add_face(
        3, {"bbox": [0.1, 0.1, 0.3, 0.3], "kps_5": np.full((5, 2), 0.2)}, 1, 1
    )
    main_window = SimpleNamespace(
        project_root_path=tmp_path,
        alignments=store,
        video_processor=SimpleNamespace(media_path="/v.mp4"),
    )
    path = aas.autosave_path(main_window)
    assert aas.autosave_now(main_window)
    assert path.is_file() and store.has_unsaved_changes()
    assert not aas.autosave_now(main_window)
    info = aas.autosave_info(path)
    assert info is not None and info["media_path"] == "/v.mp4" and info["frames"] == 1

    shown = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information", lambda *a, **k: shown.append(a)
    )
    main_window.video_processor.media_path = "/other.mp4"
    store.bind_media("/other.mp4")
    assert not aas.recover_autosave(main_window) and shown

    main_window.video_processor.media_path = "/v.mp4"
    store.bind_media("/v.mp4")
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    assert aas.recover_autosave(main_window)
    assert 3 in store and store.has_unsaved_changes()

    aas.on_close(main_window)
    assert path.is_file()
    store.save(str(tmp_path / "w.alignments.npz"))
    aas.on_close(main_window)
    assert not path.is_file()
