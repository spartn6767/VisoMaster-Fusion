from types import SimpleNamespace

from PySide6 import QtWidgets

from app.processors.video_utils.alignment_store import AlignmentStore
from app.ui.widgets.actions import alignment_edit_actions as aea
from app.ui.widgets.actions import edit_toolbar_actions as eta
from app.ui.widgets.actions import mask_paint_actions as mpa


def _main_window():
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    media = QtWidgets.QWidget()
    vertical = QtWidgets.QVBoxLayout(media)
    slider_row = QtWidgets.QHBoxLayout()
    buttons_row = QtWidgets.QHBoxLayout()
    vertical.addLayout(slider_row)
    vertical.addLayout(buttons_row)
    slider = QtWidgets.QSlider()
    slider.setRange(0, 10)
    slider.setValue(2)
    store = AlignmentStore()
    store.bind_media("/v.mp4")
    return SimpleNamespace(
        mediaLayout=media,
        verticalLayoutMediaControls=vertical,
        horizontalLayoutMediaButtons=buttons_row,
        videoSeekSlider=slider,
        alignments=store,
    )


def test_toolbar_sits_above_media_buttons_and_tracks_state(monkeypatch):
    main_window = _main_window()
    bar = eta.install_edit_toolbar(main_window)
    layout = main_window.verticalLayoutMediaControls
    assert (
        layout.indexOf(bar)
        == layout.indexOf(main_window.horizontalLayoutMediaButtons) - 1
    )
    assert eta.install_edit_toolbar(main_window) is bar

    buttons = main_window.editToolbarButtons
    assert not any(
        b.isEnabled() for k, b in buttons.items() if k.startswith(("re", "reset"))
    )

    main_window.alignments.put(
        2, [{"bbox": [0.1, 0.1, 0.5, 0.5], "kps_5": [[0.2, 0.2]] * 5}], 1, 1, "sig"
    )
    main_window.mask_paint_enabled = True
    eta.sync_edit_toolbar(main_window)
    assert buttons["redetect"].isEnabled() and buttons["reset_alignments"].isEnabled()
    assert not buttons["reset_masks"].isEnabled()
    assert buttons["paint"].isChecked() and not buttons["edit"].isChecked()

    calls = []
    monkeypatch.setattr(
        aea, "set_alignment_edit_mode", lambda mw, on: calls.append(("edit", on))
    )
    monkeypatch.setattr(
        mpa, "set_mask_paint_mode", lambda mw, on: calls.append(("paint", on))
    )
    monkeypatch.setattr(aea, "reset_all_alignments", lambda mw: calls.append("reset"))
    buttons["edit"].click()
    buttons["paint"].click()
    buttons["reset_alignments"].click()
    assert calls == [("edit", True), ("paint", False), "reset"]


def test_floating_alignment_bar(monkeypatch):
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(aea, "_refresh", lambda mw: None)
    view = QtWidgets.QWidget()
    view.resize(1400, 700)
    slider = QtWidgets.QSlider()
    slider.setRange(0, 10)
    slider.setValue(2)
    store = AlignmentStore()
    store.bind_media("/v.mp4")
    face = {"bbox": [0.1, 0.1, 0.5, 0.5], "kps_5": [[0.2, 0.2]] * 5}
    store.put(1, [face], 1, 1, "sig")
    store.put(2, [face, dict(face, bbox=[0.6, 0.1, 0.9, 0.5])], 1, 1, "sig")
    main_window = SimpleNamespace(
        graphicsViewFrame=view,
        videoSeekSlider=slider,
        alignments=store,
        control={"AlignmentsCacheEnableToggle": True},
        video_processor=SimpleNamespace(
            file_type="video", media_path="/v.mp4", processing=False
        ),
        alignment_edit_enabled=True,
        target_faces={},
    )
    eta.sync_alignment_bar(main_window)
    bar = main_window._alignment_bar
    buttons = main_window._alignment_bar_buttons
    assert not bar.isHidden() and bar.layout().count() == 2
    assert buttons["redetect"].isEnabled() and not buttons["copy"].isEnabled()

    aea.select_face(main_window, 2, 1)
    eta.sync_alignment_bar(main_window)
    assert buttons["copy"].isEnabled() and buttons["previous"].isEnabled()
    assert not buttons["next"].isEnabled()
    assert eta.alignment_bar_action(main_window, "copy")
    assert buttons["paste"].isEnabled()
    assert eta.alignment_bar_action(main_window, "delete")
    assert len(store.get(2, 1, 1, None)) == 1

    menu = QtWidgets.QMenu()
    eta.populate_assign_menu(main_window, menu)
    assert [a.text() for a in menu.actions()] == ["No target faces"]

    main_window.alignment_edit_enabled = False
    eta.sync_alignment_bar(main_window)
    assert bar.isHidden()
