from PySide6 import QtWidgets

from app.ui.main_ui import MainWindow

_BUTTONS = (
    "addMarkerButton",
    "removeMarkerButton",
    "previousMarkerButton",
    "nextMarkerButton",
    "frameRewindButton",
    "buttonMediaRecord",
    "buttonMediaPlay",
    "frameAdvanceButton",
    "liveSoundButton",
    "viewFullScreenButton",
    "theatreModeButton",
)


def test_media_controls_layout_keeps_frame_skip_step_box_after_advance_button():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = QtWidgets.QWidget()
    window.mediaLayout = QtWidgets.QWidget(window)
    window.horizontalLayoutMediaButtons = QtWidgets.QHBoxLayout()
    for name in _BUTTONS:
        button = QtWidgets.QPushButton(window.mediaLayout)
        setattr(window, name, button)
        window.horizontalLayoutMediaButtons.addWidget(button)
    spin_box = QtWidgets.QSpinBox(window.mediaLayout)
    window.frameSkipStepSpinBox = spin_box
    window.horizontalLayoutMediaButtons.insertWidget(
        window.horizontalLayoutMediaButtons.indexOf(window.frameAdvanceButton) + 1,
        spin_box,
    )
    window._queue_media_controls_balance = lambda: None

    MainWindow._install_media_controls_layout(window)

    layout = window.mediaControlsMainLayout
    assert not spin_box.isHidden()
    assert layout.indexOf(spin_box) == layout.indexOf(window.frameAdvanceButton) + 1
    assert app is not None
