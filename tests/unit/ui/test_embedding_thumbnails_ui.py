import json
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6 import QtWidgets

from app.helpers import embedding_thumbnails


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _thumbnail():
    return embedding_thumbnails.encode_thumbnail(np.full((50, 50, 3), 200, np.uint8))


def test_embedding_card_shows_picture_and_details(qapp):
    from app.ui.widgets.widget_components import EmbeddingCardButton

    button = EmbeddingCardButton(
        embedding_name="Bob",
        embedding_store={"arcface": np.ones(4)},
        embedding_id="1",
        main_window=None,
    )
    assert not button.icon().isNull()
    assert button.text() == ""
    assert "no img" in button.toolTip()

    button.set_creation_details(
        {"thumbnail": _thumbnail(), "merge_method": "Median", "source_count": 6}
    )
    assert not button.icon().isNull()
    assert "Median of 6 faces" in button.toolTip()
    assert embedding_thumbnails.metadata_of(button)["source_count"] == 6

    button.kv_map = {"layer": {"k": 1}}
    assert "Includes K/V Maps" in button.toolTip()
    assert "Median of 6 faces" in button.toolTip()
    assert "#4090a3" in button.styleSheet()


def test_embedding_cards_are_face_sized_pictures_sorted_by_name(qapp):
    import app.ui.main_ui  # noqa: F401  (avoids a circular import)
    from PySide6 import QtCore

    from app.ui.widgets.actions import list_view_actions

    list_widget = QtWidgets.QListWidget()
    main_window = SimpleNamespace(
        inputEmbeddingsList=list_widget,
        merged_embeddings={},
        control={"SortEmbeddingsAZToggle": True},
        face_thumbnail_button_size=(70, 70),
    )
    for name in ("Zed", "Amy"):
        list_view_actions.create_and_add_embed_button_to_list(
            main_window,
            name,
            {},
            name,
            details={"thumbnail": _thumbnail()} if name == "Amy" else None,
        )
    buttons = [list_widget.itemWidget(list_widget.item(i)) for i in range(2)]
    assert [b.embedding_name for b in buttons] == ["Amy", "Zed"]
    for button in buttons:
        assert button.size() == QtCore.QSize(70, 70)
        assert button.text() == ""
        assert not button.icon().isNull()


def test_editor_keeps_extra_fields_on_load_and_save(qapp, tmp_path, monkeypatch):
    from app.ui.widgets import advanced_embedding_editor as editor

    source = tmp_path / "in.json"
    source.write_text(
        json.dumps(
            [
                {
                    "name": "Bob",
                    "embedding_store": {"arcface": [1.0]},
                    "thumbnail": "abc=",
                    "kv_map": "kv.pt",
                }
            ]
        )
    )
    captured = []
    loader = editor.FileLoader([str(source)])
    loader.finished.connect(lambda *args: captured.append(args))
    loader.run()
    data, file_type, extras = captured[0]
    assert file_type == "json"
    assert extras == {"Bob": {"thumbnail": "abc=", "kv_map": "kv.pt"}}

    monkeypatch.setattr(editor.QMessageBox, "information", lambda *a, **k: None)
    gui = SimpleNamespace(entry_extras=extras)
    target = tmp_path / "out.json"
    editor.EmbeddingGUI._write_to_file(gui, str(target), data)
    assert json.loads(target.read_text()) == [
        {
            "name": "Bob",
            "embedding_store": {"arcface": [1.0]},
            "thumbnail": "abc=",
            "kv_map": "kv.pt",
        }
    ]


def test_editor_entry_shows_picture(qapp):
    from app.ui.widgets import advanced_embedding_editor as editor

    with_picture = editor.EntryWidget("Bob", thumbnail=_thumbnail())
    without_picture = editor.EntryWidget("Ann")
    labels = lambda w: [  # noqa: E731
        lab
        for lab in w.findChildren(QtWidgets.QLabel)
        if lab.pixmap() and not lab.pixmap().isNull()
    ]
    assert len(labels(with_picture)) == 1
    assert labels(without_picture) == []
