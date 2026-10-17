import cv2
import numpy as np

from app.helpers import embedding_thumbnails as et


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_closest_face_index_picks_most_similar_source():
    sources = [
        {"arcface": _unit([1, 0, 0]), "kps_5": np.zeros((5, 2))},
        {"arcface": _unit([0.9, 0.1, 0]), "kps_5": np.ones((5, 2)) * 99},
        {"arcface": _unit([0, 1, 0])},
    ]
    merged = {"arcface": _unit([0.85, 0.15, 0]), "kps_5": np.ones((5, 2)) * 99}
    assert et.closest_face_index(merged, sources) == 1


def test_closest_face_index_averages_shared_models_and_handles_none():
    sources = [
        {"a": _unit([1, 0]), "b": _unit([0, 1])},
        {"a": _unit([1, 0]), "b": _unit([1, 0])},
    ]
    merged = {"a": _unit([1, 0]), "b": _unit([1, 0])}
    assert et.closest_face_index(merged, sources) == 1
    assert et.closest_face_index(merged, [{"other": _unit([1, 0])}]) is None
    assert et.closest_face_index(merged, []) is None


def test_encode_thumbnail_round_trip():
    face = np.zeros((200, 160, 3), np.uint8)
    face[..., 2] = 255
    encoded = et.encode_thumbnail(face)
    decoded = cv2.imdecode(
        np.frombuffer(et.thumbnail_bytes(encoded), np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded.shape == (et.THUMBNAIL_SIZE, et.THUMBNAIL_SIZE, 3)
    assert decoded[..., 2].mean() > 240 and decoded[..., 0].mean() < 15
    assert et.encode_thumbnail(None) is None
    assert et.encode_thumbnail(np.zeros((4, 4), np.uint8)) is None


def test_thumbnail_bytes_rejects_bad_input():
    assert et.thumbnail_bytes("not base64!!") is None
    assert et.thumbnail_bytes(None) is None
    assert et.thumbnail_bytes(3) is None


def test_metadata_helpers():
    payload = {"name": "x", "thumbnail": "abc=", "merge_method": "Median"}
    assert et.metadata_from_payload(payload) == {
        "thumbnail": "abc=",
        "merge_method": "Median",
    }

    class Button:
        thumbnail = None
        merge_method = "Mean"
        source_count = 4

    assert et.metadata_of(Button()) == {"merge_method": "Mean", "source_count": 4}
    assert et.metadata_of(object()) == {}


def test_describe_merge_and_tooltip():
    assert et.describe_merge("Median", 6) == "Median of 6 faces"
    assert et.describe_merge("Mean", 1) == "Made from 1 face"
    assert et.describe_merge(None, None) == ""
    assert et.tooltip_text("Bob", None, None, None) == "Bob"
    assert et.tooltip_text("Bob", None, None, None, True) == "Bob (Includes K/V Maps)"
    thumb = et.encode_thumbnail(np.zeros((10, 10, 3), np.uint8))
    tip = et.tooltip_text("<Bob>", thumb, "Median", 6, True)
    assert "data:image/jpeg;base64," in tip
    assert "&lt;Bob&gt;" in tip
    assert "Median of 6 faces" in tip and "Includes K/V Maps" in tip
