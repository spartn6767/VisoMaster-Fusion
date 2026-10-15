from types import SimpleNamespace

from app.helpers.miscellaneous import resolve_assigned_target


def test_resolve_assigned_target_uses_face_parameters():
    targets = {"a": SimpleNamespace(face_id="a"), "b": SimpleNamespace(face_id="b")}
    params = {"b": {"SimilarityThresholdSlider": 0.9}}
    target, resolved = resolve_assigned_target(
        "b", targets, params, {"SimilarityThresholdSlider": 0.5, "Other": 1}
    )
    assert target is targets["b"]
    assert resolved["SimilarityThresholdSlider"] == 0.9
    assert resolved["Other"] == 1


def test_resolve_assigned_target_falls_back_when_missing():
    assert resolve_assigned_target("gone", {}, {}, {}) == (None, None)
    assert resolve_assigned_target(None, {"a": object()}, {}, {}) == (None, None)
