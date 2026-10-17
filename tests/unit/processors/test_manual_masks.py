import numpy as np

from app.processors.video_utils import manual_masks as mm
from app.processors.video_utils.alignment_store import AlignmentStore

KPS = mm.CANONICAL_KPS * 2 + [100, 50]
NOSE = tuple(KPS[2])
CANON_NOSE = (int(mm.CANONICAL_KPS[2, 1]), int(mm.CANONICAL_KPS[2, 0]))


def _face(face_id):
    return {"bbox": [0.1, 0.1, 0.5, 0.5], "kps_5": KPS / 1000.0, "face_id": face_id}


def test_canonical_keypoints_map_to_identity():
    assert np.allclose(mm.frame_to_mask_matrix(mm.CANONICAL_KPS), np.eye(3))


def test_stroke_is_painted_in_face_space_and_latest_layer_wins():
    mask = mm.paint_stroke(None, [NOSE], KPS, radius=20)
    assert mask[mm.ADD_LAYER][CANON_NOSE] == 255
    # Frame radius 20 at half scale -> about 10 px in mask space.
    assert 18 <= int((mask[mm.ADD_LAYER][CANON_NOSE[0]] > 128).sum()) <= 24

    erased = mm.paint_stroke(mask, [NOSE], KPS, radius=4, erase=True)
    assert erased[mm.ERASE_LAYER][CANON_NOSE] == 255
    assert erased[mm.ADD_LAYER][CANON_NOSE] == 0
    assert mask[mm.ERASE_LAYER][CANON_NOSE] == 0


def test_mask_follows_face_into_frame_and_crop():
    mask = mm.paint_stroke(None, [NOSE], KPS, radius=20)
    frame = mm.mask_to_frame(mask, KPS, 800, 700)
    assert frame.shape == (2, 700, 800)
    assert frame[0, int(NOSE[1]), int(NOSE[0])] > 0.99

    frame_to_crop = np.array([[0.5, 0, -10], [0, 0.5, -5], [0, 0, 1]], dtype=float)
    crop = mm.mask_to_crop(mask, KPS, frame_to_crop, 512)
    cx, cy = frame_to_crop[:2, :2] @ np.asarray(NOSE) + frame_to_crop[:2, 2]
    assert crop[0, int(cy), int(cx)] > 0.99


def test_combine_adds_then_erases():
    out = mm.combine(np.array([0.2, 0.9]), np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert np.allclose(out, [1.0, 0.0])


def test_store_holds_mask_until_next_keyframe_and_reset():
    store = AlignmentStore()
    store.bind_media("/v.mp4")
    for n in range(5):
        store.put(n, [_face("A"), _face("B")], 1, 1, "sig")
    mask = mm.paint_stroke(None, [NOSE], KPS, radius=20)
    store.update_face(1, 0, 1, 1, mask=mask, source="manual")
    assert [store.effective_mask(n, 0) is not None for n in range(5)] == [
        False,
        True,
        True,
        True,
        True,
    ]
    assert store.effective_mask(3, 1) is None
    assert store.effective_mask(7, face_id="A") is not None

    store.update_face(3, 0, 1, 1, mask=mm.empty_mask(), source="manual")
    assert [store.effective_mask(n, 0) is not None for n in range(5)] == [
        False,
        True,
        True,
        False,
        False,
    ]
    assert store.mask_frame_count() == 2
    assert store.clear_masks() == 2
    assert store.effective_mask(2, 0) is None


def test_masks_persist_and_old_files_without_masks_load(tmp_path):
    store = AlignmentStore()
    store.bind_media("/v.mp4")
    store.put(0, [_face("A")], 1, 1, "sig")
    mask = mm.paint_stroke(None, [NOSE], KPS, radius=20)
    store.update_face(0, 0, 1, 1, mask=mask, source="manual")
    path = str(tmp_path / "w.alignments.npz")
    store.save(path)

    loaded = AlignmentStore()
    loaded.load(path)
    assert np.array_equal(loaded.effective_mask(0, 0), mask)

    store.update_face(0, 0, 1, 1, mask_skip=True)
    store.save(path)
    skipped = AlignmentStore()
    skipped.load(path)
    assert skipped.effective_mask(0, 0) is None
    assert np.array_equal(skipped.effective_mask(1, face_id="A"), mask)

    with np.load(path) as data:
        legacy = {
            k: data[k]
            for k in data.files
            if k not in ("mask_index", "masks", "mask_skip")
        }
    legacy_path = str(tmp_path / "old.alignments.npz")
    np.savez_compressed(legacy_path, **legacy)
    old = AlignmentStore()
    old.load(legacy_path)
    assert len(old) == 1 and old.effective_mask(0, 0) is None


def test_crop_mask_round_trips_through_canonical_space():
    frame_to_crop = np.array([[0.5, 0, -10], [0, 0.5, -5], [0, 0, 1]], dtype=float)
    painted = mm.paint_stroke(None, [NOSE], KPS, radius=30)
    crop = mm.mask_to_crop(painted, KPS, frame_to_crop, 512)[mm.ADD_LAYER]
    canonical = mm.crop_to_canonical(crop, KPS, np.linalg.inv(frame_to_crop))
    assert canonical.dtype == np.uint8 and canonical.shape == (256, 256)
    assert canonical[CANON_NOSE] > 240
    assert abs(int((canonical > 128).sum()) - int((painted[0] > 128).sum())) < 40


def test_hair_space_covers_head_and_shoulders():
    kps = mm.HAIR_SPACE.kps
    np.testing.assert_allclose(
        mm.frame_to_mask_matrix(kps, mm.HAIR_SPACE), np.eye(3), atol=1e-6
    )
    eyes = kps[:2].mean(axis=0)
    unit = np.linalg.norm(kps[3:5].mean(axis=0) - eyes)
    assert abs(mm.HAIR_MASK_SIZE / unit - mm.HAIR_AREA_UNITS) < 1e-6
    assert abs(eyes[1] / mm.HAIR_MASK_SIZE - mm.HAIR_EYES_Y) < 1e-6
    mask = mm.paint_stroke(None, [kps[2]], kps, radius=5, space=mm.HAIR_SPACE)
    assert mask.shape == (2, mm.HAIR_MASK_SIZE, mm.HAIR_MASK_SIZE)
    assert mm.as_mask(mask) is None and mm.as_mask(mask, mm.HAIR_SPACE) is not None


def test_hair_masks_hold_reset_and_persist(tmp_path):
    store = AlignmentStore()
    store.bind_media("/v.mp4")
    face = {"bbox": [0.1, 0.1, 0.5, 0.5], "kps_5": mm.CANONICAL_KPS / 1000}
    for n in range(3):
        store.put(n, [dict(face, face_id="A")], 1, 1, "sig")
    hair = mm.empty_mask(mm.HAIR_SPACE)
    hair[0, 10:20, 10:20] = 255
    store.update_face(0, 0, 1, 1, hair_mask=hair)
    assert store.effective_mask(2, 0, layer="hair_mask") is not None
    assert store.effective_mask(2, 0) is None
    store.update_face(1, 0, 1, 1, hair_mask_skip=True)
    assert store.effective_mask(1, 0, layer="hair_mask") is None

    path = str(tmp_path / "a.npz")
    store.save(path)
    loaded = AlignmentStore()
    loaded.load(path)
    np.testing.assert_array_equal(loaded.effective_mask(2, 0, layer="hair_mask"), hair)
    assert loaded.effective_mask(1, 0, layer="hair_mask") is None
    assert loaded.clear_masks("hair_mask") == 2
    assert loaded.mask_frame_count("hair_mask") == 0
