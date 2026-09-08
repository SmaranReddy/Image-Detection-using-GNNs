"""Training and inference must build the same feature vector for the same inputs.

The single most damaging class of bug in this pipeline is a silent divergence
between the features the model was fitted on and the features it is served at
inference. These tests pin the shared path.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from relation_prediction.vg_dataset import (  # noqa: E402
    GEO_DIM, GEO_DIM_EXT, geo_extractor,
)

BOXES = [
    ((10.0, 20.0, 60.0, 90.0), (100.0, 30.0, 180.0, 95.0), 200.0, 100.0),
    ((0.0, 0.0, 199.0, 99.0), (50.0, 50.0, 60.0, 60.0), 200.0, 100.0),
    ((30.0, 30.0, 40.0, 40.0), (30.0, 30.0, 40.0, 40.0), 640.0, 480.0),
    ((5.0, 5.0, 15.0, 15.0), (500.0, 400.0, 639.0, 479.0), 640.0, 480.0),
]


@pytest.mark.parametrize("mode,dim", [("basic", GEO_DIM), ("ext", GEO_DIM_EXT)])
def test_geo_extractor_returns_the_declared_width(mode, dim):
    fn, d = geo_extractor(mode)
    assert d == dim
    for subj, obj, w, h in BOXES:
        assert len(fn(subj, obj, w, h)) == dim


@pytest.mark.parametrize("mode", ["basic", "ext"])
def test_dataset_and_predict_use_the_same_geometry_function(mode):
    """VGRelationshipDataset._geo_fn and predict._model_geo_fn must be identical.

    Both resolve through geo_extractor(); if either ever hardcodes an
    extractor, a 19-dim model gets 5-dim features (or vice versa).
    """
    from relation_prediction import predict as rel_predict

    fn, dim = geo_extractor(mode)
    rel_predict._set_geo_mode({"geo_mode": mode}, dim)
    assert rel_predict._model_geo_fn is fn
    assert rel_predict._model_geo_dim == dim
    for subj, obj, w, h in BOXES:
        assert fn(subj, obj, w, h) == rel_predict._model_geo_fn(subj, obj, w, h)


def test_set_geo_mode_refuses_a_width_mismatch():
    """A checkpoint claiming geo_dim=5 with geo_mode="ext" must not load."""
    from relation_prediction import predict as rel_predict

    with pytest.raises(ValueError):
        rel_predict._set_geo_mode({"geo_mode": "ext"}, GEO_DIM)
    with pytest.raises(ValueError):
        rel_predict._set_geo_mode({"geo_mode": "basic"}, GEO_DIM_EXT)


def test_image_size_fallback_keeps_dx_dy_normalised():
    """Training always passes the real image size; inference must too.

    With the img_w=img_h=1.0 defaults, dx/dy become RAW PIXEL offsets (order
    1e2-1e3) instead of image fractions — a ~1000x distribution shift on two
    geometry inputs. _resolve_image_size recovers the size from the PIL image.
    """
    from PIL import Image

    from relation_prediction.predict import _resolve_image_size

    img = Image.new("RGB", (640, 480))
    assert _resolve_image_size(1.0, 1.0, img) == (640.0, 480.0)
    assert _resolve_image_size(200.0, 100.0, img) == (200.0, 100.0)   # explicit wins
    assert _resolve_image_size(1.0, 1.0, None) == (1.0, 1.0)          # nothing to recover

    fn, _ = geo_extractor("ext")
    subj, obj = (10.0, 20.0, 60.0, 90.0), (300.0, 200.0, 400.0, 300.0)
    good = fn(subj, obj, *_resolve_image_size(1.0, 1.0, img))
    bad = fn(subj, obj, 1.0, 1.0)
    assert abs(good[0]) < 1.0 and abs(good[1]) < 1.0
    assert abs(bad[0]) > 100.0, "unnormalised dx should be a raw pixel offset"


@pytest.mark.parametrize("geo_dim,geo_norm", [(GEO_DIM, False), (GEO_DIM_EXT, True)])
def test_feature_norm_groups_span_the_whole_input(geo_dim, geo_norm):
    """_get_feature_group_norms must slice the real geometry block, not a hardcoded 5.

    With geo_dim=19 and the old constant, "geo" covered 5 columns and every
    later group (subj_clip, obj_clip, union, pose) was offset by 14 — the
    reported modality shares were read off the wrong weights.
    """
    from relation_prediction.model import RelationMLP
    from relation_prediction.predict import _get_feature_group_norms

    clip_dim, union_dim, pose_dim = 768, 768, 20
    model = RelationMLP(num_labels=80, num_predicates=21, embed_dim=64,
                        hidden_dims=(256, 128), clip_dim=clip_dim,
                        union_dim=union_dim, pose_dim=pose_dim,
                        geo_dim=geo_dim, geo_norm=geo_norm)
    norms = _get_feature_group_norms(model)
    assert set(norms) == {"subj_label", "obj_label", "geo",
                          "subj_clip", "obj_clip", "union_clip", "pose"}

    in_dim = model.mlp[0].weight.shape[1]
    assert in_dim == 2 * 64 + geo_dim + 2 * clip_dim + union_dim + pose_dim
    # every input column belongs to exactly one group
    total = torch.linalg.vector_norm(model.mlp[0].weight) ** 2
    parts = sum(v ** 2 for v in norms.values())
    assert float(parts) == pytest.approx(float(total), rel=1e-5)
