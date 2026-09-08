"""Cover the machinery the four-variant visual ablation depends on.

The ablation is only worth running if its arms differ in exactly one thing.
These tests pin the pieces that make that true: the zero-width geometry mode,
the clip_dim guard that lets a control ignore features it was handed, the
union cache key that survives a reordering of the sample list, and the variant
table itself.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from relation_prediction.clip_extractor import CLIPExtractor  # noqa: E402
from relation_prediction.model import RelationMLP  # noqa: E402
from relation_prediction.vg_dataset import (  # noqa: E402
    GEO_DIM, GEO_DIM_EXT, GEO_DIM_NONE, GEO_MODES, VGRelationshipDataset,
    geo_extractor,
)

SUBJ = (10.0, 20.0, 60.0, 90.0)
OBJ = (100.0, 30.0, 180.0, 95.0)


# --------------------------------------------------------------------------
# geo_mode "none" - the visual-only control
# --------------------------------------------------------------------------

def test_geo_modes_are_exactly_the_three_supported():
    assert GEO_MODES == ("none", "basic", "ext")


@pytest.mark.parametrize("mode,dim", [
    ("none", GEO_DIM_NONE), ("basic", GEO_DIM), ("ext", GEO_DIM_EXT)])
def test_geo_extractor_widths(mode, dim):
    fn, d = geo_extractor(mode)
    assert d == dim
    assert len(fn(SUBJ, OBJ, 200.0, 100.0)) == dim


def test_geo_none_emits_nothing_for_any_box():
    fn, _ = geo_extractor("none")
    for boxes in [(SUBJ, OBJ), (OBJ, SUBJ), ((0, 0, 1, 1), (0, 0, 1, 1))]:
        assert fn(boxes[0], boxes[1], 640.0, 480.0) == []


def test_unknown_geo_mode_still_raises():
    with pytest.raises(ValueError):
        geo_extractor("nope")


def test_model_trains_and_runs_with_zero_width_geometry():
    """clip_only builds a model with geo_dim=0; a (B, 0) block must concatenate."""
    model = RelationMLP(num_labels=80, num_predicates=21, embed_dim=64,
                        hidden_dims=(32,), clip_dim=768, geo_dim=0)
    assert model.mlp[0].weight.shape[1] == 2 * 64 + 0 + 2 * 768
    out = model(torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.long),
                torch.zeros(4, 0), subj_feat=torch.randn(4, 768),
                obj_feat=torch.randn(4, 768))
    assert out.shape == (4, 21)
    assert torch.isfinite(out).all()


def test_geo_norm_with_zero_geometry_is_rejected():
    """A BatchNorm1d(0) would silently accept a configuration that means nothing."""
    with pytest.raises(ValueError):
        RelationMLP(num_labels=80, num_predicates=21, geo_dim=0, geo_norm=True)


# --------------------------------------------------------------------------
# the clip_dim guard - what makes the geometry control possible
# --------------------------------------------------------------------------

def test_a_clip_dim_zero_model_ignores_features_it_is_handed():
    """--visual-filter-only builds clip_dim=0 but the loader still yields feats.

    The evaluator and the training loop both pass subj_feat/obj_feat whenever
    the dataset produced them. A control model must drop them on the floor
    rather than try to concatenate 1536 columns its first Linear never
    allocated - that used to raise a shape error on the first batch.
    """
    model = RelationMLP(num_labels=80, num_predicates=21, embed_dim=64,
                        hidden_dims=(32,), clip_dim=0, geo_dim=GEO_DIM_EXT)
    model.eval()   # otherwise dropout, not the guard, decides the comparison
    subj = torch.zeros(4, dtype=torch.long)
    obj = torch.ones(4, dtype=torch.long)
    geo = torch.randn(4, GEO_DIM_EXT)

    without = model(subj, obj, geo)
    with_feats = model(subj, obj, geo,
                       subj_feat=torch.randn(4, 768), obj_feat=torch.randn(4, 768))
    assert torch.equal(without, with_feats), \
        "a clip_dim=0 model must produce identical logits with or without feats"


# --------------------------------------------------------------------------
# union cache keys - order-independent feature/label association
# --------------------------------------------------------------------------

def test_union_key_is_stable_and_ordered():
    a = CLIPExtractor.to_union_key(12, 34, 56)
    assert a == "12_union_34_56"
    assert CLIPExtractor.to_union_key("12", "34", "56") == a   # str and int agree
    # (subj, obj) and (obj, subj) span the same pixels but are different
    # samples; keeping them distinct keeps the key a name for THIS sample.
    assert CLIPExtractor.to_union_key(12, 56, 34) != a


def test_union_key_does_not_collide_across_images():
    """Ids concatenate with separators, so 1|2,3 cannot alias 12|3,3."""
    assert CLIPExtractor.to_union_key(1, 2, 3) != CLIPExtractor.to_union_key(12, 3, 3)
    assert CLIPExtractor.to_union_key(1, 23, 4) != CLIPExtractor.to_union_key(12, 3, 4)


def test_union_key_derived_from_sample_keys_matches_the_builder():
    """The dataset's lookup key must equal the key build_clip_cache.py writes."""
    subj_key, obj_key = "2340228_obj_1058529", "2340228_obj_1058534"
    derived = VGRelationshipDataset._union_key_for_sample(subj_key, obj_key)
    expected = CLIPExtractor.to_union_key(2340228, 1058529, 1058534)
    assert derived == expected


def test_union_key_returns_none_for_malformed_sample_keys():
    assert VGRelationshipDataset._union_key_for_sample("nonsense", "2_obj_3") is None
    assert VGRelationshipDataset._union_key_for_sample("2_obj_3", "nonsense") is None


def test_union_features_follow_the_pair_not_the_sample_position():
    """Regression for the positional union cache.

    Union features used to be a list indexed by sample position, so reordering
    self.samples silently paired every union embedding with a different label.
    Keying on the object ids makes the association survive any reordering.
    """
    pairs = [("7_obj_1", "7_obj_2"), ("7_obj_3", "7_obj_4"), ("9_obj_5", "9_obj_6")]
    keys = [VGRelationshipDataset._union_key_for_sample(a, b) for a, b in pairs]
    shuffled = [pairs[2], pairs[0], pairs[1]]
    shuffled_keys = [VGRelationshipDataset._union_key_for_sample(a, b)
                     for a, b in shuffled]
    assert set(keys) == set(shuffled_keys)
    assert shuffled_keys[0] == keys[2] and shuffled_keys[1] == keys[0]


# --------------------------------------------------------------------------
# checkpoint geo_mode inference
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dim,expected", [
    (GEO_DIM_EXT, "ext"), (GEO_DIM, "basic"), (0, "none")])
def test_geo_mode_inferred_from_width(dim, expected):
    """A visual-only checkpoint (geo_dim=0) must not be read back as "basic"."""
    from eval_gt_relations import _geo_mode_for_dim
    assert _geo_mode_for_dim(dim) == expected


def test_geo_mode_inference_refuses_a_nonsense_width():
    from eval_gt_relations import _geo_mode_for_dim
    with pytest.raises(ValueError):
        _geo_mode_for_dim(7)


# --------------------------------------------------------------------------
# the variant table
# --------------------------------------------------------------------------

def test_variants_are_the_four_the_ablation_needs():
    from run_visual_experiment import VARIANTS
    assert list(VARIANTS) == ["geometry", "geometry_clip",
                              "geometry_clip_union", "clip_only"]


def test_only_the_geometry_control_uses_visual_filter_only():
    """Every arm loads the CLIP cache; only the control hides it from the model."""
    from run_visual_experiment import VARIANTS
    assert VARIANTS["geometry"]["visual_filter_only"] is True
    for name in ("geometry_clip", "geometry_clip_union", "clip_only"):
        assert VARIANTS[name]["visual_filter_only"] is False


def test_every_arm_requires_visual_so_the_population_is_identical():
    """--require-visual on every arm is what fixes the sample population.

    Without it the geometry arm would keep pairs whose crops are missing and be
    scored on a larger test set than the CLIP arms - two variables at once.
    """
    from run_visual_experiment import COMMON_TRAIN_ARGS
    assert "--require-visual" in COMMON_TRAIN_ARGS
    assert "--use-visual" in COMMON_TRAIN_ARGS


def test_arms_differ_only_in_their_feature_configuration():
    from run_visual_experiment import VARIANTS
    feature_keys = {"geo_mode", "geo_norm", "union", "visual_filter_only"}
    for name, v in VARIANTS.items():
        assert feature_keys <= set(v), f"{name} is missing a feature key"
    # geometry and geometry_clip must be identical except for the control flag
    a, b = VARIANTS["geometry"], VARIANTS["geometry_clip"]
    assert a["geo_mode"] == b["geo_mode"] == "ext"
    assert a["geo_norm"] == b["geo_norm"] is True
    assert a["union"] == b["union"] is False
    assert a["visual_filter_only"] != b["visual_filter_only"]
    # geometry_clip and geometry_clip_union must differ only in union
    c = VARIANTS["geometry_clip_union"]
    assert c["union"] is True
    assert {k: c[k] for k in ("geo_mode", "geo_norm", "visual_filter_only")} == \
           {k: b[k] for k in ("geo_mode", "geo_norm", "visual_filter_only")}


@pytest.mark.parametrize("variant", ["geometry", "geometry_clip",
                                     "geometry_clip_union", "clip_only"])
def test_built_commands_carry_the_frozen_protocol(variant, tmp_path):
    """Every generated command must pin the split, scheme, loss and seed."""
    from run_visual_experiment import VARIANTS, build_commands

    train, evaluate = build_commands(
        variant, 43, tmp_path / "vg", tmp_path / "split.json",
        tmp_path / "cache.pt", tmp_path / "ck", tmp_path / "res", 25, 256)

    joined = " ".join(str(c) for c in train)
    assert "--split-manifest" in joined
    assert "--predicate-scheme v1" in joined
    assert "--loss ce" in joined
    assert "--seed 43" in joined
    assert "--require-visual" in joined
    assert f"--geo-mode {VARIANTS[variant]['geo_mode']}" in joined
    assert ("--geo-norm" in train) == VARIANTS[variant]["geo_norm"]
    assert ("--use-union" in train) == VARIANTS[variant]["union"]
    assert ("--visual-filter-only" in train) == VARIANTS[variant]["visual_filter_only"]

    ejoined = " ".join(str(c) for c in evaluate)
    assert "--split-manifest" in ejoined
    assert f"{variant}_seed43" in ejoined


def test_seeds_are_the_three_the_audit_called_for():
    from run_visual_experiment import SEEDS
    assert SEEDS == (42, 43, 44)
