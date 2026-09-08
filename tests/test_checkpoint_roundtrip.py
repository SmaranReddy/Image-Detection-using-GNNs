"""Checkpoint metadata must describe the checkpoint.

Regression cover for: _save_checkpoint recorded hidden_dims as
[<second hidden>, <num_predicates>] instead of the real hidden widths, so a
(256, 128) model was saved as [128, 21] and rebuilding from model_config
raised a shape error. Nothing read it back, which is why it survived.

Also covers the bare-state-dict loaders, where eval_gt_relations.py hardcoded
geo_dim=5 and so reconstructed a 19-dim geometry checkpoint as
geo_dim=5 + clip_dim=7 — a model that loads cleanly and computes nonsense.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relation_prediction.model import RelationMLP  # noqa: E402
from relation_prediction.predict import _infer_clip_dim, _infer_hidden_dims  # noqa: E402
from relation_prediction.vg_dataset import GEO_DIM, GEO_DIM_EXT  # noqa: E402

CASES = [
    dict(hidden_dims=(256, 128), geo_dim=GEO_DIM_EXT, geo_norm=True, clip_dim=0),
    dict(hidden_dims=(256, 128), geo_dim=GEO_DIM, geo_norm=False, clip_dim=0),
    dict(hidden_dims=(512,), geo_dim=GEO_DIM_EXT, geo_norm=True, clip_dim=0),
    dict(hidden_dims=(1024, 512, 256), geo_dim=GEO_DIM_EXT, geo_norm=True, clip_dim=0),
    dict(hidden_dims=(256, 128), geo_dim=GEO_DIM_EXT, geo_norm=True, clip_dim=768),
]


def build(case):
    return RelationMLP(num_labels=80, num_predicates=21, embed_dim=64,
                       hidden_dims=case["hidden_dims"], geo_dim=case["geo_dim"],
                       geo_norm=case["geo_norm"], clip_dim=case["clip_dim"])


def save_config(model):
    """The exact hidden_dims derivation used by train_full_visual_semantic."""
    state = model.state_dict()
    hdims, idx = [], 0
    while f"mlp.{idx}.weight" in state:
        hdims.append(int(state[f"mlp.{idx}.weight"].shape[0]))
        idx += 3
    return hdims[:-1]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"h{c['hidden_dims']}g{c['geo_dim']}c{c['clip_dim']}")
def test_saved_hidden_dims_match_the_weights(case):
    model = build(case)
    assert tuple(save_config(model)) == tuple(case["hidden_dims"])


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"h{c['hidden_dims']}g{c['geo_dim']}c{c['clip_dim']}")
def test_model_rebuilds_from_its_own_saved_config(case, tmp_path):
    """Round-trip: build -> save -> rebuild from model_config -> load_state_dict."""
    model = build(case)
    config = {
        "model_type": "mlp", "num_labels": 80, "num_predicates": 21,
        "embed_dim": 64, "clip_dim": model.clip_dim, "pose_dim": model.pose_dim,
        "union_dim": model.union_dim, "hidden_dims": save_config(model),
        "geo_dim": model.geo_dim, "geo_norm": model.geo_norm is not None,
    }
    path = tmp_path / "relation_mlp.pt"
    torch.save({"model_state_dict": model.state_dict(), "model_config": config}, path)

    raw = torch.load(path, map_location="cpu", weights_only=True)
    cfg = raw["model_config"]
    rebuilt = RelationMLP(
        num_labels=cfg["num_labels"], num_predicates=cfg["num_predicates"],
        embed_dim=cfg["embed_dim"], hidden_dims=tuple(cfg["hidden_dims"]),
        clip_dim=cfg["clip_dim"], pose_dim=cfg["pose_dim"],
        union_dim=cfg["union_dim"], geo_dim=cfg["geo_dim"], geo_norm=cfg["geo_norm"],
    )
    rebuilt.load_state_dict(raw["model_state_dict"])   # raised before the fix

    model.eval(); rebuilt.eval()
    kw = {}
    if cfg["clip_dim"]:
        kw = dict(subj_feat=torch.zeros(4, cfg["clip_dim"]),
                  obj_feat=torch.zeros(4, cfg["clip_dim"]))
    args = (torch.randint(0, 80, (4,)), torch.randint(0, 80, (4,)),
            torch.randn(4, cfg["geo_dim"]))
    with torch.no_grad():
        assert torch.allclose(model(*args, **kw), rebuilt(*args, **kw))


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"h{c['hidden_dims']}g{c['geo_dim']}c{c['clip_dim']}")
def test_bare_state_dict_inference_recovers_the_architecture(case):
    """A config-less checkpoint must still be reconstructed correctly.

    geo_norm's BatchNorm buffer IS the geometry width; without that the
    remaining input columns get misattributed to a phantom CLIP block.
    """
    state = build(case).state_dict()
    assert tuple(_infer_hidden_dims(state)) == tuple(case["hidden_dims"])

    geo_norm = "geo_norm.weight" in state
    geo_dim = int(state["geo_norm.weight"].shape[0]) if geo_norm else GEO_DIM
    assert geo_dim == case["geo_dim"] or not geo_norm
    assert _infer_clip_dim(state, 64, geo_dim) == case["clip_dim"]


def test_wrong_geo_dim_fabricates_a_phantom_clip_block():
    """Pin the exact failure the eval loader used to hit, so it cannot return."""
    state = build(dict(hidden_dims=(256, 128), geo_dim=GEO_DIM_EXT,
                       geo_norm=True, clip_dim=0)).state_dict()
    assert _infer_clip_dim(state, 64, GEO_DIM_EXT) == 0        # correct width
    assert _infer_clip_dim(state, 64, GEO_DIM) == 7            # legacy hardcoded 5
