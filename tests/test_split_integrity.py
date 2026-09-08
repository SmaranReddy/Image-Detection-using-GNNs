"""The frozen E0 split manifest must stay image-disjoint and stay frozen.

The published relation numbers are only meaningful if no image contributes
samples to more than one split, and if the manifest that training reads is the
same manifest evaluation reads.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MANIFEST = os.path.join(ROOT, "splits", "e0_image_split.json")
pytestmark = pytest.mark.skipif(
    not os.path.isfile(MANIFEST), reason="frozen split manifest not present")


@pytest.fixture(scope="module")
def manifest():
    with open(MANIFEST, encoding="utf-8") as f:
        return json.load(f)


def test_manifest_has_the_required_keys(manifest):
    for key in ("train_ids", "val_ids", "test_ids", "seed", "split_unit"):
        assert key in manifest, f"manifest missing {key!r}"
    assert manifest["split_unit"] == "image_id"


def test_splits_are_image_disjoint(manifest):
    tr = {int(i) for i in manifest["train_ids"]}
    va = {int(i) for i in manifest["val_ids"]}
    te = {int(i) for i in manifest["test_ids"]}
    assert tr & va == set(), f"{len(tr & va)} images in both train and val"
    assert tr & te == set(), f"{len(tr & te)} images in both train and test"
    assert va & te == set(), f"{len(va & te)} images in both val and test"


def test_no_duplicate_ids_within_a_split(manifest):
    for name in ("train_ids", "val_ids", "test_ids"):
        ids = [int(i) for i in manifest[name]]
        assert len(ids) == len(set(ids)), f"{name} contains duplicate image ids"


def test_split_sizes_match_the_recorded_counts(manifest):
    counts = manifest.get("counts")
    if not counts:
        pytest.skip("manifest carries no counts block")
    assert len(manifest["train_ids"]) == counts["n_images_train"]
    assert len(manifest["val_ids"]) == counts["n_images_val"]
    assert len(manifest["test_ids"]) == counts["n_images_test"]


def test_split_is_reproducible_from_its_recorded_seed(manifest):
    """Regenerating with the recorded seed must give back the same partition."""
    from eval_gt_relations import build_image_split

    all_ids = ([int(i) for i in manifest["train_ids"]]
               + [int(i) for i in manifest["val_ids"]]
               + [int(i) for i in manifest["test_ids"]])
    fr = manifest.get("fractions", {"train": 0.70, "val": 0.15})
    tr, va, te = build_image_split(all_ids, manifest["seed"], fr["train"], fr["val"])
    assert tr == [int(i) for i in manifest["train_ids"]]
    assert va == [int(i) for i in manifest["val_ids"]]
    assert te == [int(i) for i in manifest["test_ids"]]


def test_image_split_helper_partitions_without_loss_or_overlap():
    from eval_gt_relations import build_image_split

    ids = list(range(1000))
    tr, va, te = build_image_split(ids, seed=42, train_frac=0.7, val_frac=0.15)
    assert set(tr) | set(va) | set(te) == set(ids)
    assert len(tr) + len(va) + len(te) == len(ids)
    assert set(tr) & set(va) == set(tr) & set(te) == set(va) & set(te) == set()
    assert len(tr) == 700 and len(va) == 150 and len(te) == 150


def test_assert_disjoint_raises_on_overlap():
    from eval_gt_relations import assert_disjoint

    assert assert_disjoint([1, 2], [3], [4])["status"] == "PASS"
    with pytest.raises(SystemExit):
        assert_disjoint([1, 2], [2], [4])
