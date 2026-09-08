"""POPE must not reward a system for writing longer captions.

Regression cover for two bugs in utils/pope.py:
  * `num_positives` was bound only inside `if negative_pool:` but read
    unconditionally at the return, so an exhausted pool raised NameError;
  * the negative-probe count was `max(10, len(mentioned))`, and every negative
    probe is a true negative by construction, so a caption that names more
    objects earned more free TNs and a higher pope_accuracy at identical
    correctness. The grounded system's captions are longer than the baseline's
    by construction (the injected relation prefix adds mentions), so the bias
    ran in the direction of the effect being claimed.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

nltk = pytest.importorskip("nltk", reason="utils.metrics.detect_coco_objects needs NLTK/WordNet")

from utils.metrics import COCO_80  # noqa: E402
from utils.pope import NUM_NEGATIVE_PROBES, compute_pope  # noqa: E402


def test_negative_probe_count_is_independent_of_the_caption():
    short = compute_pope("a photo of a person", {"person"})
    long = compute_pope(
        "a photo of a person riding a bicycle near a car and a dog and a bench",
        {"person", "bicycle", "car", "dog", "bench"})
    assert short["pope_num_negative_probes"] == long["pope_num_negative_probes"]
    assert short["pope_tn"] == long["pope_tn"] == NUM_NEGATIVE_PROBES


def test_two_perfect_captions_of_different_length_score_the_same_accuracy():
    short = compute_pope("a photo of a person", {"person"})
    long = compute_pope(
        "a photo of a person riding a bicycle near a car and a dog and a bench",
        {"person", "bicycle", "car", "dog", "bench"})
    for m in ("pope_precision", "pope_recall", "pope_f1", "pope_accuracy"):
        assert short[m] == pytest.approx(1.0)
        assert long[m] == pytest.approx(1.0), f"{m} differs purely by caption length"


def test_exhausted_negative_pool_does_not_crash():
    """mentioned u gt covering all of COCO-80 leaves no negatives to sample."""
    caption = " ".join(sorted(COCO_80))
    result = compute_pope(caption, set(COCO_80))          # raised NameError before
    assert result["pope_num_negative_probes"] == 0
    assert result["pope_tn"] == 0
    assert result["pope_num_positive_probes"] == len(result["pope_hallucinated_objects"]) + result["pope_tp"]


def test_hallucination_is_penalised():
    clean = compute_pope("a photo of a person", {"person"})
    halluc = compute_pope("a photo of a person and a giraffe", {"person"})
    assert halluc["pope_fp"] == 1
    assert "giraffe" in halluc["pope_hallucinated_objects"]
    assert halluc["pope_precision"] < clean["pope_precision"]


def test_missed_object_lowers_recall_not_precision():
    r = compute_pope("a photo of a person", {"person", "bicycle"})
    assert r["pope_fn"] == 1 and "bicycle" in r["pope_missed_objects"]
    assert r["pope_precision"] == pytest.approx(1.0)
    assert r["pope_recall"] == pytest.approx(0.5)


def test_empty_caption_is_handled():
    r = compute_pope("", {"person"})
    assert r["pope_tp"] == 0 and r["pope_fp"] == 0 and r["pope_fn"] == 1
    assert r["pope_num_positive_probes"] == 0


def test_deterministic_across_calls():
    a = compute_pope("a photo of a person on a bench", {"person", "bench"})
    b = compute_pope("a photo of a person on a bench", {"person", "bench"})
    assert a == b
