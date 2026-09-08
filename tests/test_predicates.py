"""Predicate normalisation and label-space tests.

Regression cover for the vocabulary bugs found in the audit:
  * "carried by" was mapped to "carrying", labelling the pair backwards;
  * "next to" sat in the label space but no annotation could ever carry it;
  * scheme "v1" must stay byte-identical so the frozen E0 numbers reproduce.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relation_prediction.vg_dataset import (  # noqa: E402
    ALLOWED_PREDICATES,
    PASSIVE_PREDICATES,
    PREDICATE_MAP,
    PREDICATE_SCHEMES,
    normalize_label,
    normalize_predicate,
    normalize_predicate_v2,
    predicate_normalizer,
    reachable_predicates,
)


# --------------------------------------------------------------------------
# direction safety
# --------------------------------------------------------------------------

def test_no_passive_survives_in_the_predicate_map():
    """A passive can only be normalised by swapping subject and object.

    PREDICATE_MAP is a pure string rewrite applied while subject and object
    stay put, so any "... by" entry silently produces a reversed training
    pair. "carried by" -> "carrying" used to do exactly that.
    """
    offenders = [k for k in PREDICATE_MAP if k.endswith(" by")]
    assert offenders == [], f"direction-inverting entries in PREDICATE_MAP: {offenders}"


@pytest.mark.parametrize("passive", sorted(PASSIVE_PREDICATES))
def test_v2_rejects_passives_instead_of_reversing_the_pair(passive):
    assert normalize_predicate_v2(passive) is None


def test_v1_keeps_its_known_passive_defect_on_purpose():
    """v1 is frozen, defect included, so the published corpus stays at 68,900.

    "carried by" -> "carrying" is wrong (it labels 29 pairs backwards) but
    removing it from v1 would move every frozen number in results/. The fix
    lives in v2; this test exists so the defect cannot be "cleaned up" by
    accident without someone confronting the reproducibility cost.
    """
    assert normalize_predicate("carried by") == "carrying"
    assert normalize_predicate_v2("carried by") is None


# --------------------------------------------------------------------------
# label space honesty
# --------------------------------------------------------------------------

def test_next_to_is_unreachable_and_therefore_not_a_v2_class():
    """PREDICATE_MAP rewrites "next to" -> "near" before the allowlist check.

    So "next to" can never be emitted, yet it is in ALLOWED_PREDICATES and was
    seeded into the label vocabulary — a permanently zero-support output class
    (it shows up as "zero_support_predicates": ["next to"] in every result
    file). reachable_predicates() must exclude it.
    """
    assert "next to" in ALLOWED_PREDICATES
    assert normalize_predicate("next to") == "near"
    assert "next to" not in reachable_predicates("v1")
    assert "next to" not in reachable_predicates("v2")


@pytest.mark.parametrize("scheme", PREDICATE_SCHEMES)
def test_every_reachable_predicate_is_a_fixed_point(scheme):
    fn = predicate_normalizer(scheme)
    for p in reachable_predicates(scheme):
        assert fn(p) == p
        assert p in ALLOWED_PREDICATES


@pytest.mark.parametrize("scheme", PREDICATE_SCHEMES)
def test_normalisers_never_invent_a_class(scheme):
    fn = predicate_normalizer(scheme)
    for surface in ("on a", "holds", "watching", "gibberish", "", "   ", "with",
                    "has", "of", "and", "carried by", None):
        out = fn(surface) if surface is not None else fn("")
        assert out is None or out in ALLOWED_PREDICATES


def test_predicate_normalizer_rejects_unknown_schemes():
    with pytest.raises(ValueError):
        predicate_normalizer("v3")


# --------------------------------------------------------------------------
# scheme v1 must not drift (frozen E0 reproducibility)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("surface,expected", [
    ("on", "on"), ("On", "on"), ("  on  ", "on"),
    ("on top of", "on"), ("lying on", "on"), ("resting on", "on"),
    ("next to", "near"), ("beside", "near"), ("close to", "near"),
    ("underneath", "under"), ("below", "under"),
    ("riding on", "riding"), ("mounted on", "riding"),
    ("holding in", "holding"), ("grasping", "holding"), ("gripping", "holding"),
    ("carrying in", "carrying"),
    # v1 deliberately does NOT recover these; that is the behaviour E0 was measured under
    ("on a", None), ("holds", None), ("watching", None), ("laying on", None),
    ("riding a", None), ("with", None), ("has", None), ("", None),
])
def test_v1_behaviour_is_frozen(surface, expected):
    assert normalize_predicate(surface) == expected


# --------------------------------------------------------------------------
# scheme v2: recovers surface variants onto the SAME classes
# --------------------------------------------------------------------------

@pytest.mark.parametrize("surface,expected", [
    # trailing determiner
    ("on a", "on"), ("in a", "in"), ("riding a", "riding"),
    ("wearing a", "wearing"), ("holding an", "holding"),
    ("carrying a", "carrying"), ("sitting on a", "sitting on"),
    ("standing on the", "standing on"),
    # inflection
    ("holds", "holding"), ("rides", "riding"), ("wears", "wearing"),
    ("carries", "carrying"), ("sits on", "sitting on"),
    ("stands on", "standing on"), ("watches", "looking at"),
    ("looks at", "looking at"),
    # leading copula
    ("is on", "on"), ("are on", "on"), ("is sitting on", "sitting on"),
    # paraphrase
    ("laying on", "on"), ("sleeping on", "on"), ("sitting on top of", "sitting on"),
    ("inside of", "inside"), ("beneath", "under"),
    ("standing next to", "near"), ("standing behind", "behind"),
    ("standing in front of", "in front of"),
    # chained rewrite: surface form -> "next to" -> "near"
    ("next to a", "near"),
    # whitespace
    ("  riding   a  ", "riding"),
    # still correctly rejected: not relations
    ("with", None), ("has", None), ("of", None), ("and", None), ("by", None),
    ("a", None), ("", None), ("   ", None),
    # still correctly rejected: real predicates outside the 19-class space
    ("eating", None), ("flying", None), ("playing with", None), ("cutting", None),
])
def test_v2_surface_recovery(surface, expected):
    assert normalize_predicate_v2(surface) == expected


def test_v2_is_a_superset_of_v1_on_every_form_v1_accepts():
    """v2 must never lose a sample v1 kept, and never relabel one differently."""
    forms = sorted(ALLOWED_PREDICATES) + sorted(PREDICATE_MAP)
    for f in forms:
        a = normalize_predicate(f)
        if a is not None:
            assert normalize_predicate_v2(f) == a, f"v2 disagrees with v1 on {f!r}"


def test_v2_is_idempotent():
    for f in ("on a", "holds", "standing next to", "is sitting on top of"):
        once = normalize_predicate_v2(f)
        assert normalize_predicate_v2(once) == once


# --------------------------------------------------------------------------
# object labels
# --------------------------------------------------------------------------

@pytest.mark.parametrize("surface,expected", [
    ("man", "person"), ("WOMAN", "person"), (" boy ", "person"),
    ("bike", "bicycle"), ("sofa", "couch"), ("television", "tv"),
    ("plant", "potted plant"),
    ("cars", "car"), ("dogs", "dog"),            # plural stripping
    ("person", "person"), ("dining table", "dining table"),
    ("wombat", "UNK"), ("", "UNK"),
])
def test_normalize_label(surface, expected):
    assert normalize_label(surface) == expected


def test_label_normalisation_is_idempotent():
    for s in ("man", "bikes", "television", "dining table", "wombat"):
        once = normalize_label(s)
        assert normalize_label(once) == once or once == "UNK"
