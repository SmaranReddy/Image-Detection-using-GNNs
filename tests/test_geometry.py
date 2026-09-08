"""Geometry-descriptor tests on hand-constructed boxes.

Every value in the 19-dim descriptor is checked against a closed-form
expectation, so a width/height swap, an x/y swap, a sign flip in dx/dy, a
symmetric-instead-of-asymmetric containment term, or a wrong normalising
denominator all fail here rather than silently degrading a training run.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relation_prediction.vg_dataset import (  # noqa: E402
    GEO_DIM,
    GEO_DIM_EXT,
    GEO_EXT_FEATURE_NAMES,
    compute_iou,
    extract_geo_features,
    extract_geo_features_ext,
    geo_extractor,
)

IMG_W, IMG_H = 200.0, 100.0


def feat(subj, obj, w=IMG_W, h=IMG_H):
    return dict(zip(GEO_EXT_FEATURE_NAMES, extract_geo_features_ext(subj, obj, w, h)))


def test_dimensions_and_names_agree():
    assert len(GEO_EXT_FEATURE_NAMES) == GEO_DIM_EXT == 19
    assert len(extract_geo_features_ext((0, 0, 10, 10), (5, 5, 15, 15), IMG_W, IMG_H)) == GEO_DIM_EXT
    assert len(extract_geo_features((0, 0, 10, 10), (5, 5, 15, 15), IMG_W, IMG_H)) == GEO_DIM == 5


def test_basic_is_a_strict_prefix_of_ext():
    """The two descriptors must stay comparable: basic == ext[:5]."""
    for subj, obj in (((0, 0, 20, 20), (40, 60, 90, 100)),
                      ((10, 10, 12, 90), (0, 0, 199, 99)),
                      ((5, 5, 105, 55), (5, 5, 105, 55))):
        assert extract_geo_features(subj, obj, IMG_W, IMG_H) == pytest.approx(
            extract_geo_features_ext(subj, obj, IMG_W, IMG_H)[:5])


def test_geo_extractor_dispatch():
    assert geo_extractor("basic") == (extract_geo_features, GEO_DIM)
    assert geo_extractor("ext") == (extract_geo_features_ext, GEO_DIM_EXT)
    with pytest.raises(ValueError):
        geo_extractor("nineteen")


# --------------------------------------------------------------------------
# dx / dy: sign, axis and normaliser
# --------------------------------------------------------------------------

def test_dx_dy_sign_and_axis():
    """Object to the RIGHT of and BELOW the subject => dx > 0 and dy > 0.

    Also pins the axis: dx is normalised by image WIDTH and dy by image
    HEIGHT. With a non-square image an x/y or w/h swap changes the value.
    """
    subj = (0.0, 0.0, 20.0, 20.0)          # centre (10, 10)
    obj = (100.0, 40.0, 120.0, 60.0)       # centre (110, 50)
    f = feat(subj, obj)
    assert f["dx_img"] == pytest.approx((110 - 10) / IMG_W)   # +0.5
    assert f["dy_img"] == pytest.approx((50 - 10) / IMG_H)    # +0.4
    assert f["dx_img"] > 0 and f["dy_img"] > 0

    back = feat(obj, subj)                                   # swap roles
    assert back["dx_img"] == pytest.approx(-f["dx_img"])
    assert back["dy_img"] == pytest.approx(-f["dy_img"])


def test_subject_relative_offsets_use_subject_size():
    """dx_subj / dy_subj divide by the SUBJECT box, not the image."""
    subj = (0.0, 0.0, 10.0, 40.0)          # w=10 h=40, centre (5, 20)
    obj = (20.0, 20.0, 30.0, 60.0)         # centre (25, 40)
    f = feat(subj, obj)
    assert f["dx_subj"] == pytest.approx((25 - 5) / 10.0)     # 2.0
    assert f["dy_subj"] == pytest.approx((40 - 20) / 40.0)    # 0.5


def test_centre_distance_is_hypot_of_image_normalised_offsets():
    f = feat((0.0, 0.0, 20.0, 20.0), (100.0, 40.0, 120.0, 60.0))
    assert f["centre_distance"] == pytest.approx(math.hypot(f["dx_img"], f["dy_img"]))
    assert f["centre_distance"] >= 0.0


# --------------------------------------------------------------------------
# ratios
# --------------------------------------------------------------------------

def test_log_ratios_are_zero_for_identical_shapes_and_antisymmetric():
    subj = (0.0, 0.0, 30.0, 10.0)
    obj = (50.0, 50.0, 80.0, 60.0)                     # same 30x10 shape
    f = feat(subj, obj)
    assert f["log_w_ratio"] == pytest.approx(0.0)
    assert f["log_h_ratio"] == pytest.approx(0.0)
    assert f["log_area_ratio"] == pytest.approx(0.0)

    big = (0.0, 0.0, 60.0, 40.0)                       # 2x wide, 4x tall
    f2 = feat(subj, big)
    assert f2["log_w_ratio"] == pytest.approx(math.log(2.0))
    assert f2["log_h_ratio"] == pytest.approx(math.log(4.0))
    assert f2["log_area_ratio"] == pytest.approx(math.log(8.0))
    # reversing subject/object negates every log ratio
    f3 = feat(big, subj)
    assert f3["log_w_ratio"] == pytest.approx(-f2["log_w_ratio"])
    assert f3["log_area_ratio"] == pytest.approx(-f2["log_area_ratio"])


def test_aspect_ratios_are_per_box_and_orientation_sensitive():
    wide = (0.0, 0.0, 40.0, 10.0)      # w/h = 4
    tall = (0.0, 0.0, 10.0, 40.0)      # w/h = 1/4
    f = feat(wide, tall)
    assert f["log_subj_aspect"] == pytest.approx(math.log(4.0))
    assert f["log_obj_aspect"] == pytest.approx(math.log(0.25))


def test_relative_scale_is_sqrt_area_fraction_and_image_normalised():
    subj = (0.0, 0.0, 100.0, 50.0)     # half the 200x100 image by each side
    f = feat(subj, (0.0, 0.0, 200.0, 100.0))
    assert f["subj_rel_scale"] == pytest.approx(math.sqrt((100 * 50) / (IMG_W * IMG_H)))
    assert f["obj_rel_scale"] == pytest.approx(1.0)     # object fills the frame
    assert 0.0 <= f["subj_rel_scale"] <= 1.0


def test_vertical_position_is_normalised_centre_y():
    subj = (0.0, 0.0, 10.0, 50.0)      # centre y = 25 of 100
    obj = (0.0, 50.0, 10.0, 100.0)     # centre y = 75 of 100
    f = feat(subj, obj)
    assert f["subj_cy_img"] == pytest.approx(0.25)
    assert f["obj_cy_img"] == pytest.approx(0.75)


# --------------------------------------------------------------------------
# overlap: IoU is symmetric, containment is not
# --------------------------------------------------------------------------

def test_iou_matches_helper_and_is_symmetric():
    a, b = (0.0, 0.0, 20.0, 20.0), (10.0, 10.0, 30.0, 30.0)
    inter = 10.0 * 10.0
    expected = inter / (400.0 + 400.0 - inter)
    assert compute_iou(a, b) == pytest.approx(expected)
    assert compute_iou(b, a) == pytest.approx(expected)
    assert feat(a, b)["iou"] == pytest.approx(expected)
    assert feat(b, a)["iou"] == pytest.approx(expected)


def test_disjoint_boxes_have_zero_overlap_terms():
    f = feat((0.0, 0.0, 10.0, 10.0), (150.0, 80.0, 190.0, 95.0))
    assert f["iou"] == 0.0
    assert f["inter_over_subj"] == 0.0
    assert f["inter_over_obj"] == 0.0


def test_containment_is_asymmetric_which_iou_cannot_express():
    """"A inside B" and "B inside A" must be distinguishable.

    This is the whole point of carrying both containment ratios: IoU is
    identical in both directions, so a descriptor with IoU alone cannot tell
    "cup in bowl" from "bowl around cup".
    """
    small = (10.0, 10.0, 20.0, 20.0)               # area 100, fully inside
    large = (0.0, 0.0, 100.0, 50.0)                # area 5000
    f = feat(small, large)                         # subject contained in object
    assert f["inter_over_subj"] == pytest.approx(1.0)
    assert f["inter_over_obj"] == pytest.approx(100.0 / 5000.0)

    g = feat(large, small)                         # object contained in subject
    assert g["inter_over_subj"] == pytest.approx(100.0 / 5000.0)
    assert g["inter_over_obj"] == pytest.approx(1.0)

    assert f["iou"] == pytest.approx(g["iou"])     # IoU cannot separate them
    assert f["inter_over_subj"] != pytest.approx(g["inter_over_subj"])


def test_identical_boxes_are_fully_contained_both_ways():
    box = (5.0, 5.0, 55.0, 45.0)
    f = feat(box, box)
    assert f["iou"] == pytest.approx(1.0)
    assert f["inter_over_subj"] == pytest.approx(1.0)
    assert f["inter_over_obj"] == pytest.approx(1.0)
    assert f["dx_img"] == pytest.approx(0.0)
    assert f["dy_img"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# signed vertical gaps (above / below / on / under evidence)
# --------------------------------------------------------------------------

def test_signed_vertical_gaps_have_the_documented_orientation():
    """gap_obj_below_subj = (obj_top - subj_bottom) / H.

    Positive when the object sits strictly BELOW the subject; the companion
    term is positive when the subject sits strictly below the object. A sign
    flip or a swap of the two would invert every above/below cue.
    """
    upper = (0.0, 0.0, 10.0, 20.0)       # y 0..20
    lower = (0.0, 60.0, 10.0, 80.0)      # y 60..80
    f = feat(upper, lower)               # subject above, object below
    assert f["gap_obj_below_subj"] == pytest.approx((60 - 20) / IMG_H)
    assert f["gap_obj_below_subj"] > 0
    assert f["gap_subj_below_obj"] == pytest.approx((0 - 80) / IMG_H)
    assert f["gap_subj_below_obj"] < 0

    g = feat(lower, upper)               # roles reversed
    assert g["gap_subj_below_obj"] == pytest.approx(f["gap_obj_below_subj"])
    assert g["gap_obj_below_subj"] == pytest.approx(f["gap_subj_below_obj"])


def test_touching_boxes_give_a_zero_gap():
    f = feat((0.0, 0.0, 10.0, 30.0), (0.0, 30.0, 10.0, 60.0))
    assert f["gap_obj_below_subj"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# degenerate input must not produce NaN / inf
# --------------------------------------------------------------------------

@pytest.mark.parametrize("subj,obj,w,h", [
    ((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), 200.0, 100.0),   # zero-area
    ((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0), 0.0, 0.0),   # zero image
    ((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0), 1.0, 1.0),   # unit image
    ((-50.0, -50.0, 10.0, 10.0), (190.0, 90.0, 260.0, 160.0), 200.0, 100.0),  # off-image
])
def test_no_nan_or_inf_on_degenerate_geometry(subj, obj, w, h):
    for fn in (extract_geo_features, extract_geo_features_ext):
        for name, v in zip(GEO_EXT_FEATURE_NAMES, fn(subj, obj, w, h)):
            assert math.isfinite(v), f"{fn.__name__}/{name} produced {v}"


def test_features_are_scale_invariant_where_they_should_be():
    """Doubling the image and every box leaves the normalised terms alone."""
    subj, obj = (10.0, 20.0, 40.0, 60.0), (80.0, 30.0, 120.0, 90.0)
    a = feat(subj, obj, IMG_W, IMG_H)
    b = feat(tuple(2 * c for c in subj), tuple(2 * c for c in obj), 2 * IMG_W, 2 * IMG_H)
    for name in GEO_EXT_FEATURE_NAMES:
        assert a[name] == pytest.approx(b[name]), f"{name} is not scale-invariant"
