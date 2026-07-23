"""
Inference for the learned relation predictor.

Two public entry points:

1.  predict_relation(subject, obj_label, box1, box2, ...)
        Returns the most likely predicate string for a single pair,
        or None when the model is not confident enough.

2.  infer_relationships_learned(detections, ...)
        Drop-in replacement for utils.causal_caption.infer_relationships().
        Accepts the same detection-dict list and returns the same
        List[Tuple[str, str, str]] format.

Loading
-------
The module loads the trained checkpoint lazily on first call.
The model can be trained in two modes:
    - Geometry-only (clip_dim=0): uses only label embeddings + geometry
    - Visual-semantic (clip_dim>0): adds CLIP embeddings from image crops

When using a visual-semantic model, you MUST pass the image to inference.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from .model import RelationMLP
from .relation_transformer import RelationTransformer
from .vg_dataset import (
    ALLOWED_PREDICATES,
    Vocab,
    extract_geo_features,
    normalize_label,
    GEO_DIM,
    POSE_FEATURE_DIM,
    POSE_OBJECT_FEATURE_DIM,
    UNION_FEATURE_DIM,
)
from .clip_extractor import CLIPExtractor, CLIP_DIM
from .pose_extractor import PoseExtractor, POSE_FEATURE_DIM, POSE_OBJECT_FEATURE_DIM

from utils.logger_utils import debug_print

from .vg_dataset import load_predicate_priors


# ---------------------------------------------------------------------------
# Configuration — Logit adjustment (Menon et al. ICLR 2021)
# ---------------------------------------------------------------------------

ENABLE_LOGIT_ADJUSTMENT: bool = True
LOGIT_ADJUST_TAU: float = 0.25

# Per-predicate tau multipliers: semantic predicates get stronger adjustment,
# common spatial predicates get minimal adjustment.
LOGIT_ADJUST_MULTIPLIERS: Dict[str, float] = {
    # Semantic (full adjustment)
    "riding": 1.0, "wearing": 1.0, "holding": 1.0, "carrying": 1.0,
    "sitting on": 1.0, "looking at": 1.0,
    # Common spatial (minimal adjustment)
    "on": 0.3, "in": 0.3, "near": 0.3, "behind": 0.3, "in front of": 0.3,
}
"""
Multiplier legend:
  1.0 — full tau (semantic predicates)
  0.3 — reduced tau (common spatial predicates — slight nudge only)
  absent — 0.5 tau (other predicates)
"""
_DEFAULT_MULTIPLIER: float = 0.5

MAX_ADJUSTMENT: float = 1.5

PREDICATE_PRIORS_PATH: str = os.path.join(
    os.environ.get("REL_CKPT_DIR", "./checkpoints"), "predicate_priors.json"
)

_predicate_priors: Dict[str, float] = {}
_predicate_priors_loaded: bool = False


def _ensure_predicate_priors() -> None:
    global _predicate_priors, _predicate_priors_loaded
    if _predicate_priors_loaded:
        return
    _predicate_priors = load_predicate_priors(PREDICATE_PRIORS_PATH)
    _predicate_priors_loaded = True
    if _predicate_priors:
        debug_print(f"[logit_adjust] Loaded {len(_predicate_priors)} predicate priors from {PREDICATE_PRIORS_PATH}")
        for pred, prior in sorted(_predicate_priors.items(), key=lambda x: -x[1]):
            debug_print(f"  {pred}: {prior:.6f}")


def apply_logit_adjustment(
    logits: torch.Tensor,
    pred_tokens: List[str],
    tau: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, Dict]]:
    """Apply logit adjustment using predicate frequency priors.

    adjusted_logit = raw_logit - τ ⋅ multiplier(p) ⋅ log(prior)

    adjusted_logit = raw_logit - τ * log(prior),
    with per-predicate multipliers and magnitude clamping.

    The standard Menon et al. post-hoc formula is:
        adjusted = raw - τ * log(prior)

    Since log(prior) < 0, this boosts rare predicates (large |log(prior)|)
    and minimally adjusts common ones (small |log(prior)|).

    To prevent over-boosting, we:
      - Apply per-predicate tau multipliers (semantic > spatial).
      - Clamp the raw adjustment to ±MAX_ADJUSTMENT.

    See: Menon et al. (ICLR 2021).

    Args:
        logits: Raw logits tensor (num_predicates,).
        pred_tokens: List of predicate name strings.
        tau: Adjustment strength (default: LOGIT_ADJUST_TAU).

    Returns:
        (adjusted_logits, debug_info)
        debug_info maps predicate name → {prior, adjustment, raw, adjusted}.
    """
    if tau is None:
        tau = LOGIT_ADJUST_TAU
    if not ENABLE_LOGIT_ADJUSTMENT or tau <= 0:
        return logits, {}

    _ensure_predicate_priors()
    if not _predicate_priors:
        return logits, {}

    adjusted = logits.clone()
    debug_info: Dict[str, Dict] = {}

    for i, pred_name in enumerate(pred_tokens):
        if pred_name in (Vocab.PAD, Vocab.UNK):
            continue
        prior = _predicate_priors.get(pred_name, 0.0)
        if prior <= 0:
            continue

        # Per-predicate tau multiplier
        mult = LOGIT_ADJUST_MULTIPLIERS.get(pred_name, _DEFAULT_MULTIPLIER)

        # adjustment = -τ * multiplier * log(prior), positive for prior < 1
        # Rare predicates (large |log(prior)|) get a larger boost.
        log_prior = math.log(max(prior, 1e-10))
        adjustment = -tau * mult * log_prior  # positive for prior < 1

        # Clamp to prevent extreme boosts
        adjustment = max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, adjustment))
        adjusted[i] = logits[i] + adjustment
        debug_info[pred_name] = {
            "prior": round(prior, 6),
            "mult": round(mult, 2),
            "raw": round(logits[i].item(), 4),
            "adjusted": round(adjusted[i].item(), 4),
            "adjustment": round(adjustment, 4),
        }

    return adjusted, debug_info


def _print_logit_adjustment_debug(debug_info: Dict[str, Dict]) -> None:
    """Print logit adjustment debug output."""
    if not debug_info:
        return
    print("[logit_adjust] Adjusting logits:")
    for pred, info in sorted(debug_info.items(), key=lambda x: -x[1]["raw"]):
        mult = info.get("mult", 1.0)
        clip_tag = " ⚡" if abs(info["adjustment"]) >= MAX_ADJUSTMENT * 0.99 else ""
        print(f"  pred={pred}  (mult={mult:.2f}){clip_tag}")
        print(f"    prior={info['prior']:.4f}")
        print(f"    adjustment={info['adjustment']:.4f}")
        arrow = " ↑" if info['adjusted'] > info['raw'] else " ↓"
        print(f"    raw={info['raw']:.4f} → adjusted={info['adjusted']:.4f}{arrow}")


# ---------------------------------------------------------------------------
# Step 1 — Object type categories (lightweight semantic type groups)
# ---------------------------------------------------------------------------

ANIMATE: frozenset = frozenset({
    "person", "dog", "horse", "cat", "bird",
    "cow", "sheep", "elephant", "bear", "zebra", "giraffe",
})

WEARABLE: frozenset = frozenset({
    "backpack", "handbag", "tie", "suitcase",
})

RIDEABLE: frozenset = frozenset({
    "bicycle", "horse", "motorcycle", "skateboard", "surfboard",
    "skis", "snowboard",
})

HANDHELD: frozenset = frozenset({
    "cell phone", "umbrella", "bottle", "cup", "book",
    "fork", "knife", "spoon", "bowl", "frisbee",
    "kite", "baseball bat", "baseball glove", "tennis racket",
    "remote", "keyboard", "mouse", "scissors", "toothbrush",
    "wine glass", "hot dog", "apple", "banana", "orange",
    "sandwich", "donut", "cake", "pizza", "carrot", "broccoli",
})

FURNITURE: frozenset = frozenset({
    "chair", "couch", "bench", "bed", "dining table", "toilet",
})

VEHICLE: frozenset = frozenset({
    "car", "truck", "bus", "train", "boat", "airplane",
})

SUPPORT_SURFACE: frozenset = frozenset({
    "chair", "bench", "couch", "bed", "dining table", "toilet",
})

_OBJECT_CATEGORIES: Dict[str, frozenset] = {
    "animate": ANIMATE,
    "wearable": WEARABLE,
    "rideable": RIDEABLE,
    "handheld": HANDHELD,
    "furniture": FURNITURE,
    "vehicle": VEHICLE,
    "support_surface": SUPPORT_SURFACE,
}

# ── PART 1: Commonsense affordance compatibility groups ──────────────────
# These define which objects are physically plausible targets for specific
# interaction types. Used to apply confidence penalties to implausible
# relations WITHOUT fully suppressing unusual but possible ones.
# ────────────────────────────────────────────────────────────────────────

SITTABLE_OBJECTS: frozenset = frozenset({
    "chair", "bench", "couch", "bed", "motorcycle", "bicycle",
    "horse", "elephant",
})

RIDABLE_OBJECTS: frozenset = frozenset({
    "bicycle", "motorcycle", "horse", "elephant", "boat",
})

WEARABLE_OBJECTS: frozenset = frozenset({
    "tie", "hat", "backpack", "shirt", "helmet",
})


def _get_categories(label: str) -> List[str]:
    cats: List[str] = []
    for cat_name, cat_set in _OBJECT_CATEGORIES.items():
        if label in cat_set:
            cats.append(cat_name)
    return cats


# ---------------------------------------------------------------------------
# Step 2 — Predicate validity priors (soft compatibility rules)
#
# These are SOFT scoring adjustments, NOT hard rules.
# The learned MLP remains the primary predictor.
# ---------------------------------------------------------------------------

_PREDICATE_PRIORS: Dict[str, Dict] = {
    "riding": {
        "requires_animate_subject": True,
        "preferred_object_cats": {"rideable"},
        "unsuitable_object_cats": {"furniture", "wearable", "handheld"},
        "animate_subject_bonus": 0.12,
        "inanimate_subject_penalty": -0.35,
        "preferred_object_bonus": 0.10,
        "unsuitable_object_penalty": -0.20,
    },
    "wearing": {
        "requires_animate_subject": True,
        "preferred_object_cats": {"wearable"},
        "unsuitable_object_cats": {"furniture", "rideable", "animate"},
        "animate_subject_bonus": 0.12,
        "inanimate_subject_penalty": -0.35,
        "preferred_object_bonus": 0.10,
        "unsuitable_object_penalty": -0.25,
    },
    "holding": {
        "requires_animate_subject": True,
        "preferred_object_cats": {"handheld"},
        "unsuitable_object_cats": {"furniture", "animate"},
        "animate_subject_bonus": 0.12,
        "inanimate_subject_penalty": -0.35,
        "preferred_object_bonus": 0.08,
        "unsuitable_object_penalty": -0.20,
    },
    "carrying": {
        "requires_animate_subject": True,
        "preferred_object_cats": {"handheld", "wearable"},
        "unsuitable_object_cats": {"furniture"},
        "animate_subject_bonus": 0.12,
        "inanimate_subject_penalty": -0.35,
        "preferred_object_bonus": 0.08,
        "unsuitable_object_penalty": -0.15,
    },
    "looking at": {
        "requires_animate_subject": True,
        "preferred_object_cats": set(),
        "unsuitable_object_cats": set(),
        "animate_subject_bonus": 0.10,
        "inanimate_subject_penalty": -0.30,
        "preferred_object_bonus": 0.0,
        "unsuitable_object_penalty": 0.0,
    },
    "sitting on": {
        "requires_animate_subject": True,
        "preferred_object_cats": {"furniture", "rideable"},
        "unsuitable_object_cats": {"handheld", "wearable"},
        "animate_subject_bonus": 0.12,
        "inanimate_subject_penalty": -0.35,
        "preferred_object_bonus": 0.08,
        "unsuitable_object_penalty": -0.15,
    },
    "standing on": {
        "requires_animate_subject": True,
        "preferred_object_cats": {"furniture"},
        "unsuitable_object_cats": {"handheld", "wearable"},
        "animate_subject_bonus": 0.10,
        "inanimate_subject_penalty": -0.30,
        "preferred_object_bonus": 0.06,
        "unsuitable_object_penalty": -0.12,
    },
}

# ── Pairwise plausibility rules ──────────────────────────────────────
# Each predicate defines:
#   subject_required:  categories the subject MUST belong to
#   object_forbidden:  categories the object MUST NOT belong to
#   object_allowed_overrides: if the object belongs to ANY of these,
#                             the forbidden check is overridden
#                          (e.g. "horse" is animate BUT rideable →
#                           riding/horse is allowed)
#   same_label_penalty: additional penalty when subject == object label
#   severity: "reject" → triple is rejected outright
#             "penalty" → triple is penalised (score reduction)
#   override_conf:    if adjusted_confidence exceeds this threshold,
#                     the rejection is overridden (for "penalty" severity)
# ---------------------------------------------------------------------

_PLAUSIBILITY_RULES: Dict[str, Dict] = {
    "wearing": {
        "subject_required": {"animate"},
        "object_forbidden": {"animate", "furniture", "vehicle"},
        "object_allowed_overrides": {"wearable"},
        "severity": "reject",
    },
    "sitting on": {
        "subject_required": {"animate"},
        "object_forbidden": {"animate", "handheld", "wearable"},
        "object_allowed_overrides": {"rideable", "furniture", "support_surface"},
        "severity": "reject",
    },
    "riding": {
        "subject_required": {"animate"},
        "object_forbidden": {"animate", "furniture", "handheld", "wearable"},
        "object_allowed_overrides": {"rideable"},
        "severity": "reject",
    },
    "holding": {
        "subject_required": {"animate"},
        "object_forbidden": {"furniture", "vehicle", "rideable"},
        "object_allowed_overrides": {"handheld", "wearable"},
        "severity": "reject",
    },
    "carrying": {
        "subject_required": {"animate"},
        "object_forbidden": {"furniture", "vehicle", "rideable"},
        "object_allowed_overrides": {"handheld", "wearable"},
        "severity": "penalty",
        "override_conf": 0.85,
    },
    "looking at": {
        "subject_required": {"animate"},
        "object_forbidden": set(),
        "object_allowed_overrides": set(),
        "same_label_penalty": -0.10,
        "severity": "penalty",
        "override_conf": 0.85,
    },
    "standing on": {
        "subject_required": {"animate"},
        "object_forbidden": {"animate", "handheld", "wearable"},
        "object_allowed_overrides": {"furniture", "support_surface"},
        "severity": "reject",
    },
}

# High-confidence override for penalty-severity violations.
_PLAUSIBILITY_OVERRIDE_CONF: float = 0.85


def _check_pairwise_plausibility(
    subject: str,
    predicate: str,
    object: str,
    adjusted_confidence: float = 0.0,
    debug: bool = False,
) -> Tuple[bool, str]:
    """Check whether a (subject, predicate, object) triple is semantically plausible.

    Returns:
        (is_plausible, reason)
        is_plausible: False means the triple should be rejected.
        reason:       human-readable explanation (empty string if plausible).
    """
    rule = _PLAUSIBILITY_RULES.get(predicate)
    if rule is None:
        return True, ""

    subj_cats = _get_categories(subject)
    obj_cats = _get_categories(object)
    is_same_label = (subject == object)

    # ── 1. Check subject requirements ─────────────────────────────────
    for required_cat in rule.get("subject_required", set()):
        if required_cat not in subj_cats:
            return False, f"{subject} not {required_cat} (required for {predicate})"

    # ── 2. Check object forbidden categories ──────────────────────────
    severity = rule.get("severity", "reject")
    overrides = rule.get("object_allowed_overrides", set())
    obj_cats = set(obj_cats)
    has_override = bool(overrides & obj_cats) if overrides else False

    for forbidden_cat in rule.get("object_forbidden", set()):
        if forbidden_cat in obj_cats and not has_override:
            reason = f"{object} is {forbidden_cat} — cannot {predicate} it"
            if severity == "reject":
                return False, reason
            # "penalty" severity: reject unless confidence is extremely high
            if adjusted_confidence < _PLAUSIBILITY_OVERRIDE_CONF:
                return False, f"{reason} (confidence {adjusted_confidence:.2f} < override {_PLAUSIBILITY_OVERRIDE_CONF:.2f})"
            if debug:
                print(f"[commonsense] Override: {reason} (confidence {adjusted_confidence:.2f} >= {_PLAUSIBILITY_OVERRIDE_CONF:.2f})")
            return True, ""

    # ── 3. Same-label penalty (e.g. "giraffe looking at giraffe") ─────
    same_label_penalty = rule.get("same_label_penalty", 0.0)
    if same_label_penalty < 0.0 and is_same_label and "animate" in subj_cats and "animate" in obj_cats:
        reason = f"{subject} {predicate} {object} — same-label animate pair is unlikely to be meaningful"
        if adjusted_confidence < _PLAUSIBILITY_OVERRIDE_CONF:
            return False, f"{reason} (confidence {adjusted_confidence:.2f} < override {_PLAUSIBILITY_OVERRIDE_CONF:.2f})"
        if debug:
            print(f"[commonsense] Override: {reason} (confidence {adjusted_confidence:.2f} >= {_PLAUSIBILITY_OVERRIDE_CONF:.2f})")
        return True, ""

    return True, ""


# ── Directionality penalties (Step 3) ────────────────────────────────
# These penalize unnatural subject-object assignments.
# Example: "backpack on person" → backpack should not be the subject.

_INANIMATE_SUBJECT_PENALTY: float = -0.50
_WEARABLE_SUBJECT_PENALTY: float = -0.60
_FURNITURE_SUBJECT_PENALTY: float = -0.50
_HANDHELD_SUBJECT_PENALTY: float = -0.40
_RIDEABLE_SUBJECT_PENALTY: float = -0.40

# Step 7 — Consistency margin for semantic vs spatial predicate selection
# When the best semantic score is within this margin of the best spatial score,
# the semantic predicate is preferred.
_SEMANTIC_CONSISTENCY_MARGIN: float = 0.10
_SEMANTIC_CANDIDATE_THRESHOLD: float = 0.12

# Hard negative predicate rules (Step 5)
# Format: (subject_category, predicate, object_category)
# "any" matches any category. These are semantically absurd triples.
_HARD_NEGATIVE_RULES: List[Tuple[str, str, str]] = [
    ("furniture", "holding", "any"),
    ("any", "holding", "furniture"),
    ("furniture", "wearing", "any"),
    ("any", "wearing", "furniture"),
    ("any", "wearing", "rideable"),
    ("rideable", "holding", "any"),
    ("rideable", "wearing", "any"),
    ("handheld", "wearing", "any"),
    ("handheld", "riding", "any"),
    ("handheld", "sitting on", "any"),
    ("handheld", "standing on", "any"),
    ("wearable", "riding", "any"),
    ("wearable", "sitting on", "any"),
    ("wearable", "standing on", "any"),
    ("furniture", "riding", "any"),
    ("furniture", "sitting on", "any"),
    ("furniture", "standing on", "any"),
    ("rideable", "sitting on", "any"),
    ("rideable", "standing on", "any"),
    ("any", "sitting on", "handheld"),
    ("any", "standing on", "handheld"),
    ("any", "wearing", "handheld"),
    # Animate subject riding another animate (non-rideable) → absurd
    ("animate", "riding", "animate"),
    # Holding another animate → unusual, penalize
    ("animate", "holding", "animate"),
]

# Default temperature for calibrated inference (Step 3)
_DEFAULT_TEMPERATURE: float = 2.0


def _calibrate_scores(logits: torch.Tensor, temperature: float = _DEFAULT_TEMPERATURE) -> torch.Tensor:
    """Temperature-scaled softmax for calibrated confidence.
    
    Higher temperature spreads probability mass, preventing the
    near-0/near-1 collapse that makes semantic priors ineffective.
    """
    print("[TRACE] entered _calibrate_scores")
    if temperature <= 0:
        return F.softmax(logits, dim=-1)
    return F.softmax(logits / temperature, dim=-1)


def _compute_prior_adjustment(subject: str, predicate: str, object: str) -> Tuple[float, float, float]:
    """Compute semantic prior bonus/penalty for a (subject, predicate, object) triple.
    
    Returns:
        (bonus, penalty, total_adjustment)
    """
    print("[TRACE] entered _compute_prior_adjustment")
    prior = _PREDICATE_PRIORS.get(predicate)
    is_semantic = predicate in SEMANTIC_PREDS

    subj_cats = _get_categories(subject)
    obj_cats = _get_categories(object)

    bonus = 0.0
    penalty = 0.0

    if prior is not None:
        if prior["requires_animate_subject"]:
            if "animate" in subj_cats:
                bonus += prior["animate_subject_bonus"]
            else:
                penalty += prior["inanimate_subject_penalty"]

        if prior["preferred_object_cats"]:
            if any(cat in obj_cats for cat in prior["preferred_object_cats"]):
                bonus += prior["preferred_object_bonus"]

        if prior["unsuitable_object_cats"]:
            if any(cat in obj_cats for cat in prior["unsuitable_object_cats"]):
                penalty += prior["unsuitable_object_penalty"]

    # ── Step 3 — Directionality penalties ──────────────────────────────
    # Penalize unnatural subject-object assignments.
    # Inanimate subject with animate object → likely reversed direction.
    if "animate" not in subj_cats and "animate" in obj_cats:
        if is_semantic:
            # Semantic preds with reversed direction are absurd
            # e.g. "tie wearing person"
            penalty += _INANIMATE_SUBJECT_PENALTY
        elif predicate in NEUTRAL_SPATIAL:
            # "backpack on person" → bad direction for spatial pred
            penalty += _INANIMATE_SUBJECT_PENALTY * 0.6
        elif predicate in WEAK_SPATIAL:
            # "backpack near person" → still bad direction
            penalty += _INANIMATE_SUBJECT_PENALTY * 0.4

    # Wearable as subject is always wrong direction for non-spatial preds
    if "wearable" in subj_cats:
        if is_semantic or predicate in NEUTRAL_SPATIAL:
            penalty += _WEARABLE_SUBJECT_PENALTY
        elif predicate in WEAK_SPATIAL:
            penalty += _WEARABLE_SUBJECT_PENALTY * 0.5

    # Furniture as subject for semantic predicates
    if "furniture" in subj_cats and is_semantic:
        penalty += _FURNITURE_SUBJECT_PENALTY

    # Handheld as subject for semantic predicates
    if "handheld" in subj_cats and is_semantic:
        penalty += _HANDHELD_SUBJECT_PENALTY

    # Rideable as subject for semantic predicates
    if "rideable" in subj_cats and is_semantic:
        penalty += _RIDEABLE_SUBJECT_PENALTY

    return bonus, penalty, bonus + penalty


# ---------------------------------------------------------------------------
# PART 1 — Commonsense affordance penalty
# These are multiplicative confidence penalties for physically implausible
# relations that pass the semantic prior checks but violate basic physics.
#
# Returns (penalty_factor, reason_string).
# penalty_factor = 1.0 means no change; < 1.0 reduces confidence.
# ---------------------------------------------------------------------------

def _compute_commonsense_penalty(predicate: str, object: str) -> Tuple[float, str]:
    """Compute multiplicative penalty for physically implausible relations.
    
    Checks whether the object plausibly supports the given interaction type.
    Does NOT fully suppress relations — only applies a soft confidence penalty
    so that unusual but possible relations (e.g. "person riding elephant")
    remain valid while absurd ones (e.g. "person sitting on dog") are
    deprioritised.
    
    Returns:
        (penalty_factor, reason_string)
        penalty_factor: 1.0 = no penalty, < 1.0 = confidence reduction.
        reason_string:  human-readable explanation for debug output.
    """
    print("[TRACE] entered _compute_commonsense_penalty")
    obj_lower = object.lower().replace("_", " ")

    if predicate == "sitting on" and obj_lower not in SITTABLE_OBJECTS:
        return (0.65, f"{object} not sit-compatible")

    if predicate == "wearing" and obj_lower not in WEARABLE_OBJECTS:
        return (0.50, f"{object} not wear-compatible")

    if predicate == "riding" and obj_lower not in RIDABLE_OBJECTS:
        return (0.60, f"{object} not ride-compatible")

    return (1.0, "")


# ---------------------------------------------------------------------------
# PART 3 — Per-predicate semantic bonuses for final reranking
# These provide fine-grained ranking differentiation between semantic
# predicates so that stronger interaction types (sitting on, wearing)
# are preferred over weaker ones (looking at) when confidences are close.
# ---------------------------------------------------------------------------

SEMANTIC_BONUSES: Dict[str, float] = {
    "wearing": 0.10,
    "sitting on": 0.08,
    "riding": 0.08,
    "holding": 0.06,
    "carrying": 0.05,
    "standing on": 0.05,
    "looking at": 0.03,
}


# ── Confidence calibration thresholds (Parts 1, 4) ──────────────────────
# These prevent the model from fabricating semantic relations when
# evidence is weak or ambiguous. The goal is to prefer NO_RELATION
# over low-confidence predicate predictions.
#
# MIN_RELATION_CONFIDENCE: minimum top1 score for any relation to survive.
# MIN_RELATION_MARGIN:     minimum top1-top2 gap needed for a confident pick.
# WEAK_PREDICATE_EXTRA_MARGIN: additional margin weak predicates must meet.
MIN_RELATION_CONFIDENCE: float = 0.38
MIN_RELATION_MARGIN: float = 0.12

# Weak predicates that require stricter evidence to survive.
# These are prone to overprediction: "looking at" grabs any animate pair,
# "near"/"in front of"/"behind" fire from geometry alone.
WEAK_PREDICATES: frozenset = frozenset({
    "looking at", "near", "in front of", "behind",
})
WEAK_PREDICATE_EXTRA_MARGIN: float = 0.08

# ═══════════════════════════════════════════════════════════════════════
# Predicate-aware calibration system
# ═══════════════════════════════════════════════════════════════════════
#
# Replaces simple global margin thresholds with adaptive thresholds that:
# 1. Classify predicates into families by semantic characteristics
# 2. Use confidence-dependent adaptive margin (higher conf → lower margin need)
# 3. Consider competing predicate family (same-family close calls are OK)
# 4. Preserve hallucination suppression while reducing false negatives
#
# Key insight from experimental evidence:
# - Semantic predicates naturally have closer competing logits (riding↔wearing)
# - Weak spatial predicates should remain strict (near/behind are geometry noise)
# - The predictor is stronger than the calibration layer now
#
# Each family defines:
#   base_conf:        minimum confidence threshold
#   base_margin:      baseline margin before any relaxation
#   relax_min:        margin multiplier at peak confidence (1.0)
#   relax_max:        margin multiplier at base_conf

PREDICATE_FAMILIES: Dict[str, Dict] = {
    "strong_semantic": {
        "predicates": {"riding", "wearing", "sitting on", "standing on", "holding", "carrying"},
        "base_conf": 0.27,
        "base_margin": 0.08,
        "relax_min": 0.50,  # at conf=1.0: margin × 0.50
        "relax_max": 1.00,  # at conf=base_conf: margin × 1.00
    },
    "attentional": {
        "predicates": {"looking at"},
        "base_conf": 0.32,
        "base_margin": 0.15,
        "relax_min": 0.70,
        "relax_max": 1.00,
    },
    "neutral_spatial": {
        "predicates": {"on", "in"},
        "base_conf": 0.38,
        "base_margin": 0.14,
        "relax_min": 0.80,
        "relax_max": 1.00,
    },
    "weak_spatial": {
        "predicates": {"near", "behind", "in front of", "under", "above",
                       "next to", "over", "inside", "attached to", "covering"},
        "base_conf": 0.40,
        "base_margin": 0.18,
        "relax_min": 1.00,  # no confidence-based relaxation
        "relax_max": 1.00,
    },
}

# Build inverse: predicate name → family config
_PREDICATE_TO_FAMILY: Dict[str, Dict] = {}
for _family_config in PREDICATE_FAMILIES.values():
    for _pred in _family_config["predicates"]:
        _PREDICATE_TO_FAMILY[_pred] = _family_config

# Competitor family relaxation: when top-1 and top-2 belong to these
# family pairs, the margin is relaxed because competition between
# semantically similar predicates is expected.
# Key case: riding (strong_semantic) vs wearing (strong_semantic) → 0.70
_COMPETITOR_FAMILY_RELAXATION: Dict[Tuple[str, str], float] = {
    ("strong_semantic", "strong_semantic"): 0.70,
    ("strong_semantic", "attentional"):      0.80,
    ("strong_semantic", "neutral_spatial"):  0.80,
    ("strong_semantic", "weak_spatial"):     0.75,
    ("attentional", "strong_semantic"):      0.80,
    ("attentional", "attentional"):          0.85,
    ("attentional", "neutral_spatial"):      0.90,
    ("attentional", "weak_spatial"):         0.90,
    ("neutral_spatial", "neutral_spatial"):  0.95,
    ("neutral_spatial", "weak_spatial"):     0.95,
    ("weak_spatial", "weak_spatial"):        1.00,
    ("weak_spatial", "strong_semantic"):     0.90,
    ("weak_spatial", "neutral_spatial"):     0.95,
}
_DEFAULT_COMPETITOR_RELAXATION: float = 1.0


def _get_predicate_family(pred: str) -> Optional[str]:
    """Get the family name for a predicate, or None if unknown."""
    config = _PREDICATE_TO_FAMILY.get(pred)
    if config is None:
        return None
    for family_name, family_config in PREDICATE_FAMILIES.items():
        if family_config is config:
            return family_name
    return None


def _compute_adaptive_calibration(
    top1_name: str,
    top1_score: float,
    top2_name: Optional[str],
    top2_score: Optional[float],
    margin: float,
    debug: bool = False,
) -> Tuple[bool, Optional[str], Dict]:
    """Predicate-aware calibration decision.
    
    Effective margin = base_margin × confidence_factor × competitor_factor
    
    - base_margin: predicate family baseline
    - confidence_factor: linear interpolation from 1.0 (at base_conf) to
      relax_min (at conf=1.0)
    - competitor_factor: relaxation based on top-1 / top-2 family compatibility
    
    Returns:
        (accepted, reject_reason, debug_info)
    """
    debug_info = {
        "top1_name": top1_name,
        "top1_score": round(top1_score, 4),
        "top2_name": top2_name or "",
        "top2_score": round(top2_score, 4) if top2_score is not None else 0.0,
        "margin": round(margin, 4),
        "family": None,
        "top2_family": None,
        "base_conf": None,
        "base_margin_val": None,
        "relax_min": None,
        "relax_max": None,
        "confidence_factor": None,
        "competitor_factor": None,
        "effective_margin": None,
    }

    family_config = _PREDICATE_TO_FAMILY.get(top1_name)
    top1_family = _get_predicate_family(top1_name)
    debug_info["family"] = top1_family

    if family_config is None:
        # Unknown predicate fallback: use global defaults
        conf_thresh = MIN_RELATION_CONFIDENCE
        margin_thresh = MIN_RELATION_MARGIN
        if top1_name in WEAK_PREDICATES:
            margin_thresh += WEAK_PREDICATE_EXTRA_MARGIN
        debug_info["base_conf"] = conf_thresh
        debug_info["base_margin_val"] = margin_thresh
        if top1_score < conf_thresh:
            reason = f"conf={top1_score:.3f} < threshold={conf_thresh:.3f}"
            if debug:
                print(f"  [calib/family=unknown] REJECTED: {top1_name} {reason}")
            return False, reason, debug_info
        if top2_name is not None and margin < margin_thresh:
            reason = f"margin={margin:.3f} < threshold={margin_thresh:.3f}"
            if debug:
                print(f"  [calib/family=unknown] REJECTED: {top1_name} {reason}")
            return False, reason, debug_info
        if debug:
            print(f"  [calib/family=unknown] ACCEPTED: {top1_name}")
        return True, None, debug_info

    base_conf = family_config["base_conf"]
    base_margin = family_config["base_margin"]
    relax_min = family_config["relax_min"]
    relax_max = family_config["relax_max"]

    debug_info["base_conf"] = base_conf
    debug_info["base_margin_val"] = base_margin
    debug_info["relax_min"] = relax_min
    debug_info["relax_max"] = relax_max

    # ── Step 1: Confidence check ─────────────────────────────────────
    if top1_score < base_conf:
        reason = f"conf={top1_score:.3f} < family threshold={base_conf:.3f}"
        if debug:
            print(f"  [calib/family={top1_family}] REJECTED: {top1_name} {reason}")
        return False, reason, debug_info

    # ── Step 2: Compute confidence-dependent margin relaxation ───────
    if relax_max == relax_min:
        confidence_factor = relax_max
    else:
        conf_range = max(0.001, 1.0 - base_conf)
        conf_above_base = max(0.0, top1_score - base_conf)
        t = min(1.0, conf_above_base / conf_range)
        confidence_factor = relax_max - (relax_max - relax_min) * t
    debug_info["confidence_factor"] = round(confidence_factor, 4)

    # ── Step 3: Competitor family relaxation ──────────────────────
    if top2_name is not None:
        top2_family = _get_predicate_family(top2_name)
        debug_info["top2_family"] = top2_family

        if top2_family is not None and top1_family is not None:
            competitor_key = (top1_family, top2_family)
            competitor_factor = _COMPETITOR_FAMILY_RELAXATION.get(competitor_key, _DEFAULT_COMPETITOR_RELAXATION)
        else:
            competitor_factor = _DEFAULT_COMPETITOR_RELAXATION
    else:
        # Single valid predicate: no competitor to worry about
        competitor_factor = _DEFAULT_COMPETITOR_RELAXATION
    debug_info["competitor_factor"] = round(competitor_factor, 4)

    # ── Step 4: Compute effective margin ─────────────────────────────
    effective_margin = base_margin * confidence_factor * competitor_factor
    min_margin = 0.02
    effective_margin = max(min_margin, min(base_margin, effective_margin))
    debug_info["effective_margin"] = round(effective_margin, 4)

    top2_score_str = f"{top2_score:.4f}" if top2_score is not None else "—"
    if debug:
        print(f"  [calib/family={top1_family}] top1={top1_name} ({top1_score:.4f}) "
              f"top2={top2_name or '—'} ({top2_score_str})")
        print(f"          margin={margin:.4f}  base_margin={base_margin:.3f}  "
              f"conf_factor={confidence_factor:.3f}  comp_factor={competitor_factor:.2f}")
        print(f"          effective_margin={effective_margin:.3f}  "
              f"conf_threshold={base_conf:.3f}")

    # ── Step 5: Margin check ─────────────────────────────────────────
    if top2_name is not None and margin < effective_margin:
        reason = (f"margin={margin:.3f} < effective={effective_margin:.3f}"
                  f" (base={base_margin:.3f} × conf={confidence_factor:.2f} × comp={competitor_factor:.2f})")
        if debug:
            print(f"  [calib] REJECTED: {top1_name} {reason}")
        return False, reason, debug_info

    if debug:
        if top2_name is not None:
            print(f"  [calib] ACCEPTED: {top1_name} "
                  f"(conf={top1_score:.3f} >= {base_conf:.3f}, "
                  f"margin={margin:.3f} >= {effective_margin:.3f})")
        else:
            print(f"  [calib] ACCEPTED: {top1_name} "
                  f"(conf={top1_score:.3f} >= {base_conf:.3f}, "
                  f"no competitor)")
    return True, None, debug_info


# ── Legacy strong predicate thresholds (kept for backward compat) ─────
# These are superseded by the adaptive calibration system above.
STRONG_PREDICATES: frozenset = frozenset({
    "wearing",
    "sitting on",
    "riding",
    "holding",
})

STRONG_PREDICATE_MIN_CONF: dict[str, float] = {
    "wearing": 0.26,
    "sitting on": 0.28,
    "riding": 0.27,
    "holding": 0.25,
}

STRONG_PREDICATE_MIN_MARGIN: dict[str, float] = {
    "wearing": 0.05,
    "sitting on": 0.06,
    "riding": 0.06,
    "holding": 0.05,
}

# Sentinel value indicating no strong relation was detected.
# Used downstream to avoid forcing interaction-centric captions.
NO_RELATION: str = "NO_RELATION"


def _is_extreme_nonsense(subject: str, predicate: str, object: str) -> bool:
    """Hard-filter semantically impossible triples (Step 5).
    
    Rejects:
    1. Any predicate requiring animate subject when subject is inanimate.
    2. Hard negative rule violations.
    3. Same-object semantic absurdities.
    """
    print("[TRACE] entered _is_extreme_nonsense")
    # Check 1: Predicate requires animate subject but subject isn't
    prior = _PREDICATE_PRIORS.get(predicate)
    if prior is not None and prior.get("requires_animate_subject", False):
        subj_cats = _get_categories(subject)
        if "animate" not in subj_cats:
            return True

    # Check 2: Hard negative rules
    if _check_hard_negative(subject, predicate, object):
        return True

    # Check 3: Same-object nonsense
    if subject == object:
        # Animate self-relations: "person holding person" → nonsense
        if "animate" in _get_categories(subject):
            if predicate in {"holding", "wearing", "carrying",
                             "sitting on", "standing on", "riding"}:
                return True
        # Inanimate self-relations with semantic preds → nonsense
        elif predicate in SEMANTIC_PREDS:
            return True

    return False


def _check_hard_negative(subject: str, predicate: str, object: str) -> bool:
    """Check if a triple matches a hard negative rule (Step 5).
    
    These are semantically absurd interactions that should never
    appear in grounded captions, e.g. "chair holding phone",
    "backpack riding bicycle", "tie wearing person".
    
    Exceptions:
    - animate+riding+animate is allowed when object is rideable
      (e.g. "person riding horse" — horse is both animate and rideable)
    
    Returns:
        True if the triple should be rejected.
    """
    print("[TRACE] entered _check_hard_negative")
    subj_cats = _get_categories(subject)
    obj_cats = _get_categories(object)

    for subj_cat, pred, obj_cat in _HARD_NEGATIVE_RULES:
        if pred != predicate:
            continue
        subj_match = subj_cat == "any" or subj_cat in subj_cats
        obj_match = obj_cat == "any" or obj_cat in obj_cats
        if not (subj_match and obj_match):
            continue

        # Exception: animate+riding+animate allowed when object is rideable
        # e.g. "person riding horse" — horse is both animate and rideable
        if (predicate == "riding" and subj_cat == "animate" and obj_cat == "animate"
                and "rideable" in obj_cats):
            continue

        return True

    # Additional semantic absurdities:
    # Cell phone cannot be "under" a person (tiny object over big = nonsense)
    if "handheld" in subj_cats and "animate" in obj_cats:
        if predicate in {"under", "above", "covering"}:
            return True

    # Wearable as subject with spatial predicate → reversed direction
    if "wearable" in subj_cats and "animate" in obj_cats:
        if predicate in NEUTRAL_SPATIAL:
            return True

    return False


# ---------------------------------------------------------------------------
# Step 4 — Feature contribution analysis utilities
# ---------------------------------------------------------------------------

def _get_feature_group_norms(model: nn.Module) -> Dict[str, float]:
    """Analyze feature group contributions.

    For MLP: L2 norm of first-layer weight columns per feature group.
    For Transformer: cross-attention weight aggregation per modality.

    Returns:
        Dict mapping group name to contribution score.
    """
    if isinstance(model, RelationTransformer):
        return _get_transformer_feature_norms(model)

    # MLP path
    first_weight = model.mlp[0].weight  # (H, in_dim)
    embed_dim = model.label_emb.weight.shape[1]
    clip_dim = model.clip_dim
    union_dim = model.union_dim
    pose_dim = model.pose_dim
    pose_object_dim = getattr(model, "pose_object_dim", 0)

    groups = {
        "subj_label": (0, embed_dim),
        "obj_label":  (embed_dim, 2 * embed_dim),
        "geo":        (2 * embed_dim, 2 * embed_dim + GEO_DIM),
    }
    offset = 2 * embed_dim + GEO_DIM
    if clip_dim > 0:
        groups["subj_clip"] = (offset, offset + clip_dim)
        offset += clip_dim
        groups["obj_clip"]  = (offset, offset + clip_dim)
        offset += clip_dim
    if union_dim > 0:
        groups["union_clip"] = (offset, offset + union_dim)
        offset += union_dim
    if pose_dim > 0:
        groups["pose"] = (offset, offset + pose_dim)
        offset += pose_dim
    if pose_object_dim > 0:
        groups["pose_object"] = (offset, offset + pose_object_dim)

    norms = {}
    for name, (start, end) in groups.items():
        group_weight = first_weight[:, start:end]
        norms[name] = group_weight.norm().item()

    return norms


def _get_transformer_feature_norms(model: RelationTransformer) -> Dict[str, float]:
    """Estimate feature importance via projection weight norms.

    For the transformer, each modality has a linear projection layer.
    We use the L2 norm of each projection's weight as a proxy for
    how much information flows through that modality.

    This mirrors the MLP approach structurally.
    """
    norms = {}
    norms["subj_label"] = model.subj_label_proj.weight.norm().item()
    norms["obj_label"] = model.obj_label_proj.weight.norm().item()
    norms["geo"] = model.geo_proj.weight.norm().item()
    if hasattr(model, 'subj_clip_proj'):
        norms["subj_clip"] = model.subj_clip_proj.weight.norm().item()
        norms["obj_clip"] = model.obj_clip_proj.weight.norm().item()
    if hasattr(model, 'union_proj'):
        norms["union_clip"] = model.union_proj.weight.norm().item()
    if hasattr(model, 'pose_proj'):
        norms["pose"] = model.pose_proj.weight.norm().item()
    if hasattr(model, 'pose_object_proj'):
        norms["pose_object"] = model.pose_object_proj.weight.norm().item()
    return norms


def _measure_feature_ablation(
    model: nn.Module,
    subj_idx: torch.Tensor,
    obj_idx: torch.Tensor,
    geo: torch.Tensor,
    subj_feat: Optional[torch.Tensor],
    obj_feat: Optional[torch.Tensor],
    pred_vocab: Vocab,
    union_feat: Optional[torch.Tensor] = None,
    pose_feat: Optional[torch.Tensor] = None,
) -> Dict:
    """Measure prediction changes when feature groups are ablated.

    Compares full model prediction to predictions with:
    - CLIP features zeroed
    - Union features zeroed
    - Pose features zeroed
    - Geometry features zeroed

    Returns:
        Dict with prediction changes and logit differences.
    """
    with torch.no_grad():
        def _forward(sf, of, uf, pf, geo_t):
            return model(subj_idx, obj_idx, geo_t,
                         subj_feat=sf, obj_feat=of,
                         union_feat=uf, pose_feat=pf)

        full_logits = _forward(subj_feat, obj_feat, union_feat, pose_feat, geo)

        full_pred_idx = full_logits[0].argmax(dim=-1).item()
        full_pred = pred_vocab.token(full_pred_idx)
        full_probs = F.softmax(full_logits[0], dim=-1)
        full_max_prob = full_probs.max().item()

        result = {
            "full_prediction": full_pred,
            "full_confidence": round(full_max_prob, 4),
        }

        # Ablate CLIP
        if subj_feat is not None and obj_feat is not None:
            zeros_c = torch.zeros_like(subj_feat)
            no_clip_logits = _forward(zeros_c, zeros_c, union_feat, pose_feat, geo)
            no_clip_pred_idx = no_clip_logits[0].argmax(dim=-1).item()
            no_clip_pred = pred_vocab.token(no_clip_pred_idx)
            no_clip_probs = F.softmax(no_clip_logits[0], dim=-1)
            no_clip_max_prob = no_clip_probs.max().item()
            logit_diff = (full_logits[0] - no_clip_logits[0]).norm().item()

            result["ablate_clip_prediction"] = no_clip_pred
            result["ablate_clip_confidence"] = round(no_clip_max_prob, 4)
            result["clip_logit_norm_diff"] = round(logit_diff, 4)
            result["pred_changed_without_clip"] = full_pred != no_clip_pred

        # Ablate union
        if union_feat is not None and model.union_dim > 0:
            zeros_u = torch.zeros_like(union_feat)
            no_union_logits = _forward(subj_feat, obj_feat, zeros_u, pose_feat, geo)
            no_union_pred_idx = no_union_logits[0].argmax(dim=-1).item()
            no_union_pred = pred_vocab.token(no_union_pred_idx)
            no_union_probs = F.softmax(no_union_logits[0], dim=-1)
            no_union_max_prob = no_union_probs.max().item()
            union_logit_diff = (full_logits[0] - no_union_logits[0]).norm().item()

            result["ablate_union_prediction"] = no_union_pred
            result["ablate_union_confidence"] = round(no_union_max_prob, 4)
            result["union_logit_norm_diff"] = round(union_logit_diff, 4)
            result["pred_changed_without_union"] = full_pred != no_union_pred

        # Ablate pose
        if pose_feat is not None and model.pose_dim > 0:
            zeros_p = torch.zeros_like(pose_feat)
            no_pose_logits = _forward(subj_feat, obj_feat, union_feat, zeros_p, geo)
            no_pose_pred_idx = no_pose_logits[0].argmax(dim=-1).item()
            no_pose_pred = pred_vocab.token(no_pose_pred_idx)
            no_pose_probs = F.softmax(no_pose_logits[0], dim=-1)
            no_pose_max_prob = no_pose_probs.max().item()
            pose_logit_diff = (full_logits[0] - no_pose_logits[0]).norm().item()

            result["ablate_pose_prediction"] = no_pose_pred
            result["ablate_pose_confidence"] = round(no_pose_max_prob, 4)
            result["pose_logit_norm_diff"] = round(pose_logit_diff, 4)
            result["pred_changed_without_pose"] = full_pred != no_pose_pred

        # Ablate geometry
        zero_geo = torch.zeros_like(geo)
        no_geo_logits = _forward(subj_feat, obj_feat, union_feat, pose_feat, zero_geo)
        no_geo_pred_idx = no_geo_logits[0].argmax(dim=-1).item()
        no_geo_pred = pred_vocab.token(no_geo_pred_idx)
        no_geo_probs = F.softmax(no_geo_logits[0], dim=-1)
        no_geo_max_prob = no_geo_probs.max().item()
        geo_logit_diff = (full_logits[0] - no_geo_logits[0]).norm().item()

        result["ablate_geo_prediction"] = no_geo_pred
        result["ablate_geo_confidence"] = round(no_geo_max_prob, 4)
        result["geo_logit_norm_diff"] = round(geo_logit_diff, 4)
        result["pred_changed_without_geo"] = full_pred != no_geo_pred

    return result


def _analyze_feature_utilization(
    model: nn.Module,
    subj_idx: torch.Tensor,
    obj_idx: torch.Tensor,
    geo: torch.Tensor,
    subj_feat: Optional[torch.Tensor],
    obj_feat: Optional[torch.Tensor],
    pred_vocab: Vocab,
    union_feat: Optional[torch.Tensor] = None,
    pose_feat: Optional[torch.Tensor] = None,
) -> Dict:
    """Full feature utilization analysis for a single pair.

    Combines:
    1. Feature group weight norms
    2. Feature ablation study
    3. Contribution breakdown across ALL feature groups

    Returns:
        Dict with all analysis results.
    """
    norms = _get_feature_group_norms(model)
    ablation = _measure_feature_ablation(
        model, subj_idx, obj_idx, geo, subj_feat, obj_feat,
        pred_vocab, union_feat=union_feat, pose_feat=pose_feat,
    )

    total_norm = sum(norms.values()) or 1.0
    contribution = {k: round(v / total_norm, 4) for k, v in norms.items()}

    clip_pct = round(
        (norms.get("subj_clip", 0) + norms.get("obj_clip", 0)) / total_norm * 100, 1
    )
    union_pct = round(norms.get("union_clip", 0) / total_norm * 100, 1)
    pose_pct = round(norms.get("pose", 0) / total_norm * 100, 1)
    geo_pct = round(norms.get("geo", 0) / total_norm * 100, 1)
    label_pct = round(
        (norms.get("subj_label", 0) + norms.get("obj_label", 0)) / total_norm * 100, 1
    )

    return {
        "feature_norms": norms,
        "feature_contribution": contribution,
        "ablation": ablation,
        "clip_contribution_pct": clip_pct,
        "union_contribution_pct": union_pct,
        "pose_contribution_pct": pose_pct,
        "geo_contribution_pct": geo_pct,
        "label_contribution_pct": label_pct,
    }


# ---------------------------------------------------------------------------
# Step 7 — Semantic consistency checking
# ---------------------------------------------------------------------------

def _apply_semantic_consistency(
    candidates: List[Dict],
) -> List[Dict]:
    """Apply consistency checking to ensure coherent semantic interpretations.
    
    For each (subject, object) pair, if a semantic predicate has strong
    enough support, suppress weaker conflicting spatial alternatives.
    
    The goal is: only one coherent semantic interpretation per pair.
    If the model thinks "person riding bicycle" is plausible, we should
    not also produce "person on bicycle" or "bicycle under person".
    
    Args:
        candidates: List of candidate relation dicts (multiple per pair possible).
        
    Returns:
        Filtered list with at most one predicate per pair (the best one).
    """
    print("[TRACE] entered _apply_semantic_consistency")
    # Group by (subject, object) pair (ordered)
    pair_groups: Dict[Tuple[str, str], List[Dict]] = {}
    for c in candidates:
        # Use the subject/object pair as-is (not sorted)
        # so direction is preserved
        key = (c["subject"], c["object"])
        if key not in pair_groups:
            pair_groups[key] = []
        pair_groups[key].append(c)

    result = []
    for key, group in pair_groups.items():
        if len(group) == 1:
            result.append(group[0])
            continue

        # Find best semantic and spatial scores
        best_semantic = None
        best_semantic_score = -float('inf')
        best_spatial = None
        best_spatial_score = -float('inf')
        best_any = None
        best_any_score = -float('inf')

        for c in group:
            score = c.get("adjusted_confidence", c["confidence"])
            if score > best_any_score:
                best_any_score = score
                best_any = c
            if c["predicate"] in SEMANTIC_PREDS:
                if score > best_semantic_score:
                    best_semantic_score = score
                    best_semantic = c
            else:
                if score > best_spatial_score:
                    best_spatial_score = score
                    best_spatial = c

        # Decision logic:
        # 1. If semantic has good support and is close to spatial best → prefer semantic
        # 2. If only spatial exists → keep best spatial
        # 3. If only semantic exists → keep best semantic
        if (best_semantic is not None and
            best_semantic_score >= _SEMANTIC_CANDIDATE_THRESHOLD and
            (best_spatial is None or
             best_semantic_score >= best_spatial_score - _SEMANTIC_CONSISTENCY_MARGIN)):
            result.append(best_semantic)
            if best_spatial is not None:
                debug_print(f"  [consistency] PREFERRED semantic '{best_semantic['predicate']}' "
                            f"({best_semantic_score:.3f}) over spatial "
                            f"'{best_spatial['predicate']}' ({best_spatial_score:.3f})")
        else:
            result.append(best_any)

    return result


# ---------------------------------------------------------------------------
# Step 8 — Relation quality evaluation
# ---------------------------------------------------------------------------

def evaluate_relation_quality(
    relations: List[Dict],
    raw_debug: List[Dict],
) -> Dict:
    """Evaluate quality of extracted relations.
    
    Computes metrics for:
    - Semantic relation precision (higher is better)
    - Inanimate subject rate (lower is better)
    - Reversed direction rate (lower is better)
    - Weak spatial clutter rate (lower is better)
    
    Tracks separately:
    - Semantic predicates (riding, holding, wearing, carrying, looking at,
      sitting on, standing on)
    - Spatial predicates (on, in, under, above, near, next to, etc.)
    
    Args:
        relations: Final selected relations.
        raw_debug: All evaluated pairs from infer_relationships_semantic.
        
    Returns:
        Dict with all computed metrics.
    """
    print("[TRACE] entered evaluate_relation_quality")
    metrics: Dict = {
        "total_relations": len(relations),
        "semantic_relations": 0,
        "spatial_relations": 0,
        "animate_subject": 0,
        "inanimate_subject": 0,
        "reversed_direction_suspected": 0,
        "weak_spatial": 0,
        "neutral_spatial": 0,
    }

    predicate_breakdown: Dict[str, int] = {}

    for r in relations:
        pred = r["predicate"]
        predicate_breakdown[pred] = predicate_breakdown.get(pred, 0) + 1

        subj_cats = _get_categories(r["subject"])
        obj_cats = _get_categories(r["object"])

        if pred in SEMANTIC_PREDS:
            metrics["semantic_relations"] += 1
        elif pred in WEAK_SPATIAL:
            metrics["weak_spatial"] += 1
            metrics["spatial_relations"] += 1
        elif pred in NEUTRAL_SPATIAL:
            metrics["neutral_spatial"] += 1
            metrics["spatial_relations"] += 1
        else:
            metrics["spatial_relations"] += 1

        if "animate" in subj_cats:
            metrics["animate_subject"] += 1
        else:
            metrics["inanimate_subject"] += 1
            if "animate" in obj_cats:
                metrics["reversed_direction_suspected"] += 1

    total = max(len(relations), 1)
    metrics["semantic_precision"] = round(metrics["semantic_relations"] / total, 4)
    metrics["animate_subject_rate"] = round(metrics["animate_subject"] / total, 4)
    metrics["reversed_direction_rate"] = round(
        metrics["reversed_direction_suspected"] / total, 4
    )
    metrics["weak_spatial_rate"] = round(metrics["weak_spatial"] / total, 4)
    metrics["predicate_breakdown"] = predicate_breakdown

    # Pair-level stats from raw_debug
    total_evaluated = len([d for d in raw_debug if d.get("status") == "candidate"])
    metrics["total_pairs_evaluated"] = total_evaluated
    metrics["selection_ratio"] = round(
        len(relations) / max(total_evaluated, 1), 4
    )

    return metrics


# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------

_model:         Optional[nn.Module] = None
_label_vocab:   Optional[Vocab]      = None
_pred_vocab:    Optional[Vocab]      = None
_device:        Optional[torch.device] = None
_clip_model:    Optional[CLIPExtractor] = None
_pose_model:    Optional[PoseExtractor] = None
_model_clip_dim: int = 0
_model_pose_dim: int = 0
_model_pose_object_dim: int = 0
_model_union_dim: int = 0
_model_type:     str = "mlp"

_DEFAULT_CKPT_DIR = os.environ.get("REL_CKPT_DIR", "./checkpoints")


def load_relation_model(checkpoint_dir: str = _DEFAULT_CKPT_DIR) -> None:
    print("[TRACE] entered load_relation_model")
    global _model, _label_vocab, _pred_vocab, _device, _model_clip_dim
    global _model_pose_dim, _model_pose_object_dim, _model_union_dim, _model_type

    model_path = os.path.join(checkpoint_dir, "relation_mlp.pt")
    lv_path    = os.path.join(checkpoint_dir, "label_vocab.json")
    pv_path    = os.path.join(checkpoint_dir, "pred_vocab.json")

    for p, name in [(model_path, "relation_mlp.pt"),
                    (lv_path,    "label_vocab.json"),
                    (pv_path,    "pred_vocab.json")]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"[relation_prediction] Checkpoint file not found: {p}\n"
                f"  Run training first:\n"
                f"    python -m relation_prediction.train --vg-root ./data/visual_genome"
            )

    _label_vocab = Vocab.load(lv_path)
    _pred_vocab  = Vocab.load(pv_path)
    _device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_data = torch.load(model_path, map_location=_device, weights_only=True)

    # Determine model type from config or infer from state_dict keys.
    if isinstance(raw_data, dict) and "model_config" in raw_data:
        config = raw_data.get("model_config", {})
        model_type = config.get("model_type", "mlp")
    else:
        model_type = "mlp"

    if model_type == "transformer":
        state = raw_data["model_state_dict"]
        clip_dim = config.get("clip_dim", 0)
        pose_dim = config.get("pose_dim", 0)
        pose_object_dim = config.get("pose_object_dim", 0)
        union_dim = config.get("union_dim", 0)
        embed_dim = config.get("embed_dim", 64)
        d_model = config.get("d_model", 256)

        _model_clip_dim = clip_dim
        _model_pose_dim = pose_dim
        _model_pose_object_dim = pose_object_dim
        _model_union_dim = union_dim
        _model_type = "transformer"

        _model = RelationTransformer(
            num_labels=len(_label_vocab),
            num_predicates=len(_pred_vocab),
            d_model=d_model,
            embed_dim=embed_dim,
            clip_dim=clip_dim,
            pose_dim=pose_dim,
            pose_object_dim=pose_object_dim,
            union_dim=union_dim,
        )
        _model.load_state_dict(state)
        _model.to(_device)
        _model.eval()

        mode_parts = []
        if clip_dim > 0:
            mode_parts.append("visual-semantic")
        if union_dim > 0:
            mode_parts.append("union")
        if pose_dim > 0:
            mode_parts.append("pose")
        if pose_object_dim > 0:
            mode_parts.append("pose_object")
        mode_str = "+".join(mode_parts) if mode_parts else "geometry-only"
        debug_print(
            f"[relation_prediction] Loaded TRANSFORMER model from {checkpoint_dir} "
            f"({len(_label_vocab):,} labels, {len(_pred_vocab):,} predicates, "
            f"{mode_str}, d_model={d_model})"
        )

    else:
        # Support both new format (with config) and old format (bare state_dict).
        if isinstance(raw_data, dict) and "model_state_dict" in raw_data:
            state = raw_data["model_state_dict"]
            config = raw_data.get("model_config", {})
            pose_dim = config.get("pose_dim", 0)
            pose_object_dim = config.get("pose_object_dim", 0)
            union_dim = config.get("union_dim", 0)
            clip_dim = config.get("clip_dim", 0)
            embed_dim = config.get("embed_dim", state["label_emb.weight"].shape[1])
            hidden_dims = _infer_hidden_dims(state)
        else:
            state = raw_data
            embed_dim = state["label_emb.weight"].shape[1]
            hidden_dims = _infer_hidden_dims(state)
            clip_dim = _infer_clip_dim(state, embed_dim)
            pose_dim = 0
            pose_object_dim = 0
            union_dim = 0

        _model_clip_dim = clip_dim
        _model_pose_dim = pose_dim
        _model_pose_object_dim = pose_object_dim
        _model_union_dim = union_dim
        _model_type = "mlp"

        _model = RelationMLP(
            num_labels=len(_label_vocab),
            num_predicates=len(_pred_vocab),
            embed_dim=embed_dim,
            hidden_dims=hidden_dims,
            clip_dim=clip_dim,
            pose_dim=pose_dim,
            pose_object_dim=pose_object_dim,
            union_dim=union_dim,
        )
        _model.load_state_dict(state)
        _model.to(_device)
        _model.eval()

        input_dim = 2 * embed_dim + GEO_DIM + 2 * clip_dim + union_dim + pose_dim
        mode_parts = []
        if clip_dim > 0:
            mode_parts.append("visual-semantic")
        if union_dim > 0:
            mode_parts.append("union")
        if pose_dim > 0:
            mode_parts.append("pose")
        mode_str = "+".join(mode_parts) if mode_parts else "geometry-only"
        debug_print(
            f"[relation_prediction] Loaded model from {checkpoint_dir} "
            f"({len(_label_vocab):,} labels, {len(_pred_vocab):,} predicates, "
            f"{mode_str}, input_dim={input_dim})"
        )
        debug_print(f"[RelationMLP]")
        debug_print(f"  Loaded config:")
        debug_print(f"    input_dim={input_dim}")
        debug_print(f"    hidden_dims={hidden_dims}")
        debug_print(f"    pose_dim={pose_dim}")
        debug_print(f"    union_dim={union_dim}")


def _infer_hidden_dims(state: dict) -> Tuple[int, ...]:
    hidden: List[int] = []
    idx = 0
    while True:
        key = f"mlp.{idx}.weight"
        if key not in state:
            break
        hidden.append(state[key].shape[0])
        idx += 3
    return tuple(hidden[:-1])


def _infer_clip_dim(state: dict, embed_dim: int) -> int:
    """
    Infer clip_dim from the first MLP layer's input weight shape.
    clip_dim = (in_features - 2*embed_dim - GEO_DIM) // 2
    """
    first_weight = state["mlp.0.weight"]  # (out_features, in_features)
    in_features = first_weight.shape[1]
    clip_portion = in_features - 2 * embed_dim - GEO_DIM
    if clip_portion <= 0:
        return 0
    # clip_portion should be 2 * clip_dim
    if clip_portion % 2 != 0:
        debug_print(f"[WARNING] Unexpected clip_portion={clip_portion}, treating as 0")
        return 0
    return clip_portion // 2


def _ensure_loaded() -> None:
    if _model is None:
        load_relation_model()


def _ensure_clip_model() -> None:
    global _clip_model
    if _clip_model is None and _model_clip_dim > 0:
        _clip_model = CLIPExtractor(device=_device)


def _ensure_pose_model() -> None:
    global _pose_model
    if _pose_model is None and _model_pose_dim > 0:
        _pose_model = PoseExtractor()


# ---------------------------------------------------------------------------
# Step 6 — Directionality preference helper
# ---------------------------------------------------------------------------

def _directionality_preference(candidate: Dict) -> Tuple:
    """Score tuple for direction naturalness.
    
    Higher tuple = more natural subject→object direction.
    Used during symmetric dedup to prefer animate subjects
    and semantic predicates.
    """
    subj_cats = _get_categories(candidate["subject"])
    is_animate = "animate" in subj_cats
    is_semantic = candidate["predicate"] in SEMANTIC_PREDS
    adj_conf = candidate.get("adjusted_confidence", candidate.get("confidence", 0.0))
    return (is_animate, is_semantic, adj_conf)


# ---------------------------------------------------------------------------
# Raw logits extraction (for Step 3 calibration + Step 8 debug)
# ---------------------------------------------------------------------------

def _get_raw_logits(
    subject: str,
    obj_label: str,
    box1: List[float],
    box2: List[float],
    img_w: float = 1.0,
    img_h: float = 1.0,
    image: Optional[Image.Image] = None,
) -> Tuple[Optional[torch.Tensor], List[str], int, int]:
    """Get raw logits from the MLP for a single pair, before any softmax.

    Automatically extracts union-region CLIP and pose features
    when the loaded model supports them.

    Returns:
        (logits_tensor, pred_tokens_list, subj_idx, obj_idx)
        or (None, [], -1, -1) on lookup failure.
    """
    print("[TRACE] entered _get_raw_logits")
    _ensure_loaded()

    subj_norm = normalize_label(subject)
    obj_norm  = normalize_label(obj_label)
    if subj_norm == "UNK" or obj_norm == "UNK":
        return None, [], -1, -1

    subj_idx = _label_vocab[subj_norm]
    obj_idx  = _label_vocab[obj_norm]

    subj_box = (float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3]))
    obj_box  = (float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3]))
    geo      = extract_geo_features(subj_box, obj_box, img_w, img_h)

    with torch.no_grad():
        s = torch.tensor([subj_idx], dtype=torch.long,    device=_device)
        o = torch.tensor([obj_idx],  dtype=torch.long,    device=_device)
        g = torch.tensor([geo],      dtype=torch.float32, device=_device)

        subj_emb = None
        obj_emb  = None
        union_emb = None
        pose_feat = None

        if _model_clip_dim > 0 or _model_union_dim > 0:
            if image is not None:
                _ensure_clip_model()
                if _model_clip_dim > 0:
                    subj_emb = _clip_model.extract_crop(image, subj_box).to(_device).unsqueeze(0)
                    obj_emb  = _clip_model.extract_crop(image, obj_box).to(_device).unsqueeze(0)

                if _model_union_dim > 0:
                    uemb = _clip_model.extract_union_embedding(image, subj_box, obj_box).to(_device)
                    union_emb = uemb.unsqueeze(0)
            else:
                if _model_clip_dim > 0:
                    subj_emb = torch.zeros((1, _model_clip_dim), device=_device)
                    obj_emb  = torch.zeros((1, _model_clip_dim), device=_device)
                if _model_union_dim > 0:
                    union_emb = torch.zeros((1, _model_union_dim), device=_device)

        pose_object_feat = None
        if _model_pose_dim > 0 and image is not None and subj_norm == "person":
            _ensure_pose_model()
            if PoseExtractor.is_available():
                pf = _pose_model.extract_pose_features(image, subj_box)
                if pf is not None:
                    pose_feat = pf.to(_device).unsqueeze(0)
                if _model_pose_object_dim > 0:
                    pof = _pose_model.extract_pose_object_features(image, subj_box, obj_box)
                    if pof is not None:
                        pose_object_feat = pof.to(_device).unsqueeze(0)

        if pose_feat is None and _model_pose_dim > 0:
            pose_feat = torch.zeros((1, _model_pose_dim), device=_device)
        if pose_object_feat is None and _model_pose_object_dim > 0:
            pose_object_feat = torch.zeros((1, _model_pose_object_dim), device=_device)

        raw_logits = _model(s, o, g,
                            subj_feat=subj_emb, obj_feat=obj_emb,
                            union_feat=union_emb, pose_feat=pose_feat,
                            pose_object_feat=pose_object_feat)

        # ── Apply logit adjustment ──────────────────────────────────
        pred_tokens = [_pred_vocab.token(i) for i in range(len(_pred_vocab))]
        adjusted_logits, adj_debug = apply_logit_adjustment(raw_logits[0], pred_tokens)

        # Print raw vs adjusted comparison
        if adj_debug:
            print("\n[logit_adjust] RAW vs ADJUSTED:")
            for pname, info in sorted(adj_debug.items(), key=lambda x: -x[1]["raw"]):
                marker = " ↑" if info["adjusted"] > info["raw"] else " ↓"
                print(f"  {pname:20s} raw={info['raw']:.4f}  adj={info['adjusted']:.4f}  "
                      f"adjust={info['adjustment']:.4f}{marker}")
            _print_logit_adjustment_debug(adj_debug)

        return adjusted_logits, pred_tokens, subj_idx, obj_idx


# ---------------------------------------------------------------------------
# Single-pair prediction
# ---------------------------------------------------------------------------

def predict_relation(
    subject: str,
    obj_label: str,
    box1: List[float],
    box2: List[float],
    img_w: float = 1.0,
    img_h: float = 1.0,
    threshold: float = 0.15,
    image: Optional[Image.Image] = None,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> Optional[Tuple[str, float]]:
    print("[TRACE] entered predict_relation")
    _ensure_loaded()

    subj_norm = normalize_label(subject)
    obj_norm  = normalize_label(obj_label)
    if subj_norm == "UNK" or obj_norm == "UNK":
        return None

    subj_idx = _label_vocab[subj_norm]
    obj_idx  = _label_vocab[obj_norm]

    subj_box = (float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3]))
    obj_box  = (float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3]))
    geo      = extract_geo_features(subj_box, obj_box, img_w, img_h)

    with torch.no_grad():
        s = torch.tensor([subj_idx], dtype=torch.long,    device=_device)
        o = torch.tensor([obj_idx],  dtype=torch.long,    device=_device)
        g = torch.tensor([geo],      dtype=torch.float32, device=_device)

        subj_emb = None; obj_emb = None; union_emb = None; pose_feat = None

        if _model_clip_dim > 0 or _model_union_dim > 0:
            if image is not None:
                _ensure_clip_model()
                if _model_clip_dim > 0:
                    subj_emb = _clip_model.extract_crop(image, subj_box).to(_device).unsqueeze(0)
                    obj_emb  = _clip_model.extract_crop(image, obj_box).to(_device).unsqueeze(0)
                if _model_union_dim > 0:
                    uemb = _clip_model.extract_union_embedding(image, subj_box, obj_box).to(_device)
                    union_emb = uemb.unsqueeze(0)
            else:
                if _model_clip_dim > 0:
                    subj_emb = torch.zeros((1, _model_clip_dim), device=_device)
                    obj_emb  = torch.zeros((1, _model_clip_dim), device=_device)
                if _model_union_dim > 0:
                    union_emb = torch.zeros((1, _model_union_dim), device=_device)

        pose_object_feat = None
        if _model_pose_dim > 0 and image is not None and subj_norm == "person":
            _ensure_pose_model()
            if PoseExtractor.is_available():
                pf = _pose_model.extract_pose_features(image, subj_box)
                if pf is not None:
                    pose_feat = pf.to(_device).unsqueeze(0)
                if _model_pose_object_dim > 0:
                    pof = _pose_model.extract_pose_object_features(image, subj_box, obj_box)
                    if pof is not None:
                        pose_object_feat = pof.to(_device).unsqueeze(0)

        if pose_feat is None and _model_pose_dim > 0:
            pose_feat = torch.zeros((1, _model_pose_dim), device=_device)
        if pose_object_feat is None and _model_pose_object_dim > 0:
            pose_object_feat = torch.zeros((1, _model_pose_object_dim), device=_device)

        # Print pose-object features debug
        if pose_object_feat is not None and _model_pose_object_dim > 0:
            pof = pose_object_feat[0]
            debug_print(f"[pose_features] {subj_norm} + {obj_label}")
            debug_print(f"  wrist_to_obj_center={pof[0].item():.4f}")
            debug_print(f"  ankle_to_obj_center={pof[1].item():.4f}")
            debug_print(f"  hip_to_obj_bottom={pof[2].item():.4f}")
            debug_print(f"  shoulder_to_obj_center={pof[3].item():.4f}")
            debug_print(f"  knee_to_obj_center={pof[4].item():.4f}")
            debug_print(f"  head_to_obj_center={pof[5].item():.4f}")
            debug_print(f"  limb_overlap={pof[6].item():.4f}")

        raw_logits = _model(s, o, g,
                            subj_feat=subj_emb, obj_feat=obj_emb,
                            union_feat=union_emb, pose_feat=pose_feat,
                            pose_object_feat=pose_object_feat)

        # ── Apply logit adjustment ──────────────────────────────────
        pred_tokens = [_pred_vocab.token(i) for i in range(len(_pred_vocab))]
        logits, _ = apply_logit_adjustment(raw_logits, pred_tokens)
        logits = logits.unsqueeze(0)  # (1, num_predicates)

        # Step 3 — Temperature-calibrated softmax
        probs  = _calibrate_scores(logits, temperature=temperature)
        top_prob, top_idx = probs[0].max(dim=-1)

    confidence = top_prob.item()
    if confidence < threshold:
        return None

    pred_token = _pred_vocab.token(top_idx.item())
    if pred_token in (Vocab.PAD, Vocab.UNK):
        return None
    if pred_token not in ALLOWED_PREDICATES:
        return None
    return (pred_token, confidence)


def predict_relation_topk(
    subject: str,
    obj_label: str,
    box1: List[float],
    box2: List[float],
    img_w: float = 1.0,
    img_h: float = 1.0,
    k: int = 3,
    image: Optional[Image.Image] = None,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> List[Tuple[str, float]]:
    print("[TRACE] entered predict_relation_topk")
    _ensure_loaded()

    subj_norm = normalize_label(subject)
    obj_norm  = normalize_label(obj_label)
    if subj_norm == "UNK" or obj_norm == "UNK":
        return []

    subj_idx = _label_vocab[subj_norm]
    obj_idx  = _label_vocab[obj_norm]
    geo = extract_geo_features(
        (float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3])),
        (float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3])),
        img_w, img_h,
    )

    with torch.no_grad():
        s = torch.tensor([subj_idx], dtype=torch.long,    device=_device)
        o = torch.tensor([obj_idx],  dtype=torch.long,    device=_device)
        g = torch.tensor([geo],      dtype=torch.float32, device=_device)

        subj_emb = None; obj_emb = None; union_emb = None; pose_feat = None

        if _model_clip_dim > 0 or _model_union_dim > 0:
            if image is not None:
                subj_box = (float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3]))
                obj_box  = (float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3]))
                _ensure_clip_model()
                if _model_clip_dim > 0:
                    subj_emb = _clip_model.extract_crop(image, subj_box).to(_device).unsqueeze(0)
                    obj_emb  = _clip_model.extract_crop(image, obj_box).to(_device).unsqueeze(0)
                if _model_union_dim > 0:
                    uemb = _clip_model.extract_union_embedding(image, subj_box, obj_box).to(_device)
                    union_emb = uemb.unsqueeze(0)
            else:
                if _model_clip_dim > 0:
                    subj_emb = torch.zeros((1, _model_clip_dim), device=_device)
                    obj_emb  = torch.zeros((1, _model_clip_dim), device=_device)
                if _model_union_dim > 0:
                    union_emb = torch.zeros((1, _model_union_dim), device=_device)

        pose_object_feat = None
        if _model_pose_dim > 0 and image is not None and normalize_label(subject) == "person":
            _ensure_pose_model()
            if PoseExtractor.is_available():
                pf = _pose_model.extract_pose_features(image, subj_box)
                if pf is not None:
                    pose_feat = pf.to(_device).unsqueeze(0)
                if _model_pose_object_dim > 0:
                    pof = _pose_model.extract_pose_object_features(image, subj_box, obj_box)
                    if pof is not None:
                        pose_object_feat = pof.to(_device).unsqueeze(0)

        if pose_feat is None and _model_pose_dim > 0:
            pose_feat = torch.zeros((1, _model_pose_dim), device=_device)
        if pose_object_feat is None and _model_pose_object_dim > 0:
            pose_object_feat = torch.zeros((1, _model_pose_object_dim), device=_device)

        raw_logits = _model(s, o, g,
                            subj_feat=subj_emb, obj_feat=obj_emb,
                            union_feat=union_emb, pose_feat=pose_feat,
                            pose_object_feat=pose_object_feat)

        # ── Apply logit adjustment ──────────────────────────────────
        pred_tokens = [_pred_vocab.token(i) for i in range(len(_pred_vocab))]
        logits, _ = apply_logit_adjustment(raw_logits, pred_tokens)

        # Step 3 — Temperature-calibrated softmax
        probs = _calibrate_scores(logits, temperature=temperature)[0]

    topk_probs, topk_idxs = probs.topk(min(k, len(probs)))
    results = []
    for prob, idx in zip(topk_probs.tolist(), topk_idxs.tolist()):
        token = _pred_vocab.token(idx)
        if token not in (Vocab.PAD, Vocab.UNK):
            results.append((token, prob))
    return results


# ---------------------------------------------------------------------------
# Drop-in replacement for utils.causal_caption.infer_relationships
# ---------------------------------------------------------------------------

Detection    = Dict
Relationship = Tuple[str, str, str]


def infer_relationships_learned(
    detections: List[Detection],
    threshold: float = 0.15,
    img_w: float = 1.0,
    img_h: float = 1.0,
    top_k: int = 2,
    image: Optional[Image.Image] = None,
) -> List[Relationship]:
    """
    Drop-in replacement for utils.causal_caption.infer_relationships().

    When using a visual-semantic model (trained with clip_dim > 0),
    pass the image to enable CLIP feature extraction from crops.

    Args:
        detections: [{"label": str, "box": [x1,y1,x2,y2], "score": float}, ...]
        threshold:  Minimum model confidence to include a relationship.
        img_w:      Image width for normalisation.
        img_h:      Image height for normalisation.
        top_k:      Maximum number of relations to return (default 2).
        image:      PIL Image (required for visual-semantic model).

    Returns:
        [(subject_label, predicate, object_label), ...]
    """
    print("[TRACE] entered infer_relationships_learned")
    if len(detections) < 2:
        return []

    candidates: List[Tuple[Relationship, float]] = []
    seen: set = set()

    n = len(detections)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            a = detections[i]
            b = detections[j]

            subj_norm = normalize_label(a["label"])
            obj_norm  = normalize_label(b["label"])
            if subj_norm == "UNK" or obj_norm == "UNK":
                continue

            result = predict_relation(
                subj_norm, obj_norm,
                a["box"], b["box"],
                img_w=img_w, img_h=img_h,
                threshold=threshold,
                image=image,
            )
            if result is None:
                continue

            pred, confidence = result
            triple = (subj_norm, pred, obj_norm)
            if triple not in seen:
                seen.add(triple)
                candidates.append((triple, confidence))

    if not candidates:
        return []

    candidates.sort(key=lambda x: -x[1])

    seen_pair: set = set()
    deduped: List[Relationship] = []
    for triple, _ in candidates:
        s, p, o = triple
        pair_key = (s, o)
        if pair_key not in seen_pair:
            seen_pair.add(pair_key)
            deduped.append(triple)

    final = deduped[:top_k]

    total_candidates = len(candidates)
    total_deduped    = len(deduped)
    discarded = [t for t in deduped if t not in final]
    debug_print(f"[relation_prediction] infer_relationships_learned: "
                f"{total_candidates} candidates -> {total_deduped} after dedup "
                f"-> {len(final)} selected (top_k={top_k})")
    if discarded:
        debug_print(f"[relation_prediction]   discarded: "
                    f"{[f'{s} {p} {o}' for s, p, o in discarded]}")
    debug_print(f"[relation_prediction]   final relations: "
                f"{[f'{s} {p} {o}' for s, p, o in final]}")

    return final


def infer_relationships_structured(
    detections: List[Detection],
    threshold: float = 0.1,
    img_w: float = 1.0,
    img_h: float = 1.0,
    top_k: int = 3,
    image: Optional[Image.Image] = None,
    semantic_boost: bool = True,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> List[Dict]:
    """
    Structured relation inference with confidence scores.

    Returns structured dicts with subject, predicate, object, confidence.
    Prioritises semantic predicates (riding, holding, wearing, carrying, looking at)
    over spatial relations.

    Args:
        detections: [{"label": str, "box": [x1,y1,x2,y2], "score": float}, ...]
        threshold:  Minimum model confidence.
        img_w:      Image width.
        img_h:      Image height.
        top_k:      Max relations to return.
        image:      PIL Image (required for visual-semantic model).
        semantic_boost: When True, applies lower threshold for semantic predicates.

    Returns:
        [{"subject": str, "predicate": str, "object": str, "confidence": float,
          "subject_box": [x1,y1,x2,y2], "object_box": [x1,y1,x2,y2]}, ...]
    """
    print("[TRACE] entered infer_relationships_structured")
    SEMANTIC_PREDS = frozenset({
        "riding", "holding", "wearing", "carrying", "looking at",
    })

    if len(detections) < 2:
        return []

    candidates: List[Dict] = []
    seen: set = set()

    n = len(detections)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            a = detections[i]
            b = detections[j]

            subj_norm = normalize_label(a["label"])
            obj_norm  = normalize_label(b["label"])
            if subj_norm == "UNK" or obj_norm == "UNK":
                continue

            result = predict_relation(
                subj_norm, obj_norm,
                a["box"], b["box"],
                img_w=img_w, img_h=img_h,
                threshold=threshold,
                image=image,
                temperature=temperature,
            )
            if result is None:
                continue

            pred, confidence = result
            triple = (subj_norm, pred, obj_norm)
            if triple not in seen:
                seen.add(triple)
                candidates.append({
                    "subject": subj_norm,
                    "predicate": pred,
                    "object": obj_norm,
                    "confidence": round(confidence, 4),
                    "subject_box": a["box"],
                    "object_box": b["box"],
                    "subject_score": a.get("score", 1.0),
                    "object_score": b.get("score", 1.0),
                })

    if not candidates:
        return []

    # Sort by confidence, but boost semantic predicates
    if semantic_boost:
        def _sort_key(x: Dict) -> float:
            boost = 0.2 if x["predicate"] in SEMANTIC_PREDS else 0.0
            return x["confidence"] + boost
        candidates.sort(key=_sort_key, reverse=True)
    else:
        candidates.sort(key=lambda x: x["confidence"], reverse=True)

    # Deduplicate by (subject, object) pair, keep highest confidence
    seen_pair: set = set()
    deduped: List[Dict] = []
    for c in candidates:
        pair_key = (c["subject"], c["object"])
        if pair_key not in seen_pair:
            seen_pair.add(pair_key)
            deduped.append(c)

    final = deduped[:top_k]

    total_candidates = len(candidates)
    total_deduped    = len(deduped)
    debug_print(f"[relation_prediction] infer_relationships_structured: "
                f"{total_candidates} candidates -> {total_deduped} after dedup "
                f"-> {len(final)} selected (top_k={top_k})")
    for r in final:
        debug_print(f"[relation_prediction]   {r['subject']} {r['predicate']} {r['object']} "
                    f"(conf={r['confidence']:.3f})")

    return final


# ---------------------------------------------------------------------------
# Semantic relation inference (Steps 1, 4, 5)
# ---------------------------------------------------------------------------

SEMANTIC_PREDS: frozenset = frozenset({
    "riding", "holding", "carrying", "wearing", "looking at",
    "sitting on", "standing on",
})

WEAK_SPATIAL: frozenset = frozenset({
    "under", "above", "over", "inside", "next to", "near",
    "attached to", "behind", "in front of", "covering",
})

NEUTRAL_SPATIAL: frozenset = frozenset({
    "on", "in",
})

_RERANK_GAP: float = 0.15
_TOP_K_CANDIDATES: int = 6

# ── Precision filtering thresholds (Steps 1, 5) ────────────────────────
# These transform the pipeline from "many noisy relations" to
# "small set of high-confidence grounded semantic constraints".
#
# MIN_SEMANTIC_SCORE: minimum adjusted_confidence for non-semantic predicates.
#                     Semantic predicates (riding, holding, etc.) bypass this
#                     because they have their own prior-based scoring.
MIN_SEMANTIC_SCORE: float = 0.30

# Weak spatial predicates need extremely high confidence to survive.
# This suppresses relations like "near", "next to", "behind", etc.
# unless the model is very certain.
WEAK_SPATIAL_THRESHOLD: float = 0.85

# Spatial relations between two inanimate objects are always rejected.
# This eliminates pure geometry clutter like "chair near chair"
# or "book under table" that have no semantic meaning.
REJECT_INANIMATE_SPATIAL: bool = True


# ── Improved semantic priors (Phase 7) ──────────────────────────────
_IMPROVED_PRIORS = {
    "sitting on": {
        "reject_objects": {"dog", "cat", "bird", "cell phone", "bottle", "cup",
                           "book", "frisbee", "kite", "baseball bat",
                           "tennis racket", "sports ball"},
        "soft_reject_penalty": -0.35,
    },
    "riding": {
        "reject_objects": {"chair", "couch", "bench", "bed", "dining table",
                           "dog", "cat", "cell phone", "bottle", "cup", "book",
                           "frisbee", "kite", "backpack", "handbag", "suitcase",
                           "tie", "umbrella", "sports ball"},
        "soft_reject_penalty": -0.40,
    },
    "holding": {
        "reject_objects": {"chair", "couch", "bench", "bed", "dining table",
                           "car", "bicycle", "truck", "bus", "train",
                           "horse", "elephant", "cow", "toilet", "sink",
                           "refrigerator"},
        "soft_reject_penalty": -0.30,
    },
    "wearing": {
        "reject_objects": {"chair", "couch", "bench", "bed", "dining table",
                           "car", "bicycle", "bus", "truck", "train",
                           "cell phone", "bottle", "cup", "book",
                           "dog", "cat", "horse", "cow", "elephant",
                           "frisbee", "kite", "sports ball", "skateboard",
                           "surfboard", "toilet", "sink", "refrigerator"},
        "soft_reject_penalty": -0.35,
    },
    "carrying": {
        "reject_objects": {"chair", "couch", "bench", "bed", "dining table",
                           "car", "bicycle", "bus", "truck", "train",
                           "toilet", "sink", "refrigerator",
                           "horse", "cow", "elephant"},
        "soft_reject_penalty": -0.25,
    },
    "standing on": {
        "reject_objects": {"cell phone", "bottle", "cup", "book",
                           "dog", "cat", "bird", "frisbee", "kite",
                           "sports ball"},
        "soft_reject_penalty": -0.30,
    },
}


def _apply_refined_priors(subject: str, predicate: str, object: str) -> float:
    """Apply object-level soft semantic penalties (Phase 7)."""
    prior = _IMPROVED_PRIORS.get(predicate)
    if not prior:
        return 0.0
    obj_lower = object.lower().replace("_", " ")
    reject_set = prior.get("reject_objects", set())
    if obj_lower in reject_set:
        return prior.get("soft_reject_penalty", -0.25)
    return 0.0


def infer_relationships_semantic(
    detections: List[Detection],
    threshold: float = 0.05,
    img_w: float = 1.0,
    img_h: float = 1.0,
    top_k: int = 3,
    image: Optional[Image.Image] = None,
    temperature: float = _DEFAULT_TEMPERATURE,
    debug: bool = True,
    improved_priors: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Calibrated semantic relation inference with lightweight priors (Steps 3-6).

    Full pipeline per pair:
      1. Extract raw logits from trained MLP (no softmax)
      2. Apply temperature-scaled softmax → calibrated scores
      3. Compute semantic prior adjustment per predicate
      4. Combine: final = calibrated_score + prior_adjustment
      5. Filter extreme nonsense (inanimate subjects with animate-requiring preds)
      6. Directionality-aware symmetric dedup

    Args:
        detections: [{"label", "box", "score"}, ...]
        threshold:  Minimum final score to include a relation.
        img_w:      Image width.
        img_h:      Image height.
        top_k:      Maximum relations to return.
        image:      PIL Image for CLIP crop extraction.
        temperature: Softmax temperature for calibration (default 2.0).
        debug:      Print detailed per-pair debug info.

    Returns:
        (filtered_relations, debug_info)
    """
    print("[TRACE] entered infer_relationships_semantic")
    if len(detections) < 2:
        print("[EARLY RETURN] infer_relationships_semantic: less than 2 detections")
        return [], []

    print("[PAIRS GENERATED] Starting pair loop")
    all_pred_tokens: List[str] = []
    raw_debug: List[Dict] = []
    candidates: List[Dict] = []
    n = len(detections)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            a = detections[i]
            b = detections[j]

            subj_norm = normalize_label(a["label"])
            obj_norm  = normalize_label(b["label"])
            if subj_norm == "UNK" or obj_norm == "UNK":
                continue

            # Step 1 — Get raw logits
            logits, pred_tokens, _, _ = _get_raw_logits(
                subj_norm, obj_norm,
                a["box"], b["box"],
                img_w=img_w, img_h=img_h,
                image=image,
            )
            if logits is None:
                continue

            all_pred_tokens = pred_tokens

            # TRACE: Pair trace header
            print(f"[PAIRS GENERATED] {subj_norm} + {obj_norm}")
            if debug:
                debug_print(f"\n{'='*50}")
                debug_print(f"PAIR TRACE: {subj_norm} + {obj_norm}")
                debug_print(f"{'='*50}")

            # Step 1b — DEBUG: Print adjusted logits BEFORE softmax/temperature/priors
            if debug:
                debug_print(f"[adjusted logits (after logit_debias)]")
                sorted_raw = sorted(
                    [(pred_tokens[i], logits[i].item()) for i in range(len(pred_tokens))
                     if pred_tokens[i] not in (Vocab.PAD, Vocab.UNK)
                     and pred_tokens[i] in ALLOWED_PREDICATES],
                    key=lambda x: -x[1],
                )
                for pname, rval in sorted_raw:
                    marker = ""
                    if pname in SEMANTIC_PREDS:
                        marker = " [SEM]"
                    elif pname in WEAK_SPATIAL:
                        marker = " [WEAK_SP]"
                    elif pname in NEUTRAL_SPATIAL:
                        marker = " [NEUTRAL_SP]"
                    debug_print(f"  {pname:20s}: {rval:8.4f}{marker}")

                # Logit statistics
                raw_vals = torch.tensor([v for _, v in sorted_raw])
                debug_print(f"  \u2500\u2500 ADJUSTED LOGIT STATS \u2500\u2500")
                debug_print(f"    max={raw_vals.max():.4f}  min={raw_vals.min():.4f}  "
                            f"mean={raw_vals.mean():.4f}  std={raw_vals.std():.4f}")
                if len(raw_vals) >= 2:
                    top2_values, top2_indices = raw_vals.topk(2)
                    gap = (top2_values[0] - top2_values[1]).item()
                    debug_print(f"    top1-top2 gap={gap:.4f}")
                    debug_print(f"    top1={sorted_raw[0][0]} ({sorted_raw[0][1]:.4f})  "
                                f"top2={sorted_raw[1][0]} ({sorted_raw[1][1]:.4f})")

                # TRACE: Top-K predicates by adjusted logit
                print(f"[ADJUSTED TOP PREDICATES] {subj_norm} + {obj_norm}:")
                for pname, rval in sorted_raw[:6]:
                    print(f"  {pname}: {rval:.4f}")
                debug_print(f"[top-k by adjusted logit]")
                top_n_raw = sorted_raw[:6]
                top_n_set = {p for p, _ in top_n_raw}
                for pname, rval in sorted_raw:
                    if pname in top_n_set:
                        debug_print(f"  \u2713 kept:     {pname:20s} ({rval:.4f})")
                    else:
                        debug_print(f"  \u2717 removed:  {pname:20s} ({rval:.4f})")

                # TRACE: Predicate type breakdown
                sem_preds = [p for p, _ in sorted_raw if p in SEMANTIC_PREDS]
                spat_preds = [p for p, _ in sorted_raw if p not in SEMANTIC_PREDS]
                debug_print(f"[predicate type breakdown]")
                debug_print(f"  semantic ({len(sem_preds)}): {sem_preds[:5]}")
                debug_print(f"  spatial  ({len(spat_preds)}): {spat_preds[:5]}")

            # Step 2 — Temperature-calibrated softmax
            calibrated = _calibrate_scores(logits.unsqueeze(0), temperature=temperature)[0]

            # Step 4 — DEBUG: Temperature scaling comparison
            if debug and temperature != 1.0:
                valid_mask = torch.tensor([
                    0 if pred_tokens[i] in (Vocab.PAD, Vocab.UNK) or pred_tokens[i] not in ALLOWED_PREDICATES
                    else 1 for i in range(len(pred_tokens))
                ], device=logits.device)
                valid_logits = logits.clone()
                valid_logits[valid_mask == 0] = -float('inf')
                top3_raw = valid_logits.argsort(descending=True)[:3]
                debug_print(f"  \u2500\u2500 TEMPERATURE SCALING ({subj_norm}+{obj_norm}) \u2500\u2500")
                for T in [1.0, 2.0, 5.0, 10.0]:
                    scaled = _calibrate_scores(logits.unsqueeze(0), temperature=T)[0]
                    probs_str = "  ".join([f"{pred_tokens[i.item()]}: {scaled[i].item():.6f}" for i in top3_raw])
                    debug_print(f"    T={T:.1f}:  {probs_str}")

            # Step 3+4 — Compute prior adjustments and final scores
            per_pred_debug: List[Dict] = []
            best_pred = None
            best_final = -float("inf")
            best_calib = 0.0
            best_prior_total = 0.0
            best_prior_idx = -1

            for pidx, pred_name in enumerate(pred_tokens):
                if pred_name in (Vocab.PAD, Vocab.UNK):
                    continue
                if pred_name not in ALLOWED_PREDICATES:
                    continue

                calib_score = calibrated[pidx].item()
                bonus, penalty, prior_total = _compute_prior_adjustment(subj_norm, pred_name, obj_norm)
                if improved_priors:
                    refined_penalty = _apply_refined_priors(subj_norm, pred_name, obj_norm)
                    penalty += refined_penalty
                    prior_total = bonus + penalty
                    # TRACE: Phase 7 refined priors
                    if debug and refined_penalty != 0.0:
                        debug_print(f"  [phase 7] {pred_name}: refined_prior={refined_penalty:+.2f}")
                final_score = calib_score + prior_total

                # TRACE: Pre-commonsense score
                pre_cs_score = final_score

                # ── PART 1 — Commonsense affordance penalty ──────────────
                cs_penalty, cs_reason = _compute_commonsense_penalty(pred_name, obj_norm)
                if cs_penalty < 1.0:
                    if debug:
                        debug_print(f"  [commonsense] {pred_name}:")
                        debug_print(f"    factor={cs_penalty:.2f}  reason={cs_reason}")
                        debug_print(f"    pre_penalty={pre_cs_score:.4f}  post_penalty={pre_cs_score * cs_penalty:.4f}")
                    final_score *= cs_penalty
                else:
                    if debug:
                        debug_print(f"  [commonsense] {pred_name}: factor=1.0 (no penalty)")

                # TRACE: Per-predicate final summary
                if debug:
                    debug_print(f"  [{pred_name:15s}] calib={calib_score:.4f}  prior={prior_total:+.4f}  "
                                f"cs={cs_penalty:.2f}  final={final_score:.4f}")

                per_pred_debug.append({
                    "predicate": pred_name,
                    "calibrated": round(calib_score, 4),
                    "prior_bonus": round(bonus, 4),
                    "prior_penalty": round(penalty, 4),
                    "prior_total": round(prior_total, 4),
                    "commonsense_penalty": round(cs_penalty, 4),
                    "final": round(final_score, 4),
                })

                if final_score > best_final:
                    best_final = final_score
                    best_pred = pred_name
                    best_calib = calib_score
                    best_prior_total = prior_total
                    best_prior_idx = pidx

            if best_pred is None:
                if debug:
                    debug_print(f"[diagnostic] \u2717 No valid predicates survived for: {subj_norm} + {obj_norm}")
                    debug_print(f"[diagnostic]   All predicates were filtered out in scoring loop")
                continue

            # TRACE: Best candidate before filtering
            if debug:
                debug_print(f"[pre-filter] Best candidate: {subj_norm} {best_pred} {obj_norm}")
                debug_print(f"  calib={best_calib:.4f}  prior={best_prior_total:+.4f}  final={best_final:.4f}")

            # ═══════════════════════════════════════════════════════════
            # STEP 2+7 — Semantic consistency override
            # ═══════════════════════════════════════════════════════════
            if debug:
                debug_print(f"\n[consistency filter]")
            # If a semantic predicate has reasonable support and is close
            # to the best spatial predicate, prefer the semantic one.
            # This strengthens "riding" over "on" for person+bicycle,
            # "sitting on" over "on" for person+chair, etc.
            #
            # Uses TWO complementary checks:
            # 1. Calibrated score margin (works when T spreads probability)
            # 2. Raw logit relative support (works when T can't spread)
            _best_sem_name = None
            _best_sem_final = -float('inf')
            _best_spatial_final = -float('inf')
            for pp in per_pred_debug:
                pf = pp["final"]
                if pp["predicate"] in SEMANTIC_PREDS:
                    if pf > _best_sem_final:
                        _best_sem_final = pf
                        _best_sem_name = pp["predicate"]
                else:
                    if pf > _best_spatial_final:
                        _best_spatial_final = pf

            if debug:
                debug_print(f"  best_sem={_best_sem_name} ({_best_sem_final:.4f})  "
                            f"best_spatial=({_best_spatial_final:.4f})")

            # Check 1: Calibrated score margin
            consistency_triggered = False
            if (best_pred is not None and
                best_pred not in SEMANTIC_PREDS and
                _best_sem_name is not None and
                _best_sem_final >= _SEMANTIC_CANDIDATE_THRESHOLD and
                _best_sem_final >= _best_spatial_final - _SEMANTIC_CONSISTENCY_MARGIN):
                consistency_triggered = True
                if debug:
                    debug_print(f"  [check 1] calibrated margin: sem={_best_sem_final:.3f} >= "
                                f"spatial={_best_spatial_final:.3f} - margin={_SEMANTIC_CONSISTENCY_MARGIN}")

            # Check 2: Raw logit relative support (handles extreme saturation)
            if not consistency_triggered and best_pred is not None and _best_sem_name is not None:
                # ONLY apply when subject is animate — inanimate subjects
                # should not get semantic predicates (caught by extreme nonsense)
                if "animate" in _get_categories(subj_norm):
                    raw_logit_values = []
                    for pidx, pname in enumerate(pred_tokens):
                        if pname in ALLOWED_PREDICATES and pname not in (Vocab.PAD, Vocab.UNK):
                            raw_logit_values.append((pname, logits[pidx].item()))
                    if raw_logit_values:
                        max_raw_logit = max(v for _, v in raw_logit_values)
                        best_raw_spatial = 0.0
                        best_raw_semantic = 0.0
                        for pname, rval in raw_logit_values:
                            if max_raw_logit > 0:
                                relative = rval / max_raw_logit
                            else:
                                relative = 0.0
                            if pname in SEMANTIC_PREDS and relative > best_raw_semantic:
                                best_raw_semantic = relative
                            elif pname not in SEMANTIC_PREDS and relative > best_raw_spatial:
                                best_raw_spatial = relative

                        # Trigger if semantic has strong relative support vs spatial
                        if (best_raw_semantic > 0.3 and
                            best_raw_semantic >= best_raw_spatial * 0.5):
                            consistency_triggered = True
                            if debug:
                                debug_print(f"  [check 2] raw logit: sem_relative={best_raw_semantic:.3f} >= "
                                            f"spatial_relative={best_raw_spatial:.3f} * 0.5")

            if debug and not consistency_triggered:
                if _best_sem_name is not None:
                    debug_print(f"  \u2717 consistency NOT triggered "
                                f"(sem={_best_sem_final:.3f} thresh={_SEMANTIC_CANDIDATE_THRESHOLD} "
                                f"spatial={_best_spatial_final:.3f})")
                else:
                    debug_print(f"  \u2717 consistency NOT triggered (no semantic candidate)")

            if consistency_triggered:
                # Override: use semantic predicate instead of spatial
                if debug:
                    debug_print(f"  \u2713 OVERRIDE: {subj_norm} {best_pred} {obj_norm} \u2192 "
                                f"{_best_sem_name} "
                                f"(sem final={_best_sem_final:.3f} vs spatial final={best_final:.3f})")
                best_pred = _best_sem_name
                for pp in per_pred_debug:
                    if pp["predicate"] == best_pred:
                        best_final = pp["final"]
                        best_calib = pp["calibrated"]
                        best_prior_total = pp["prior_total"]
                        break

            # TRACE: Post-consistency state
            if debug:
                debug_print(f"[post-consistency] Best: {subj_norm} {best_pred} {obj_norm} final={best_final:.4f}")

            # ═══════════════════════════════════════════════════════════
            # Pairwise plausibility check (after logit adjustment,
            # before final selection)
            # ═══════════════════════════════════════════════════════════
            plausible, plaus_reason = _check_pairwise_plausibility(
                subj_norm, best_pred, obj_norm,
                adjusted_confidence=best_final,
                debug=debug,
            )
            if not plausible:
                if debug:
                    debug_print(f"[commonsense] Rejected: {subj_norm} {best_pred} {obj_norm}")
                    debug_print(f"  reason: {plaus_reason}")
                print(f"[commonsense] Rejected: {subj_norm} {best_pred} {obj_norm}")
                print(f"  reason: {plaus_reason}")
                raw_debug.append({
                    "subject": subj_norm,
                    "object": obj_norm,
                    "status": "rejected_pairwise_plausibility",
                    "best_predicate": best_pred,
                    "reason": plaus_reason,
                    "per_predicate": per_pred_debug,
                })
                continue
            if debug:
                debug_print(f"[commonsense] \u2713 plausible: {subj_norm} {best_pred} {obj_norm}")

            # Step 5 — Filter extreme nonsense
            print(f"[TRACE] extreme_nonsense_filter: checking {subj_norm} {best_pred} {obj_norm}")
            if debug:
                debug_print(f"[extreme nonsense filter]")
            if _is_extreme_nonsense(subj_norm, best_pred, obj_norm):
                if debug:
                    debug_print(f"  \u2717 REMOVED: {subj_norm} {best_pred} {obj_norm} (extreme nonsense)")
                raw_debug.append({
                    "subject": subj_norm,
                    "object": obj_norm,
                    "status": "rejected_extreme_nonsense",
                    "best_predicate": best_pred,
                    "per_predicate": per_pred_debug,
                })
                continue
            if debug:
                debug_print(f"  \u2713 passed")

            # Threshold check (global minimum)
            if best_final < threshold:
                if debug:
                    debug_print(f"[threshold filter] \u2717 REMOVED: {subj_norm} {best_pred} {obj_norm} "
                                f"(final={best_final:.4f} < threshold={threshold})")
                continue
            if debug:
                debug_print(f"[threshold filter] \u2713 passed (final={best_final:.4f} >= {threshold})")

            # ═══════════════════════════════════════════════════════════
            # PART 1+4 — Predicate-aware adaptive calibration
            # ═══════════════════════════════════════════════════════════
            # Uses predicate family, confidence-dependent relaxation, and
            # competitor family awareness to set adaptive thresholds.
            print(f"[TRACE] calibration_stage: {subj_norm} {best_pred} {obj_norm}")
            if debug:
                debug_print(f"[calibration filter] (adaptive)")
            valid_preds = [
                (pp["predicate"], pp["final"])
                for pp in per_pred_debug
                if pp["predicate"] not in (Vocab.PAD, Vocab.UNK)
            ]
            valid_preds.sort(key=lambda x: -x[1])

            # TRACE: Full sorted predicate list
            if debug:
                debug_print(f"  all predicates by final score:")
                for rank, (pn, pv) in enumerate(valid_preds[:10]):
                    family_tag = _get_predicate_family(pn) or "?"
                    debug_print(f"    #{rank+1}: {pn:20s} = {pv:.4f}  [{family_tag}]")
                if len(valid_preds) > 10:
                    debug_print(f"    ... ({len(valid_preds) - 10} more)")

            calibration_rejected = False
            top1_name = best_pred
            top1_score_val = best_final
            margin = 0.0
            top2_name: Optional[str] = None
            top2_score_val: Optional[float] = None
            calib_debug_info: Optional[Dict] = None

            if len(valid_preds) >= 2:
                top1_name, top1_score_val = valid_preds[0]
                top2_name, top2_score_val = valid_preds[1]
                margin = top1_score_val - top2_score_val

                # ── Apply predicate-aware adaptive calibration ────────
                accepted, reject_reason, calib_debug_info = _compute_adaptive_calibration(
                    top1_name, top1_score_val,
                    top2_name, top2_score_val, margin,
                    debug=debug,
                )

                if not accepted:
                    calibration_rejected = True
                    if debug:
                        print(f"  [calib] TOP-2 FAMILY: {calib_debug_info.get('top2_family')} "
                              f"(competitor_factor={calib_debug_info.get('competitor_factor', 'N/A')})")
                    raw_debug.append({
                        "subject": subj_norm,
                        "object": obj_norm,
                        "status": "rejected_calibration",
                        "best_predicate": best_pred,
                        "top1": top1_name,
                        "top1_score": round(top1_score_val, 4),
                        "top2": top2_name,
                        "top2_score": round(top2_score_val, 4),
                        "margin": round(margin, 4),
                        "reject_reason": reject_reason,
                        "calib_debug": calib_debug_info,
                        "per_predicate": per_pred_debug,
                    })
                else:
                    if debug:
                        print(f"  [calib] ACCEPTED: {top1_name}")

            elif len(valid_preds) == 1:
                top1_name, top1_score_val = valid_preds[0]

                # Single predicate: use adaptive calibration with no competitor
                accepted, reject_reason, calib_debug_info = _compute_adaptive_calibration(
                    top1_name, top1_score_val,
                    None, None, 0.0,
                    debug=debug,
                )

                if not accepted:
                    calibration_rejected = True
                    raw_debug.append({
                        "subject": subj_norm,
                        "object": obj_norm,
                        "status": "rejected_calibration",
                        "best_predicate": best_pred,
                        "top1": top1_name,
                        "top1_score": round(top1_score_val, 4),
                        "reject_reason": reject_reason,
                        "calib_debug": calib_debug_info,
                        "per_predicate": per_pred_debug,
                    })
                else:
                    if debug:
                        print(f"  [calib] ACCEPTED (single pred): {top1_name}")

            elif debug:
                debug_print(f"  [calib] REJECTED: no valid predicates")

            if calibration_rejected:
                continue

            # TRACE: Post-calibration acceptance
            if debug:
                debug_print(f"[post-calibration] \u2713 ACCEPTED: {subj_norm} {top1_name} {obj_norm}")
                debug_print(f"  conf={top1_score_val:.4f}  margin={margin:.4f}")

            # TRACE: Candidate accepted
            if debug:
                debug_print(f"[candidate accepted] \u2713 {subj_norm} {best_pred} {obj_norm}")
                debug_print(f"  calib={best_calib:.4f}  prior={best_prior_total:+.4f}  "
                            f"final={best_final:.4f}  margin={margin:.4f}")
                sem_rank = sum(1 for pp in per_pred_debug if pp["predicate"] in SEMANTIC_PREDS and pp["final"] > best_final)
                spat_rank = sum(1 for pp in per_pred_debug if pp["predicate"] not in SEMANTIC_PREDS and pp["final"] > best_final)
                debug_print(f"  semantic rank: {sem_rank + 1}/{sum(1 for pp in per_pred_debug if pp['predicate'] in SEMANTIC_PREDS)}")

            candidates.append({
                "subject": subj_norm,
                "predicate": best_pred,
                "object": obj_norm,
                "confidence": round(best_calib, 4),
                "adjusted_confidence": round(best_final, 4),
                "prior_adjustment": round(best_prior_total, 4),
                "subject_box": a["box"],
                "object_box": b["box"],
                "subject_score": a.get("score", 1.0),
                "object_score": b.get("score", 1.0),
                "best_predicate_idx": best_prior_idx,
                "calibration_top1": top1_name,
                "calibration_top1_score": round(top1_score_val, 4),
                "calibration_margin": round(margin, 4),
                "calib_debug": calib_debug_info,
            })

            raw_debug.append({
                "subject": subj_norm,
                "object": obj_norm,
                "status": "candidate",
                "best_predicate": best_pred,
                "best_calibrated": round(best_calib, 4),
                "best_prior_adjustment": round(best_prior_total, 4),
                "best_final_score": round(best_final, 4),
                "calibration_top1": top1_name,
                "calibration_top1_score": round(top1_score_val, 4),
                "calibration_margin": round(margin, 4),
                "calib_debug": calib_debug_info,
                "per_predicate": per_pred_debug,
            })

    # TRACE: Empty relation diagnostic
    if debug and not candidates:
        debug_print(f"\n{'='*50}")
        debug_print(f"EMPTY RELATION DIAGNOSTIC")
        debug_print(f"{'='*50}")
        debug_print(f"No pairs produced candidates.")
        debug_print(f"Total pairs evaluated: {len(raw_debug)}")
        for rd_entry in raw_debug:
            s = rd_entry["subject"]
            o = rd_entry["object"]
            st = rd_entry["status"]
            bp = rd_entry.get("best_predicate", "N/A")
            debug_print(f"  {s}+{o}: status={st}")
            if st == "rejected_pairwise_plausibility":
                debug_print(f"    rejected by pairwise plausibility: {bp}")
                debug_print(f"    reason={rd_entry.get('reason', 'unknown')}")
            elif st == "rejected_calibration":
                debug_print(f"    top_candidate={bp}")
                debug_print(f"    reason={rd_entry.get('reject_reason', 'unknown')}")
                if "top1_score" in rd_entry:
                    debug_print(f"    conf={rd_entry['top1_score']}")
                if "margin" in rd_entry:
                    debug_print(f"    margin={rd_entry['margin']}")
            elif st == "rejected_extreme_nonsense":
                debug_print(f"    rejected as extreme nonsense: {bp}")
            pp_list = rd_entry.get("per_predicate", [])
            if pp_list:
                top_pp = sorted(pp_list, key=lambda x: -x["final"])[:3]
                for pp in top_pp:
                    debug_print(f"    candidate: {pp['predicate']:15s} calib={pp['calibrated']:.4f}  "
                                f"final={pp['final']:.4f}")
        debug_print()

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1+3+5 — Precision filtering
    # ═══════════════════════════════════════════════════════════════════
    print(f"[TRACE] precision_filter: {len(candidates)} candidates")
    if debug and candidates:
        debug_print(f"\n[precision filter] Processing {len(candidates)} candidates")
    filtered_candidates: List[Dict] = []
    precision_log: List[Dict] = []

    for c in candidates:
        pred = c["predicate"]
        subj = c["subject"]
        obj = c["object"]
        adj_conf = c.get("adjusted_confidence", c["confidence"])
        subj_cats = _get_categories(subj)
        is_animate_subj = "animate" in subj_cats

        # TRACE: Precision filter candidate
        if debug:
            debug_print(f"  checking: {subj} {pred} {obj} (adj_conf={adj_conf:.4f}, animate_subj={is_animate_subj})")

        # Step 3 — Animate subject enforcement for semantic predicates
        if pred in SEMANTIC_PREDS and not is_animate_subj:
            if debug:
                debug_print(f"    \u2717 REMOVED: semantic predicate requires animate subject")
            precision_log.append({
                "subject": subj, "predicate": pred, "object": obj,
                "adj_conf": adj_conf, "status": "rejected",
                "reason": "semantic predicate requires animate subject",
            })
            continue

        # Step 1+5 — Suppress geometry-only relations
        is_inanimate_pair = (not is_animate_subj and "animate" not in _get_categories(obj))

        if pred in WEAK_SPATIAL:
            # Reject spatial relations between two inanimates (pure clutter)
            if REJECT_INANIMATE_SPATIAL and is_inanimate_pair:
                if debug:
                    debug_print(f"    \u2717 REMOVED: spatial between inanimates (pure clutter)")
                precision_log.append({
                    "subject": subj, "predicate": pred, "object": obj,
                    "adj_conf": adj_conf, "status": "rejected",
                    "reason": "spatial relation between inanimate objects (no semantic value)",
                })
                continue
            if adj_conf < WEAK_SPATIAL_THRESHOLD:
                if debug:
                    debug_print(f"    \u2717 REMOVED: weak spatial (adj_conf={adj_conf:.3f} < {WEAK_SPATIAL_THRESHOLD})")
                precision_log.append({
                    "subject": subj, "predicate": pred, "object": obj,
                    "adj_conf": adj_conf, "status": "rejected",
                    "reason": f"weak spatial relation (adj_conf={adj_conf:.3f} < {WEAK_SPATIAL_THRESHOLD})",
                })
                continue

        # Reject ALL neutral spatial relations between inanimates
        # "on"/"in" between two inanimate objects is pure geometry clutter
        if pred in NEUTRAL_SPATIAL and is_inanimate_pair:
            if debug:
                debug_print(f"    \u2717 REMOVED: neutral spatial between inanimates")
            precision_log.append({
                "subject": subj, "predicate": pred, "object": obj,
                "adj_conf": adj_conf, "status": "rejected",
                "reason": f"neutral spatial between inanimates (no semantic value, adj_conf={adj_conf:.3f})",
            })
            continue

        # Directionality: reject spatial predicates with inanimate subject + animate object
        # This catches "bicycle on person", "book above person", etc.
        if (pred in WEAK_SPATIAL or pred in NEUTRAL_SPATIAL
                and not is_animate_subj and "animate" in _get_categories(obj)):
            if debug:
                debug_print(f"    \u2717 REMOVED: reversed direction (inanimate subj + animate obj)")
            precision_log.append({
                "subject": subj, "predicate": pred, "object": obj,
                "adj_conf": adj_conf, "status": "rejected",
                "reason": f"spatial predicate with inanimate subject + animate object "
                          f"(reversed direction, adj_conf={adj_conf:.3f})",
            })
            continue

        # Step 5 — MIN_SEMANTIC_SCORE for non-semantic predicates
        if pred not in SEMANTIC_PREDS:
            threshold_for_this = MIN_SEMANTIC_SCORE
            if adj_conf < threshold_for_this:
                if debug:
                    debug_print(f"    \u2717 REMOVED: below min semantic score "
                                f"(adj_conf={adj_conf:.3f} < {threshold_for_this})")
                precision_log.append({
                    "subject": subj, "predicate": pred, "object": obj,
                    "adj_conf": adj_conf, "status": "rejected",
                    "reason": f"below min semantic score (adj_conf={adj_conf:.3f} < {threshold_for_this})",
                })
                continue

        if debug:
            debug_print(f"    \u2713 passed precision filters")
        precision_log.append({
            "subject": subj, "predicate": pred, "object": obj,
            "adj_conf": adj_conf, "status": "kept",
            "reason": "passed precision filters",
        })
        filtered_candidates.append(c)

    candidates = filtered_candidates

    # ═══════════════════════════════════════════════════════════════════
    # STEP 7 — Semantic consistency: prefer semantic over spatial per pair
    # ═══════════════════════════════════════════════════════════════════
    print(f"[TRACE] consistency_filter: {len(candidates)} candidates")
    # For each (subject, object) pair, if there are multiple candidates
    # (shouldn't happen normally, but just in case), keep only the best
    # semantic one if scores are close.
    if len(candidates) > 1:
        pre_consistency = len(candidates)
        candidates = _apply_semantic_consistency(candidates)
        if debug and len(candidates) < pre_consistency:
            debug_print(f"\n[consistency] Semantic consistency removed "
                        f"{pre_consistency - len(candidates)} candidates")

    if not candidates:
        print("[EARLY RETURN] infer_relationships_semantic: no candidates after precision+consistency filters")
        if debug:
            debug_print(f"\n[calibration] \u2192 no strong semantic interaction detected")
            debug_print(f"[calibration] Returning NO_RELATION ({len(raw_debug)} pairs evaluated, "
                        f"0 passed calibration)")
            for pl in precision_log:
                if pl["status"] == "kept":
                    continue
                debug_print(f"  REJECTED: {pl['subject']} {pl['predicate']} {pl['object']}")
                debug_print(f"    reason: {pl['reason']}")
        return [], raw_debug

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2 — Strong semantic priority sort
    # ═══════════════════════════════════════════════════════════════════
    def _sort_key(x: Dict) -> float:
        is_animate = "animate" in _get_categories(x["subject"])
        is_sem = x["predicate"] in SEMANTIC_PREDS
        adj_conf = x.get("adjusted_confidence", x["confidence"])
        # Large gap so semantic predicates always rank above spatial
        sem_boost = 10.0 if is_sem else 0.0
        anim_boost = 5.0 if is_animate else 0.0
        # ── PART 3 — Per-predicate semantic bonus ─────────────────────
        pred_bonus = SEMANTIC_BONUSES.get(x["predicate"], 0.0)
        return anim_boost + sem_boost + pred_bonus + adj_conf

    candidates.sort(key=_sort_key, reverse=True)

    # ── PART 3+4 — Debug output for per-predicate bonuses ─────────────
    if debug:
        boosted = [c for c in candidates if SEMANTIC_BONUSES.get(c["predicate"], 0.0) > 0]
        for c in boosted[:5]:
            bonus = SEMANTIC_BONUSES.get(c["predicate"], 0.0)
            debug_print(f"  [reranking] Boost applied: "
                        f"{c['subject']} {c['predicate']} {c['object']}")
            debug_print(f"    bonus=+{bonus:.2f}")
            debug_print(f"    reason=semantic predicate")

    # Step 6 — Directionality-aware symmetric dedup
    print(f"[TRACE] dedup_stage: {len(candidates)} candidates before dedup")
    if debug:
        debug_print(f"\n[dedup] {len(candidates)} candidates before dedup")
    seen_pairs: Dict[tuple, Dict] = {}
    for c in candidates:
        pair = tuple(sorted([c["subject"], c["object"]]))
        direction_score = _directionality_preference(c)
        if pair not in seen_pairs:
            seen_pairs[pair] = c
            if debug:
                debug_print(f"  \u2713 kept: {c['subject']} {c['predicate']} {c['object']} "
                            f"(direction={direction_score})")
        else:
            existing = seen_pairs[pair]
            existing_score = _directionality_preference(existing)
            if direction_score > existing_score:
                if debug:
                    debug_print(f"  \u2717 replaced: {existing['subject']} {existing['predicate']} {existing['object']} "
                                f"(dir={existing_score}) \u2192 {c['subject']} {c['predicate']} {c['object']} "
                                f"(dir={direction_score})")
                seen_pairs[pair] = c
            else:
                if debug:
                    debug_print(f"  \u2717 dropped:  {c['subject']} {c['predicate']} {c['object']} "
                                f"(dir={direction_score}) < existing ({existing_score})")

    deduped = list(seen_pairs.values())
    if debug:
        debug_print(f"[dedup] {len(deduped)} candidates after dedup ({len(candidates) - len(deduped)} removed)")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 4 — One primary relation per animate subject
    # ═══════════════════════════════════════════════════════════════════
    print(f"[TRACE] one_per_animate_subject: {len(deduped)} candidates")
    if debug:
        debug_print(f"\n[one-per-animate] {len(deduped)} candidates")
    subject_best: Dict[str, Dict] = {}
    for c in deduped:
        subj = c["subject"]
        subj_cats = _get_categories(subj)
        if "animate" not in subj_cats:
            key = f"__{subj}__{c['object']}"
            subject_best[key] = c
            if debug:
                debug_print(f"  \u2713 (inanimate) {subj} {c['predicate']} {c['object']}")
        else:
            if subj not in subject_best:
                subject_best[subj] = c
                if debug:
                    debug_print(f"  \u2713 (animate, first) {subj} {c['predicate']} {c['object']}")
            else:
                existing = subject_best[subj]
                existing_key = (
                    existing["predicate"] in SEMANTIC_PREDS,
                    existing.get("adjusted_confidence", existing["confidence"]),
                )
                current_key = (
                    c["predicate"] in SEMANTIC_PREDS,
                    c.get("adjusted_confidence", c["confidence"]),
                )
                if current_key > existing_key:
                    if debug:
                        debug_print(f"  \u2717 replaced (animate): {existing['subject']} {existing['predicate']} {existing['object']} "
                                    f"\u2192 {c['subject']} {c['predicate']} {c['object']}")
                    subject_best[subj] = c
                else:
                    if debug:
                        debug_print(f"  \u2717 dropped (animate): {subj} {c['predicate']} {c['object']} "
                                    f"(existing {existing['predicate']} preferred)")

    selected_relations = list(subject_best.values())[:top_k]
    if debug:
        debug_print(f"[one-per-animate] {len(selected_relations)} selected (top_k={top_k})")
        for r_idx, r in enumerate(selected_relations):
            debug_print(f"  #{r_idx+1}: {r['subject']} {r['predicate']} {r['object']} "
                        f"(conf={r.get('calibration_top1_score', r['confidence']):.4f})")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 8 — Debug output: precision filter transparency
    # ═══════════════════════════════════════════════════════════════════
    if debug:
        rejected = [p for p in precision_log if p["status"] == "rejected"]
        kept = [p for p in precision_log if p["status"] == "kept"]
        if rejected:
            debug_print(f"\n[precision filter] \u2500\u2500\u2500 PRECISION FILTER \u2500\u2500\u2500")
            for pl in rejected:
                debug_print(f"  REJECTED: {pl['subject']} {pl['predicate']} {pl['object']}")
                debug_print(f"    reason: {pl['reason']}")
        if kept:
            kept_selected = [p for p in precision_log
                             if p["status"] == "kept" and
                             f"{p['subject']} {p['predicate']} {p['object']}" in
                             {f'{r["subject"]} {r["predicate"]} {r["object"]}' for r in selected_relations}]
            if kept_selected:
                debug_print(f"  KEPT:")
                for pl in kept_selected:
                    debug_print(f"  KEPT: {pl['subject']} {pl['predicate']} {pl['object']}")
                    debug_print(f"    reason: {pl['reason']}")

        # Per-pair debug
        debug_print(f"\n[calibrated] \u2500\u2500\u2500 Relation Debug \u2500\u2500\u2500")
        for rd in raw_debug:
            if rd["status"] in ("rejected_extreme_nonsense", "rejected_calibration", "rejected_pairwise_plausibility"):
                continue
            s, o = rd["subject"], rd["object"]
            debug_print(f"  {s}+{o}:")
            top_pp = sorted(rd["per_predicate"], key=lambda x: -x["final"])[:5]
            for pp in top_pp:
                p = pp["predicate"]
                cal = pp["calibrated"]
                prior = pp["prior_total"]
                final_val = pp["final"]
                marker = " \u25c0" if p == rd["best_predicate"] else ""
                debug_print(f"    {p}:")
                debug_print(f"      calibrated={cal:.4f}")
                debug_print(f"      prior={prior:+.4f}")
                debug_print(f"      final={final_val:.4f}{marker}")
            debug_print(f"  SELECTED: {rd['best_predicate']} "
                        f"(calib={rd['best_calibrated']:.4f} \u2192 "
                        f"final={rd['best_final_score']:.4f})")
            debug_print()

        # TRACE: Per-pair final outcome
        debug_print(f"\n{'='*50}")
        debug_print(f"PER-PAIR FINAL OUTCOMES")
        debug_print(f"{'='*50}")
        selected_set = {(r['subject'], r['predicate'], r['object']) for r in selected_relations}
        for rd in raw_debug:
            s, o = rd["subject"], rd["object"]
            bp = rd.get("best_predicate", "N/A")
            pair_key = (s, bp, o)
            if pair_key in selected_set:
                debug_print(f"[final]  \u2713 SELECTED: {s} {bp} {o}")
            elif rd["status"] == "rejected_pairwise_plausibility":
                debug_print(f"[final]  \u2717 REJECTED AT: pairwise plausibility  ({s} {bp} {o})")
                debug_print(f"    reason: {rd.get('reason', 'N/A')}")
            elif rd["status"] == "rejected_calibration":
                debug_print(f"[final]  \u2717 REJECTED AT: calibration filter  ({s} {bp} {o})")
            elif rd["status"] == "rejected_extreme_nonsense":
                debug_print(f"[final]  \u2717 REJECTED AT: extreme nonsense filter  ({s} {bp} {o})")
            else:
                debug_print(f"[final]  \u2717 REJECTED AT: unknown stage  ({s} {bp} {o})")
        for r in selected_relations:
            cal_margin = r.get("calibration_margin", 0.0)
            cal_conf = r.get("calibration_top1_score", r["confidence"])
            debug_print(f"  \u2192 {r['subject']} {r['predicate']} {r['object']}  "
                        f"conf={cal_conf:.4f}  margin={cal_margin:.4f}")

        debug_print(f"\n[summary] {len(raw_debug)} pairs evaluated -> "
                    f"{len(precision_log)} after precision filter -> "
                    f"{len(candidates)} after calibration+prior -> "
                    f"{len(deduped)} after dedup -> "
                    f"{len(selected_relations)} selected (top_k={top_k})")
        for r in selected_relations:
            cal_margin = r.get("calibration_margin", 0.0)
            cal_conf = r.get("calibration_top1_score", r["confidence"])
            debug_print(f"\n[calibration] \u2713 accepted:")
            debug_print(f"  {r['subject']} {r['predicate']} {r['object']}")
            debug_print(f"  conf={cal_conf:.2f} margin={cal_margin:.2f}")

        # ═══════════════════════════════════════════════════════════════
        # STEP 4 — Feature utilization analysis
        # ═══════════════════════════════════════════════════════════════
        if _model is not None:
            try:
                feature_norms = _get_feature_group_norms(_model)
                total_fn = sum(feature_norms.values()) or 1.0
                header = "FEATURE GROUP NORMS (projection weights)" if _model_type == "transformer" else "FEATURE GROUP NORMS (first layer)"
                debug_print(f"\n  \u2500\u2500\u2500 {header} \u2500\u2500\u2500")
                for name, norm in sorted(feature_norms.items(), key=lambda x: -x[1]):
                    pct = norm / total_fn * 100
                    debug_print(f"    {name:15s}: {norm:8.4f}  ({pct:5.1f}%)")
                LINE = "\u2500" * 35
                debug_print(f"    {LINE}")
                clip_pct = (feature_norms.get("subj_clip", 0) + feature_norms.get("obj_clip", 0)) / total_fn * 100
                union_pct = feature_norms.get("union_clip", 0) / total_fn * 100
                pose_pct = feature_norms.get("pose", 0) / total_fn * 100
                geo_pct = feature_norms.get("geo", 0) / total_fn * 100
                label_pct = (feature_norms.get("subj_label", 0) + feature_norms.get("obj_label", 0)) / total_fn * 100
                debug_print(f"    Subject CLIP:      {clip_pct:5.1f}%  "
                            f"{'(underused)' if clip_pct < 30 else '(active)'}")
                if _model_union_dim > 0:
                    debug_print(f"    Union-region CLIP: {union_pct:5.1f}%  "
                                f"{'(underused)' if union_pct < 10 else '(active)'}")
                if _model_pose_dim > 0:
                    debug_print(f"    Pose features:     {pose_pct:5.1f}%  "
                                f"{'(underused)' if pose_pct < 5 else '(active)'}")
                if _model_pose_object_dim > 0:
                    pose_obj_pct = feature_norms.get("pose_object", 0) / total_fn * 100
                    debug_print(f"    Pose-object:       {pose_obj_pct:5.1f}%  "
                                f"{'(active)' if pose_obj_pct > 3 else '(underused)'}")
                debug_print(f"    Geometry:          {geo_pct:5.1f}%  "
                            f"{'(dominant)' if geo_pct > 15 else '(controlled)'}")
                debug_print(f"    Label embeddings:  {label_pct:5.1f}%")

                # For transformer, also show attention-based modality usage
                if _model_type == "transformer":
                    try:
                        attn_contribs = _model.get_feature_contributions(
                            torch.tensor([0], device=_device),
                            torch.tensor([0], device=_device),
                            torch.zeros((1, GEO_DIM), device=_device),
                        )
                        if attn_contribs:
                            total_attn = sum(attn_contribs.values()) or 1.0
                            debug_print(f"\n  \u2500\u2500\u2500 ATTENTION-BASED MODALITY USAGE \u2500\u2500\u2500")
                            for name, val in sorted(attn_contribs.items(), key=lambda x: -x[1]):
                                pct = val / total_attn * 100
                                debug_print(f"    {name:15s}: {pct:5.1f}%")
                    except Exception as e2:
                        debug_print(f"  [attention analysis] ({e2})")
            except Exception as e:
                debug_print(f"  [feature analysis] Skipped ({e})")

        # ═══════════════════════════════════════════════════════════════
        # STEP 8 — Quality evaluation
        # ═══════════════════════════════════════════════════════════════
        quality = evaluate_relation_quality(selected_relations, raw_debug)
        debug_print(f"\n  \u2500\u2500\u2500 RELATION QUALITY METRICS \u2500\u2500\u2500")
        debug_print(f"    Total relations:           {quality['total_relations']}")
        debug_print(f"    Semantic precision:        {quality['semantic_precision']:.2%} "
                    f"({quality['semantic_relations']}/{quality['total_relations']})")
        debug_print(f"    Animate subject rate:      {quality['animate_subject_rate']:.2%}")
        debug_print(f"    Reversed direction rate:   {quality['reversed_direction_rate']:.2%}")
        debug_print(f"    Weak spatial rate:         {quality['weak_spatial_rate']:.2%}")
        if quality.get("predicate_breakdown"):
            debug_print(f"    Predicate breakdown:       {quality['predicate_breakdown']}")

    return selected_relations, raw_debug
