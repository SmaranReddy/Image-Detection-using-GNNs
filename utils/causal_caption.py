"""
Rule-based causal caption generator for object detection outputs.

Converts detected objects and their pairwise relationships into human-readable
causal sentences without any additional model training or external dependencies.

Typical usage (inference time only):
    from utils.causal_caption import infer_relationships, generate_causal_caption

    detections    = extract_top_detections(pred_logits, pred_boxes, class_names)
    relationships = infer_relationships(detections)
    caption       = generate_causal_caption(detections, relationships)
    print(caption)
"""

from __future__ import annotations

from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Relationship → clean verb phrase
# ---------------------------------------------------------------------------
# Each value is only the verb (or verb + preposition), never including the
# object noun.  The object is always appended separately via build_object_phrase.

RELATION_PHRASES: Dict[str, str] = {
    "riding":           "riding",
    "holding_umbrella": "holding",
    "running":          "running near",
    "near_pedestrian":  "walking near",
    "next_to":          "standing next to",
    "chasing":          "chasing",
    "carrying":         "carrying",
    "walking_with":     "walking with",
    "parked_near":      "parked near",
    "looking_at":       "looking at",
    "playing_with":     "playing with",
}

# ---------------------------------------------------------------------------
# Relationship → causal explanation (appended after a comma)
# ---------------------------------------------------------------------------

CAUSAL_EXPLANATIONS: Dict[str, str] = {
    "riding":           "likely for transportation",
    "holding_umbrella": "possibly because it is raining",
    "running":          "possibly for exercise or urgency",
    "near_pedestrian":  "indicating a potential interaction or risk",
    "next_to":          "indicating proximity or interaction",
    "chasing":          "suggesting pursuit or play",
    "carrying":         "likely transporting an object",
    "walking_with":     "suggesting companionship or guidance",
    "parked_near":      "indicating a stationary interaction",
    "looking_at":       "showing attention or interest",
    "playing_with":     "possibly for recreation or exercise",
}

# ---------------------------------------------------------------------------
# Causal reason noun phrases  (used in confidence-qualified hypothesis lines)
# ---------------------------------------------------------------------------
# Plain noun phrases — no pre-baked qualifier.  The qualifier is added at
# runtime based on detection confidence via _confidence_qualifier().

CAUSAL_REASONS: Dict[str, str] = {
    "riding":           "a need for transportation",
    "holding_umbrella": "rain or wet weather",
    "running":          "exercise or urgency",
    "near_pedestrian":  "a potential pedestrian interaction",
    "next_to":          "physical proximity or deliberate interaction",
    "chasing":          "pursuit or play behaviour",
    "carrying":         "the need to move an object",
    "walking_with":     "companionship or guidance",
    "parked_near":      "a stationary spatial relationship",
    "looking_at":       "curiosity or focused attention",
    "playing_with":     "recreational activity or exercise",
}

# ---------------------------------------------------------------------------
# Counterfactual statements  (what would be different if the cause were absent)
# ---------------------------------------------------------------------------

COUNTERFACTUALS: Dict[str, str] = {
    "riding":           "If transportation were not needed, the bicycle would likely be absent.",
    "holding_umbrella": "If there were no rain, the umbrella would likely not be in use.",
    "running":          "If there were no urgency or exercise intent, the subject would likely be stationary.",
    "near_pedestrian":  "If no interaction were occurring, the pedestrian and vehicle would likely be farther apart.",
    "next_to":          "If proximity were not required, the objects would likely appear farther apart.",
    "chasing":          "If no pursuit were occurring, the subjects would likely be separated.",
    "carrying":         "If the object did not need to be transported, it would likely remain in place.",
    "walking_with":     "If there were no companionship, the two subjects would likely move independently.",
    "parked_near":      "If no spatial relationship existed, the objects would likely be positioned elsewhere.",
    "looking_at":       "If there were no interest, the subject would likely be oriented away.",
    "playing_with":     "If no recreational activity were occurring, the ball would likely be stationary.",
}

# ---------------------------------------------------------------------------
# Evidence cue sets
# ---------------------------------------------------------------------------
# Each key names a causal context; the value lists detected labels that count
# as corroborating evidence for that context.  Labels mirror COCO class names
# used in this project.  Entries can be extended without touching any logic.

EVIDENCE_CUES: Dict[str, List[str]] = {
    "rain":        ["umbrella", "rain", "wet_road", "raincoat"],
    "transport":   ["bicycle", "car", "motorcycle", "road"],
    "interaction": ["person", "car", "crosswalk"],
    "recreation":  ["sports ball", "frisbee", "tennis racket", "kite"],
    "work":        ["suitcase", "backpack", "laptop", "handbag"],
}

# Maps each relation key → the EVIDENCE_CUES context it should be gated on.
# Relations absent from this dict fall back to confidence-only gating.

RELATION_EVIDENCE: Dict[str, str] = {
    "riding":           "transport",
    "holding_umbrella": "rain",
    "near_pedestrian":  "interaction",
    "running":          "transport",
    "parked_near":      "transport",
    "carrying":         "work",
    "playing_with":     "recreation",
}

# ---------------------------------------------------------------------------
# Sentence templates
# ---------------------------------------------------------------------------
# {A}    — capitalised indefinite article for the subject ("A" / "An")
# {a}    — lowercase indefinite article for the subject ("a" / "an")
# {subj} — cleaned subject noun
# {verb} — verb phrase from RELATION_PHRASES
# {obj}  — object noun phrase including its own article
# {cause}— causal clause ", <explanation>" or "" when absent

TEMPLATES: List[str] = [
    "{A} {subj} is {verb} {obj}{cause}.",
    "In the scene, {a} {subj} is {verb} {obj}{cause}.",
    "We observe {a} {subj} {verb} {obj}{cause}.",
]

# ---------------------------------------------------------------------------
# Object-pair → relationship rules
# ---------------------------------------------------------------------------
# Keys are frozensets so order of detection does not matter.

_PAIR_RULES: Dict[frozenset, str] = {
    frozenset({"person", "bicycle"}):     "riding",
    frozenset({"person", "motorcycle"}):  "riding",
    frozenset({"person", "umbrella"}):    "holding_umbrella",
    frozenset({"person", "dog"}):         "walking_with",
    frozenset({"person", "cat"}):         "looking_at",
    frozenset({"person", "car"}):         "near_pedestrian",
    frozenset({"person", "suitcase"}):    "carrying",
    frozenset({"person", "backpack"}):    "carrying",
    frozenset({"person", "sports ball"}): "playing_with",
    frozenset({"car", "bicycle"}):        "parked_near",
}

# Boxes whose normalised-diagonal distance is within this value are considered
# "next_to" each other (spatial proximity fallback).
_PROXIMITY_THRESHOLD = 0.35


# ---------------------------------------------------------------------------
# Public helper utilities
# ---------------------------------------------------------------------------

def clean_label(label: str) -> str:
    """
    Normalise a raw class label to a readable noun phrase.

    Replaces underscores with spaces and strips surrounding whitespace.

    Examples:
        clean_label("sports_ball") → "sports ball"
        clean_label("person")     → "person"
    """
    return label.replace("_", " ").strip()


def add_article(noun: str) -> str:
    """
    Prepend the correct indefinite article to a noun string.

    Uses the first character of the (cleaned) noun to decide between
    'a' and 'an'.  Returns the full phrase, e.g. 'a bicycle', 'an umbrella'.

    Examples:
        add_article("bicycle")  → "a bicycle"
        add_article("umbrella") → "an umbrella"
        add_article("elephant") → "an elephant"
    """
    noun = clean_label(noun)
    article = "an" if noun[0].lower() in "aeiou" else "a"
    return f"{article} {noun}"


def build_object_phrase(label: str) -> str:
    """
    Build a complete noun phrase for use as a sentence object.

    Cleans the label, then prepends 'a' / 'an'.

    Examples:
        build_object_phrase("bicycle")    → "a bicycle"
        build_object_phrase("umbrella")   → "an umbrella"
        build_object_phrase("sports_ball") → "a sports ball"
    """
    return add_article(clean_label(label))


def _confidence_qualifier(score: float) -> str:
    """
    Map a detection confidence score to a graduated causal qualifier.

    Thresholds:
        > 0.7   → "likely due to"      (high confidence)
        ≥ 0.4   → "possibly due to"    (medium confidence)
        < 0.4   → "may be related to"  (low confidence)
    """
    if score > 0.7:
        return "likely due to"
    if score >= 0.4:
        return "possibly due to"
    return "may be related to"


def has_evidence(detections: List[Detection], cues: List[str]) -> bool:
    """
    Return True if at least one detected label appears in the cues list.

    Used to gate causal hypothesis sentences on observable scene evidence,
    reducing hallucinated causes.

    Args:
        detections: List of detection dicts containing at least a "label" key.
        cues:       Label strings that constitute corroborating evidence.

    Examples:
        has_evidence([{"label": "umbrella", ...}], ["umbrella", "rain"])  → True
        has_evidence([{"label": "dog", ...}],      ["umbrella", "rain"])  → False
    """
    detected = {d["label"] for d in detections}
    return bool(detected & set(cues))


# ---------------------------------------------------------------------------
# Internal sentence builder
# ---------------------------------------------------------------------------

def _sentence(subject: str, relation: str, obj: str) -> str:
    """
    Compose one natural-language causal sentence by filling a randomly chosen
    template from TEMPLATES.

    All three templates share the same placeholders; only the framing differs
    so grammar and article handling remain uniform across variants.

    Examples (any of the three templates may be chosen):
        "A person is riding a bicycle, likely for transportation."
        "In the scene, a person is riding a bicycle, likely for transportation."
        "We observe a person riding a bicycle, likely for transportation."
    """
    import random

    subj_clean  = clean_label(subject)
    obj_phrase  = build_object_phrase(obj)
    verb        = RELATION_PHRASES.get(relation, clean_label(relation))
    explanation = CAUSAL_EXPLANATIONS.get(relation, "")

    article = "an" if subj_clean[0].lower() in "aeiou" else "a"
    cause   = f", {explanation}" if explanation else ""

    return random.choice(TEMPLATES).format(
        A    = article.capitalize(),
        a    = article,
        subj = subj_clean,
        verb = verb,
        obj  = obj_phrase,
        cause= cause,
    )


def _merge_subject_sentence(subject: str, rel_obj_pairs: List[Tuple[str, str]]) -> str:
    """
    Combine two verb-object pairs for the same subject into one sentence.

    Verb-object pairs are joined with "and"; causal phrases (when present)
    are collected and appended together after a comma.

    Example:
        _merge_subject_sentence("person", [
            ("walking_with", "dog"),
            ("near_pedestrian", "car"),
        ])
        → "A person is walking with a dog and walking near a car,
           suggesting companionship or guidance and indicating a potential interaction or risk."
    """
    subj_clean = clean_label(subject)
    article    = "An" if subj_clean[0].lower() in "aeiou" else "A"

    parts:  List[str] = []
    causes: List[str] = []
    for relation, obj in rel_obj_pairs:
        verb       = RELATION_PHRASES.get(relation, clean_label(relation))
        obj_phrase = build_object_phrase(obj)
        parts.append(f"{verb} {obj_phrase}")
        expl = CAUSAL_EXPLANATIONS.get(relation, "")
        if expl:
            causes.append(expl)

    verb_obj_clause = " and ".join(parts)
    cause_clause    = f", {' and '.join(causes)}" if causes else ""

    return f"{article} {subj_clean} is {verb_obj_clause}{cause_clause}."


def _causal_block(
    subject: str,
    relation: str,
    obj: str,
    score: float,
    detections: List[Detection],
) -> str:
    """
    Build a causal block for a single relationship with evidence-gated hypothesis.

    Components emitted:
      1. Pure observation sentence  (always present).
      2. Causal hypothesis          (gated on scene evidence + confidence):
           - Evidence present        → "This is {qualifier} {reason}."
           - No evidence, score≥0.4  → "This could be due to {reason}, but evidence is limited."
           - No evidence, score<0.4  → omitted entirely (avoids hallucination).
           - No cues defined         → confidence-only qualifier (unchanged fallback).
      3. Counterfactual statement    (always present when defined).

    Args:
        subject:    Subject label string.
        relation:   Relation key (must exist in RELATION_PHRASES).
        obj:        Object label string.
        score:      min(subject_score, object_score) from detection confidences.
        detections: Full detection list — used by has_evidence() for cue lookup.
    """
    import random

    subj_clean = clean_label(subject)
    obj_phrase = build_object_phrase(obj)
    verb       = RELATION_PHRASES.get(relation, clean_label(relation))
    article    = "an" if subj_clean[0].lower() in "aeiou" else "a"

    # Component 1 — pure observation; no causal clause.
    observation = random.choice(TEMPLATES).format(
        A=article.capitalize(), a=article,
        subj=subj_clean, verb=verb, obj=obj_phrase, cause=""
    )
    parts = [observation]

    # Component 2 — evidence-gated causal hypothesis.
    reason = CAUSAL_REASONS.get(relation, "")
    if reason:
        cue_name = RELATION_EVIDENCE.get(relation)

        if cue_name is None:
            # No evidence cues defined — fall back to pure confidence qualifier.
            qualifier = _confidence_qualifier(score)
            parts.append(f"This is {qualifier} {reason}.")
        else:
            cues     = EVIDENCE_CUES.get(cue_name, [])
            evidence = has_evidence(detections, cues)

            if evidence:
                # Corroborating objects detected — confident causal claim.
                qualifier = _confidence_qualifier(score)
                parts.append(f"This is {qualifier} {reason}.")
            elif score >= 0.4:
                # No supporting evidence but detection is reasonably confident.
                parts.append(f"This could be due to {reason}, but evidence is limited.")
            # else: low confidence + no evidence → silently omit (no hallucination).

    # Component 3 — counterfactual.
    counterfactual = COUNTERFACTUALS.get(relation, "")
    if counterfactual:
        parts.append(counterfactual)

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

Detection    = Dict   # {"label": str, "box": List[float], "score": float (optional)}
Relationship = Tuple[str, str, str]   # (subject_label, relation_verb, object_label)


def infer_relationships(
    detections: List[Detection],
    proximity_threshold: float = _PROXIMITY_THRESHOLD,
) -> List[Relationship]:
    """
    Derive pairwise relationships from detections using rule-based heuristics.

    Rules are applied in priority order:
      1. Semantic pair rules  – e.g. person + bicycle → "riding"
      2. Spatial proximity    – close box centres → "next_to" (fallback)

    Args:
        detections:          List of dicts with keys "label" and "box".
                             "box" must be [x1, y1, x2, y2] in normalised [0, 1] space.
        proximity_threshold: Normalised-diagonal-distance cutoff for the proximity fallback.

    Returns:
        Deduplicated list of (subject, relation, object) triples.
    """
    relationships: List[Relationship] = []
    seen: set = set()

    n = len(detections)
    for i in range(n):
        for j in range(i + 1, n):
            a, b         = detections[i], detections[j]
            label_a, label_b = a["label"], b["label"]

            relation = _PAIR_RULES.get(frozenset({label_a, label_b}))

            if relation is None:
                dist = _normalised_distance(a["box"], b["box"])
                if dist <= proximity_threshold:
                    relation = "next_to"

            if relation is None:
                continue

            # Canonical subject: "person" wins; otherwise alphabetical order.
            if label_a == "person" or (label_b != "person" and label_a < label_b):
                subject, obj = label_a, label_b
            else:
                subject, obj = label_b, label_a

            triple = (subject, relation, obj)
            if triple not in seen:
                seen.add(triple)
                relationships.append(triple)

    return relationships


def generate_causal_caption(
    detections: List[Detection],
    relationships: List[Relationship],
) -> str:
    """
    Convert detected objects and their relationships into a causal caption.

    Each relationship triple produces one natural sentence.  When no
    relationships exist the function falls back to a plain object description.

    Args:
        detections:    [{"label": str, "box": [x1,y1,x2,y2]}, ...]
                       The "score" key is optional and is not used here.
        relationships: [(subject, relation, object), ...]

    Returns:
        A single string — one or more sentences joined by spaces.

    Examples:
        >>> detections = [{"label": "person", "box": [.1,.1,.4,.9]},
        ...               {"label": "bicycle", "box": [.3,.3,.7,.9]}]
        >>> generate_causal_caption(detections, [("person", "riding", "bicycle")])
        'A person is riding a bicycle, likely for transportation.'

        >>> generate_causal_caption(
        ...     [{"label": "person", "box": [.1,.1,.5,.9]},
        ...      {"label": "umbrella", "box": [.4,.1,.8,.5]}],
        ...     [("person", "holding_umbrella", "umbrella")]
        ... )
        'A person is holding an umbrella, possibly because it is raining.'
    """
    if not detections:
        return "No objects detected in the scene."

    if relationships:
        # Map each label to its highest detection score for confidence lookup.
        label_to_score: Dict[str, float] = {}
        for d in detections:
            lbl   = d["label"]
            score = float(d.get("score", 0.5))
            if score > label_to_score.get(lbl, 0.0):
                label_to_score[lbl] = score

        # Group triples by subject, preserving first-appearance order.
        grouped: Dict[str, List[Tuple[str, str]]] = {}
        for subject, relation, obj in relationships:
            grouped.setdefault(subject, []).append((relation, obj))

        blocks: List[str] = []
        for subject, rel_obj_list in grouped.items():
            for i in range(0, len(rel_obj_list), 2):
                chunk = rel_obj_list[i : i + 2]

                if len(chunk) == 1:
                    # Single relation — full 3-component causal block.
                    relation, obj = chunk[0]
                    score = min(
                        label_to_score.get(subject, 0.5),
                        label_to_score.get(obj, 0.5),
                    )
                    blocks.append(_causal_block(subject, relation, obj, score, detections))

                else:
                    # Two relations for the same subject — merged observation
                    # (includes causes via CAUSAL_EXPLANATIONS) followed by one
                    # counterfactual per relation.
                    parts = [_merge_subject_sentence(subject, chunk)]
                    for relation, obj in chunk:
                        cf = COUNTERFACTUALS.get(relation, "")
                        if cf:
                            parts.append(cf)
                    blocks.append(" ".join(parts))

        return " ".join(blocks)

    # Fallback: list detected objects without any relationship context.
    unique_labels = list(dict.fromkeys(d["label"] for d in detections))

    if len(unique_labels) == 1:
        return f"{build_object_phrase(unique_labels[0]).capitalize()} is present in the scene."

    phrases    = [build_object_phrase(lbl) for lbl in unique_labels]
    enumerated = ", ".join(phrases[:-1]) + f", and {phrases[-1]}"
    return f"The scene contains {enumerated}."


def extract_top_detections(
    pred_logits,           # torch.Tensor (Q, C+1)
    pred_boxes,            # torch.Tensor (Q, 4)  cxcywh normalised
    class_names: List[str],
    threshold: float = 0.5,
    top_k: int = 10,
) -> List[Detection]:
    """
    Convert raw model outputs into detection dicts for the captioning pipeline.

    Applies softmax, filters by confidence, converts boxes from (cx, cy, w, h)
    to (x1, y1, x2, y2) in normalised [0, 1] space, and returns at most
    `top_k` results sorted by score descending.

    Args:
        pred_logits:  Shape (Q, C+1) — raw class logits; column 0 is background.
        pred_boxes:   Shape (Q, 4)   — boxes in (cx, cy, w, h) normalised format.
        class_names:  C foreground class names, 0-indexed (background excluded).
        threshold:    Minimum foreground softmax score to keep a detection.
        top_k:        Maximum detections returned.

    Returns:
        [{"label": str, "box": [x1,y1,x2,y2], "score": float}, ...]
    """
    import torch

    probs               = torch.softmax(pred_logits.float(), dim=-1)   # (Q, C+1)
    fg_probs            = probs[:, 1:]                                  # (Q, C)
    scores, class_ids   = fg_probs.max(dim=-1)                          # (Q,), (Q,)

    keep      = scores > threshold
    scores    = scores[keep]
    class_ids = class_ids[keep]
    boxes     = pred_boxes[keep]

    order     = scores.argsort(descending=True)[:top_k]
    scores    = scores[order]
    class_ids = class_ids[order]
    boxes     = boxes[order]

    cx, cy, w, h = boxes.unbind(dim=-1)
    x1 = (cx - 0.5 * w).clamp(0.0, 1.0)
    y1 = (cy - 0.5 * h).clamp(0.0, 1.0)
    x2 = (cx + 0.5 * w).clamp(0.0, 1.0)
    y2 = (cy + 0.5 * h).clamp(0.0, 1.0)

    detections: List[Detection] = []
    for idx in range(len(scores)):
        cidx = int(class_ids[idx].item())
        if cidx >= len(class_names):
            continue
        detections.append({
            "label": class_names[cidx],
            "box":   [x1[idx].item(), y1[idx].item(), x2[idx].item(), y2[idx].item()],
            "score": float(scores[idx].item()),
        })

    return detections


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------

def _box_centre(box: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _normalised_distance(box_a: List[float], box_b: List[float]) -> float:
    """Euclidean centre distance normalised by the unit diagonal (√2)."""
    cx_a, cy_a = _box_centre(box_a)
    cx_b, cy_b = _box_centre(box_b)
    dist = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5
    return dist / (2 ** 0.5)
