"""
Relation-grounded caption correction.

Extracts action/interaction phrases from BLIP captions and repairs
unsupported actions using grounded relations from the MLP predictor.

Integration point: called after BLIP generation, before evidence gating.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Verb normalisation: BLIP surface form -> canonical predicate
# ---------------------------------------------------------------------------

_VERB_NORM: Dict[str, str] = {
    "ride": "riding", "rides": "riding", "riding": "riding",
    "hold": "holding", "holds": "holding", "holding": "holding",
    "carry": "carrying", "carries": "carrying", "carrying": "carrying",
    "wear": "wearing", "wears": "wearing", "wearing": "wearing",
    "look at": "looking at", "looks at": "looking at", "looking at": "looking at",
    "sit on": "sitting on", "sits on": "sitting on", "sitting on": "sitting on",
    "stand on": "standing on", "stands on": "standing on", "standing on": "standing on",
    "use": "using", "uses": "using", "using": "using",
    "play": "playing", "plays": "playing", "playing": "playing",
    "walk": "walking", "walks": "walking", "walking": "walking",
    "run": "running", "runs": "running", "running": "running",
    "chase": "chasing", "chases": "chasing", "chasing": "chasing",
    "sit": "sitting on", "sits": "sitting on",
    "stand": "standing on", "stands": "standing on",
    "look": "looking at", "looks": "looking at",
}

# Semantic predicates (high-priority grounded interactions)
_SEMANTIC_ACTIONS: frozenset = frozenset({
    "riding", "holding", "carrying", "wearing", "looking at",
    "sitting on", "standing on",
})

# All detectable action verb forms in BLIP captions.
# Only includes verbs that describe actual interactions/actions,
# NOT generic prepositions (on, in, under) which cause false matches.
_ACTION_VERBS: frozenset = frozenset({
    "riding", "rides", "ride",
    "holding", "holds", "hold",
    "carrying", "carries", "carry",
    "wearing", "wears", "wear",
    "looking at", "looks at", "look at",
    "sitting on", "sits on", "sit on",
    "standing on", "stands on", "stand on",
    "walking", "walks", "walk",
    "running", "runs", "run",
    "standing", "stands", "stand",
    "sitting", "sits", "sit",
    "using", "uses", "use",
    "playing", "plays", "play",
    "chasing", "chases", "chase",
    "near", "next to", "beside", "behind", "in front of",
})

_MULTI_WORD_VERBS: List[str] = sorted(
    [v for v in _ACTION_VERBS if " " in v], key=len, reverse=True
)

# Semantic predicate priority order (higher = preferred for fallback)
_SEMANTIC_PRIORITY: List[str] = [
    "sitting on", "standing on", "riding", "holding",
    "carrying", "wearing", "looking at",
]

# Safe fallback spatial predicates (in priority order)
_FALLBACK_SPATIAL: List[str] = ["standing near", "next to", "beside", "near"]

# Object synonym mapping
_SYNONYMS: Dict[str, str] = {
    "woman": "person", "man": "person", "people": "person",
    "child": "person", "children": "person", "girl": "person",
    "boy": "person", "adult": "person", "kid": "person",
    "lady": "person", "guy": "person", "individual": "person",
    "someone": "person", "somebody": "person",
    "bike": "bicycle",
    "motorcycle": "motorbike",
    "cell": "cell phone", "phone": "cell phone",
    "mobile": "cell phone", "smartphone": "cell phone",
    "puppy": "dog", "canine": "dog", "pooch": "dog",
    "kitten": "cat", "kitty": "cat",
    "sofa": "couch", "settee": "couch",
    "truck": "car", "automobile": "car",
    "bottle": "bottle", "drink": "bottle",
    "sunglasses": "glasses", "eyeglasses": "glasses",
    "television": "tv", "telly": "tv", "screen": "tv",
    "luggage": "suitcase", "bag": "handbag",
    "purse": "handbag", "backpack": "backpack",
    "football": "sports ball", "soccer": "sports ball",
    "baseball": "sports ball", "basketball": "sports ball",
    "ball": "sports ball",
}

_INVERTED_SYNONYMS: Dict[str, set] = {}
for syn, canon in _SYNONYMS.items():
    if canon not in _INVERTED_SYNONYMS:
        _INVERTED_SYNONYMS[canon] = set()
    _INVERTED_SYNONYMS[canon].add(syn)
    for part in syn.split():
        _INVERTED_SYNONYMS[canon].add(part)


def _normalize_verb(verb: str) -> str:
    """Normalize a verb to its canonical predicate form."""
    v = verb.lower().strip()
    return _VERB_NORM.get(v, v)


def _resolve_label(label: str) -> str:
    """Resolve a label through synonym mapping."""
    l = label.lower().strip()
    return _SYNONYMS.get(l, l)


def _build_label_set(detections: List[Dict]) -> frozenset:
    """Build set of all detection labels + synonyms for matching."""
    words: set = set()
    for d in detections:
        label = d["label"].lower().replace("_", " ")
        words.add(label)
        words.update(label.split())
    for w in list(words):
        if w in _INVERTED_SYNONYMS:
            words.update(_INVERTED_SYNONYMS[w])
    return frozenset(words)


def _indefinite_article(word: str) -> str:
    w = word.lower().strip()
    return "an" if w and w[0] in "aeiou" else "a"


# ---------------------------------------------------------------------------
# Step 1 — Extract action/interaction phrases from BLIP caption
# ---------------------------------------------------------------------------

def _extract_action_phrases(
    caption: str,
    detections: List[Dict],
) -> List[Dict]:
    """
    Extract (subject, verb, object) action phrases from a BLIP caption.

    Returns list of dicts with:
        verb:       the action verb as found in caption
        subject:    subject label (matched to detection)
        object:     object label (matched to detection)
        subj_word:  actual word used in caption for subject
        obj_word:   actual word used in caption for object
        verb_start: start position of verb in caption
        verb_end:   end position of verb in caption
        phrase:     full matched text span
    """
    label_set = _build_label_set(detections)
    caption_lower = caption.lower()
    results: List[Dict] = []
    covered_intervals: List[Tuple[int, int]] = []

    def _acceptable(s_label, verb_start, o_label, s_word) -> bool:
        """Check if extracted subject-verb-object pair is valid.

        Rejects:
            - Missing subject or object.
            - Subject and object are the same label.
            - Subject crosses a sentence boundary (different sentence
              than the verb), which causes false cross-sentence matches.
        """
        if s_label is None or o_label is None or s_label == o_label:
            return False
        if s_word is not None:
            subj_pos = caption_lower.rfind(s_word, 0, verb_start)
            if subj_pos >= 0:
                if _has_sentence_boundary(caption_lower, subj_pos, verb_start):
                    return False
        return True

    # --- Multi-word verbs first ---
    for verb in _MULTI_WORD_VERBS:
        for m in re.finditer(re.escape(verb), caption_lower):
            vs, ve = m.start(), m.end()
            if any(s <= vs < e or s < ve <= e for s, e in covered_intervals):
                continue
            subj_word, subj_label = _find_subject_left(caption_lower, vs, label_set, detections)
            obj_word, obj_label = _find_object_right(caption_lower, ve, label_set, detections)
            if _acceptable(subj_label, vs, obj_label, subj_word):
                covered_intervals.append((vs, ve))
                results.append({
                    "verb": verb,
                    "subject": subj_label,
                    "object": obj_label,
                    "subj_word": subj_word,
                    "obj_word": obj_word,
                    "verb_start": vs,
                    "verb_end": ve,
                })

    # --- Single-word verbs ---
    single_words = sorted(
        [v for v in _ACTION_VERBS if " " not in v],
        key=len, reverse=True,
    )
    for verb in single_words:
        for m in re.finditer(r'\b' + re.escape(verb) + r'\b', caption_lower):
            vs, ve = m.start(), m.end()
            if any(s <= vs < e or s < ve <= e for s, e in covered_intervals):
                continue
            subj_word, subj_label = _find_subject_left(caption_lower, vs, label_set, detections)
            obj_word, obj_label = _find_object_right(caption_lower, ve, label_set, detections)
            if _acceptable(subj_label, vs, obj_label, subj_word):
                covered_intervals.append((vs, ve))
                results.append({
                    "verb": verb,
                    "subject": subj_label,
                    "object": obj_label,
                    "subj_word": subj_word,
                    "obj_word": obj_word,
                    "verb_start": vs,
                    "verb_end": ve,
                })

    return results


_MAX_PROXIMITY_WORDS: int = 8


def _tokens_between(text: str, start: int, end: int) -> List[str]:
    """Get word tokens between two character positions (exclusive of boundaries)."""
    segment = text[start:end]
    return re.findall(r'\b(\w+)\b', segment)


def _has_sentence_boundary(text: str, start: int, end: int) -> bool:
    """Check if there is a sentence boundary between two character positions."""
    segment = text[start:end]
    return bool(re.search(r'[.!?]', segment))


def _find_subject_left(
    text: str,
    verb_start: int,
    label_set: frozenset,
    detections: List[Dict],
) -> Tuple[Optional[str], Optional[str]]:
    """Find the nearest detected object label to the left of verb_start.

    Constraints:
        - Subject must be within _MAX_PROXIMITY_WORDS of the verb.
        - No sentence boundary between subject and verb.
        - For semantic predicates, prefers animate subjects.
    Returns (actual_caption_word, resolved_label) or (None, None).
    """
    left_text = text[:verb_start]
    tokens = re.findall(r'\b(\w+)\b', left_text)

    window_tokens = tokens[-_MAX_PROXIMITY_WORDS:] if len(tokens) > _MAX_PROXIMITY_WORDS else tokens

    # Reconstruct the window text for multi-word label matching
    window_approx = left_text
    if len(tokens) > _MAX_PROXIMITY_WORDS:
        # Find approximate char position for window start
        token_words = sum(len(t) for t in window_tokens) + len(window_tokens) - 1
        window_approx = left_text[-token_words - 20:] if token_words > 0 else left_text

    # Multi-word labels: check in window
    det_labels = sorted(
        [d["label"].lower().replace("_", " ") for d in detections],
        key=len, reverse=True,
    )
    for label in det_labels:
        if " " in label:
            idx = window_approx.rfind(label)
            if idx >= 0:
                remaining = window_approx[idx + len(label):]
                words_after = len(re.findall(r'\b(\w+)\b', remaining))
                if words_after < _MAX_PROXIMITY_WORDS and not _has_sentence_boundary(window_approx, idx + len(label), verb_start):
                    resolved = _resolve_label(label)
                    return label, resolved

    # Single-word: scan from nearest (rightmost) to farthest
    for token in reversed(window_tokens):
        if token in label_set:
            resolved = _resolve_label(token)
            return token, resolved

    return None, None


def _find_object_right(
    text: str,
    verb_end: int,
    label_set: frozenset,
    detections: List[Dict],
) -> Tuple[Optional[str], Optional[str]]:
    """Find the nearest detected object label to the right of verb_end.

    Uses proximity constraint: object must be within _MAX_PROXIMITY_WORDS
    words of the verb. Returns the first (nearest) match from left to right.
    Returns (actual_caption_word, resolved_label).
    """
    right_text = text[verb_end:]
    tokens = re.findall(r'\b(\w+)\b', right_text)

    # Only check within proximity window
    window_tokens = tokens[:_MAX_PROXIMITY_WORDS] if len(tokens) > _MAX_PROXIMITY_WORDS else tokens

    # Build window text for multi-word matching
    window_len = sum(len(t) + 1 for t in window_tokens)
    window_text = right_text[:max(10, window_len)]

    det_labels = sorted(
        [d["label"].lower().replace("_", " ") for d in detections],
        key=len, reverse=True,
    )

    # Multi-word: check the window
    for label in det_labels:
        idx = window_text.find(label)
        if idx >= 0:
            # Verify proximity
            leading_words = len(re.findall(r'\b(\w+)\b', window_text[:idx]))
            if leading_words < _MAX_PROXIMITY_WORDS:
                resolved = _resolve_label(label)
                return label, resolved

    # Single-word: scan from nearest (leftmost) to farthest
    for token in window_tokens:
        if token in label_set:
            resolved = _resolve_label(token)
            return token, resolved

    return None, None


# ---------------------------------------------------------------------------
# Step 2 — Align caption relations with grounded relations
# ---------------------------------------------------------------------------

def _is_supported(
    subject: str,
    verb: str,
    obj: str,
    relations: List[Dict],
) -> Tuple[bool, Optional[Dict]]:
    """
    Check if a (subject, verb, object) triple is supported by grounded relations.

    Returns:
        (supported, matching_relation_dict)
    """
    subj_canon = _resolve_label(subject)
    obj_canon = _resolve_label(obj)
    pred_canon = _normalize_verb(verb)

    for r in relations:
        r_subj = _resolve_label(r.get("subject", ""))
        r_pred = r.get("predicate", "").lower()
        r_obj = _resolve_label(r.get("object", ""))

        if r_subj == subj_canon and r_pred == pred_canon and r_obj == obj_canon:
            return True, r

    return False, None


# ---------------------------------------------------------------------------
# Step 4-5 — Find best grounded relation for fallback
# ---------------------------------------------------------------------------

def _find_best_rel(
    subject: str,
    obj: str,
    relations: List[Dict],
) -> Optional[Dict]:
    """
    Find the best grounded relation that can replace an unsupported action.

    Rules (Step 5 — Relation Priority):
        - Full match (both subj+obj): always allowed, any predicate type.
        - Semantic predicates: ONLY allowed if BOTH subject AND object match.
          Using a semantic predicate on mismatched objects invents new actions.
        - Spatial predicates: allowed on partial matches (subject-only or
          object-only), since spatial relations are generic enough to apply.
        - Direction swap (subj↔obj): spatial predicates only.

    Returns the highest-scored compatible relation, or None.
    """
    subj_canon = _resolve_label(subject)
    obj_canon = _resolve_label(obj)

    candidates: List[Tuple[float, Dict]] = []

    for r in relations:
        r_subj = _resolve_label(r.get("subject", ""))
        r_pred = r.get("predicate", "").lower()
        r_obj = _resolve_label(r.get("object", ""))
        conf = r.get("confidence", 0.0)
        is_semantic = r_pred in _SEMANTIC_ACTIONS

        score = 0.0
        full_match = False
        swapped = False

        # Determine match quality
        if r_subj == subj_canon and r_obj == obj_canon:
            full_match = True
            score += 3.0
        elif r_subj == obj_canon and r_obj == subj_canon:
            swapped = True
            score += 2.0
        elif r_subj == subj_canon:
            score += 2.0
        elif r_obj == obj_canon:
            score += 1.0

        if score == 0.0:
            continue

        # ── Safety constraint: semantic predicates need full match ──
        if is_semantic and not full_match:
            # Semantic predicates on mismatched objects = invented actions
            continue

        # Swapped direction: only spatial predicates
        if swapped and is_semantic:
            continue

        # Predicate quality bonus
        if is_semantic:
            score += 2.0
        elif r_pred in {"near", "next to", "beside"}:
            score += 0.5

        # Confidence contribution
        score += conf * 0.5

        candidates.append((score, r))

    if not candidates:
        return None

    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Step 6 — Safe fallback replacement builder
# ---------------------------------------------------------------------------

def _is_animate(label: str) -> bool:
    """Check if a label refers to an animate entity."""
    animate_set = {
        "person", "woman", "man", "people", "child", "girl", "boy",
        "dog", "cat", "horse", "bird", "cow", "sheep",
        "elephant", "bear", "zebra", "giraffe",
    }
    return label.lower() in animate_set


def _build_replacement_phrase(
    subj_word: str,
    predicate: str,
    obj_word: str,
    old_verb: str,
) -> str:
    """
    Build a replacement verb phrase for the caption.

    Uses the caption's actual words (subj_word, obj_word) to maintain fluency.
    Weak spatial predicates are upgraded to natural-sounding phrases
    (e.g., "near" -> "standing near" for animate subjects).
    """
    obj_article = _indefinite_article(obj_word)

    if predicate in _SEMANTIC_ACTIONS:
        return f"{predicate} {obj_article} {obj_word}"

    # Enhance weak spatial predicates for fluency
    if predicate == "near":
        if _is_animate(subj_word):
            return f"standing near {obj_article} {obj_word}"
        return f"near {obj_article} {obj_word}"
    elif predicate == "next to":
        if _is_animate(subj_word):
            return f"standing next to {obj_article} {obj_word}"
        return f"next to {obj_article} {obj_word}"
    elif predicate == "beside":
        if _is_animate(subj_word):
            return f"standing beside {obj_article} {obj_word}"
        return f"beside {obj_article} {obj_word}"
    elif predicate == "standing near":
        return f"standing near {obj_article} {obj_word}"
    elif predicate in {"on", "in"}:
        return f"{predicate} {obj_article} {obj_word}"
    else:
        return f"{predicate} {obj_article} {obj_word}"


def _build_safe_fallback(subj_word: str, obj_word: str) -> str:
    """Build safest fallback when no grounded relation exists."""
    obj_article = _indefinite_article(obj_word)
    if _is_animate(subj_word):
        return f"standing near {obj_article} {obj_word}"
    return f"near {obj_article} {obj_word}"


# ---------------------------------------------------------------------------
# Step 4 — Replace unsupported action in caption text
# ---------------------------------------------------------------------------

def _replace_action_in_caption(
    caption: str,
    subject: str,
    old_verb: str,
    obj: str,
    new_predicate: str,
) -> Tuple[str, bool]:
    """
    Replace an action verb in the caption with a new predicate.

    Uses regex to find the verb and surrounding article context.
    Preserves subject and object nouns exactly as they appear in caption.

    Returns:
        (modified_caption, was_modified)
    """
    caption_lower = caption.lower()

    # Build regex pattern that handles:
    # Case 1: "subject verb article [adj] object"
    # Case 2: "subject is/was verb article [adj] object"
    subj_esc = re.escape(subject.lower())
    verb_esc = re.escape(old_verb.lower())
    obj_esc = re.escape(obj.lower())

    new_phrase = _build_replacement_phrase(subject, new_predicate, obj, old_verb)

    # Pattern: flexible subject-verb-object matching
    # Allow optional preposition between verb and article (e.g., "sitting in a chair")
    opt_prep = r'(?:\s+\w+)?'
    patterns = [
        # "subject ... verb [prep] article [adj] object"
        rf'(\b{subj_esc}\s+(?:\w+\s+)*?){verb_esc}{opt_prep}(\s+(?:a|an|the)\s+(?:\w+\s+)?{obj_esc}\b)',
        # "subject ... verb ... object" (no article, more flexible)
        rf'(\b{subj_esc}\s+(?:\w+\s+)*?){verb_esc}(\s+(?:\w+\s+)?{obj_esc}\b)',
    ]

    for pat in patterns:
        match = re.search(pat, caption_lower, re.IGNORECASE)
        if match:
            before_subj = caption[:match.start(1)]
            prefix = caption[match.start(1):match.end(1)]
            rest = caption[match.end():]
            result = before_subj + prefix + new_phrase + rest
            return result, True

    return caption, False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def correct_caption_relations(
    caption: str,
    detections: List[Dict],
    relations: List[Dict],
    debug: bool = True,
) -> Tuple[str, Dict]:
    """
    Correct unsupported action verbs in a BLIP caption using grounded relations.

    Steps:
        1. Extract action/interaction phrases from caption
        2. Align with grounded relations
        3. Preserve supported interactions
        4. Repair unsupported actions using grounded fallbacks
        5. Use relation priority (semantic > spatial)
        6. Safe fallback phrases when no grounded interaction exists
        7. Maintain fluency

    Args:
        caption:    Raw BLIP caption.
        detections: Detection list.
        relations:  Grounded relation list from MLP.
        debug:      Print debug output.

    Returns:
        (corrected_caption, correction_log)
    """
    log: Dict = {
        "input_caption": caption,
        "extracted_actions": [],
        "preserved": [],
        "repaired": [],
        "removed": [],
        "fallback_used": [],
        "output_caption": caption,
    }

    if not relations:
        if debug:
            print(f"\n[relation correction] no grounded relations — skipping")
        return caption, log

    # Step 1 — Extract action phrases
    actions = _extract_action_phrases(caption, detections)
    log["extracted_actions"] = [
        {"verb": a["verb"], "subject": a["subject"], "object": a["object"]}
        for a in actions
    ]

    if not actions:
        if debug:
            print(f"\n[relation correction] no action phrases detected — caption unchanged")
        return caption, log

    corrected = caption

    if debug:
        print(f"\n[relation correction]")
        print(f"  BLIP: {caption}")
        print(f"  Grounded relations:")
        for r in relations:
            print(f"    {r['subject']} {r['predicate']} {r['object']} (conf={r.get('confidence',0):.3f})")

    # Step 2-4 — Check and repair each action
    for action in actions:
        verb = action["verb"]
        subject = action["subject"]
        obj = action["object"]

        normalized = _normalize_verb(verb)
        supported, matching_rel = _is_supported(subject, verb, obj, relations)

        if supported:
            log["preserved"].append({
                "action": normalized,
                "subject": subject,
                "object": obj,
                "grounded_with": {
                    "predicate": matching_rel["predicate"],
                    "subject": matching_rel["subject"],
                    "object": matching_rel["object"],
                },
            })
            if debug:
                print(f"  [+] preserved: {subject} {normalized} {obj} (grounded: {matching_rel['subject']} {matching_rel['predicate']} {matching_rel['object']})")
        else:
            # Step 4 — Repair: find best grounded relation
            best_rel = _find_best_rel(subject, obj, relations)

            if best_rel is not None:
                new_predicate = best_rel["predicate"]
                corrected, was_modified = _replace_action_in_caption(
                    corrected, action.get("subj_word", subject),
                    verb, action.get("obj_word", obj),
                    new_predicate,
                )
                if was_modified:
                    log["repaired"].append({
                        "action": normalized,
                        "subject": subject,
                        "object": obj,
                        "replaced_with": new_predicate,
                        "grounded_relation": {
                            "predicate": best_rel["predicate"],
                            "subject": best_rel["subject"],
                            "object": best_rel["object"],
                        },
                    })
                    if debug:
                        print(f"  [-] replaced unsupported action: {normalized}")
                        print(f"    subject: {subject}, object: {obj}")
                        print(f"    -> using grounded: {best_rel['subject']} {best_rel['predicate']} {best_rel['object']}")
                else:
                    log["repaired"].append({
                        "action": normalized,
                        "status": "replacement_failed",
                    })
                    if debug:
                        print(f"  [!] failed to replace: {normalized} in caption")
            else:
                # Step 6 — No grounded relation exists: use safe fallback
                fallback_pred = _FALLBACK_SPATIAL[0]
                corrected, was_modified = _replace_action_in_caption(
                    corrected, action.get("subj_word", subject),
                    verb, action.get("obj_word", obj),
                    fallback_pred,
                )
                if was_modified:
                    log["fallback_used"].append({
                        "action": normalized,
                        "fallback": fallback_pred,
                    })
                    if debug:
                        print(f"  [-] no grounded relation -- using fallback: {fallback_pred}")
                else:
                    if debug:
                        print(f"  [!] no grounded relation and could not replace: {normalized}")

    log["output_caption"] = corrected

    if debug and log["preserved"]:
        for p in log["preserved"]:
            print(f"  [+] preserved grounded interaction: {p['action']}")

    if debug and log["repaired"]:
        for r in log["repaired"]:
            if "replaced_with" in r:
                print(f"  -> replaced unsupported action \"{r['action']}\" with \"{r['replaced_with']}\"")
        print(f"  final: {corrected}")

    return corrected, log
