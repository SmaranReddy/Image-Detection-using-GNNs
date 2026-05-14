"""
Comprehensive metric computation for grounded caption evaluation.

Supports:
    BLEU        — n-gram precision (1-4)
    METEOR      — alignment-based with synonym matching via WordNet
    BERTScore   — BERT-based semantic similarity
    SPICE       — scene-graph semantic grounding (WordNet-enhanced)
    CHAIR       — hallucination detection and grounding reliability
    POPE        — probing-based object hallucination evaluation

Usage:
    from utils.metrics import evaluate_caption, evaluate_all

    # Single caption
    result = evaluate_caption("a person riding a bike", refs, objects)

    # Batch evaluation
    results = evaluate_all(candidates, references_list, objects_list)
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Lazy NLTK / WordNet imports — avoid slow startup when metrics are unused.
# ---------------------------------------------------------------------------

_wnl = None  # wordnet lemmatizer
_wn  = None  # wordnet synset lookup


def _ensure_nltk():
    global _wnl, _wn
    if _wnl is None:
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import wordnet as wn
        _wnl = WordNetLemmatizer()
        _wn = wn


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

# MS-COCO 80 class names — used by CHAIR and SPICE for object filtering.
COCO_80: frozenset = frozenset({
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
})

# Multi-word COCO labels (must be checked before single-word tokens).
COCO_MULTI_WORD: List[str] = sorted(
    [l for l in COCO_80 if " " in l],
    key=lambda x: -len(x),
)

# Common English stop words for caption parsing.
_STOPS: frozenset = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "to", "of", "in",
    "on", "at", "by", "for", "with", "from", "into", "and", "or", "but",
    "not", "no", "it", "its", "this", "that", "there", "they", "them",
    "what", "which", "who", "when", "where", "why", "how", "all", "both",
    "each", "some", "any", "such", "than", "too", "very", "just", "also",
    "if", "then", "else", "so", "about", "up", "out", "off", "over",
})


def tokenize(text: str) -> List[str]:
    """Standard tokenisation: lower-case, split on non-alpha chars."""
    return re.findall(r"[a-z]+", text.lower())


def sent_tokenize(text: str) -> List[str]:
    """Split text into sentences on sentence-ending punctuation."""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sents if s]


# Synonym mapping for CHAIR — mirrors vg_dataset.py SYNONYM_MAP.
CHAIR_SYNONYM_MAP: Dict[str, str] = {
    "man": "person", "men": "person", "woman": "person", "women": "person",
    "boy": "person", "girl": "person", "people": "person", "child": "person",
    "children": "person", "guy": "person", "lady": "person",
    "bike": "bicycle", "cycle": "bicycle",
    "vehicle": "car", "automobile": "car",
    "sofa": "couch",
    "television": "tv", "tv monitor": "tv", "monitor": "tv",
    "cellphone": "cell phone", "mobile": "cell phone", "phone": "cell phone",
    "motorbike": "motorcycle",
    "aeroplane": "airplane", "aero plane": "airplane",
    "plant": "potted plant",
    "motor": "motorcycle",
    "back pack": "backpack",
}


def _resolve_synonym(word: str) -> str:
    """Map a word to canonical COCO-80 label using synonym map and WordNet."""
    word_lower = word.lower().strip()
    if word_lower in CHAIR_SYNONYM_MAP:
        return CHAIR_SYNONYM_MAP[word_lower]
    if word_lower in COCO_80:
        return word_lower
    _ensure_nltk()
    for syn in _wn.synsets(word_lower, pos=_wn.NOUN):
        for hyper in syn.hypernyms():
            hyper_name = hyper.name().split(".")[0]
            if hyper_name in COCO_80:
                return hyper_name
        for hyper in syn.hypernyms():
            for hyper2 in hyper.hypernyms():
                hyper_name = hyper2.name().split(".")[0]
                if hyper_name in COCO_80:
                    return hyper_name
    return word_lower


def detect_coco_objects(caption: str) -> Set[str]:
    """Detect which COCO-80 objects are mentioned in a caption, with synonym resolution."""
    caption_lower = caption.lower()
    found: Set[str] = set()

    for mw in COCO_MULTI_WORD:
        if mw in caption_lower:
            resolved = _resolve_synonym(mw)
            if resolved in COCO_80:
                found.add(resolved)

    words = set(tokenize(caption))
    for w in words:
        resolved = _resolve_synonym(w)
        if resolved in COCO_80:
            found.add(resolved)

    return found


# ---------------------------------------------------------------------------
# 1. BLEU
# ---------------------------------------------------------------------------

def _n_grams(tokens: List[str], n: int) -> Counter:
    return Counter(zip(*[tokens[i:] for i in range(n)]))


def _bleu_compute(
    candidate: List[str],
    references: List[List[str]],
    max_n: int = 4,
) -> Dict[str, float]:
    """Standard BLEU with corpus-level brevity penalty."""
    c_len = len(candidate)
    r_len = min(
        (abs(len(r) - c_len), len(r))
        for r in references
    )[1]

    precisions: Dict[int, float] = {}
    for n in range(1, max_n + 1):
        c_ngrams = _n_grams(candidate, n)
        if not c_ngrams:
            precisions[n] = 0.0
            continue

        ref_counts: Counter = Counter()
        for ref in references:
            ref_ngrams = _n_grams(ref, n)
            for ng, cnt in ref_ngrams.items():
                ref_counts[ng] = max(ref_counts[ng], cnt)

        matches = sum(
            min(cnt, ref_counts.get(ng, 0))
            for ng, cnt in c_ngrams.items()
        )
        total = sum(c_ngrams.values())
        precisions[n] = matches / total if total > 0 else 0.0

    prod = 1.0
    for n in range(1, max_n + 1):
        prod *= precisions[n]
    geo_mean = prod ** (1.0 / max_n)

    if c_len < r_len:
        bp = math.exp(1.0 - r_len / c_len) if c_len > 0 else 0.0
    else:
        bp = 1.0

    return {
        "bleu1": round(precisions[1], 4),
        "bleu2": round(precisions[2], 4),
        "bleu3": round(precisions[3], 4),
        "bleu4": round(precisions[4], 4),
        "bleu": round(bp * geo_mean, 4),
    }


def compute_bleu(
    candidate: str,
    references: List[str],
) -> Dict[str, float]:
    if not references:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0, "bleu": 0.0}
    cand_tokens = tokenize(candidate)
    ref_tokens = [tokenize(r) for r in references]
    return _bleu_compute(cand_tokens, ref_tokens)


# ---------------------------------------------------------------------------
# 2. METEOR
# ---------------------------------------------------------------------------

def _meteor_align(
    cand_tokens: List[str],
    ref_tokens: List[str],
) -> Tuple[List[int], List[int]]:
    """Greedy alignment with exact → stemmed matching stages."""
    _ensure_nltk()
    cand_map = [0] * len(cand_tokens)
    ref_map = [0] * len(ref_tokens)
    matches = 0

    cand_stems = [_wnl.lemmatize(t) for t in cand_tokens]
    ref_stems = [_wnl.lemmatize(t) for t in ref_tokens]

    # Stage 1: exact match
    for i, ct in enumerate(cand_tokens):
        if cand_map[i]:
            continue
        for j, rt in enumerate(ref_tokens):
            if ref_map[j]:
                continue
            if ct == rt:
                cand_map[i] = 1
                ref_map[j] = 1
                matches += 1
                break

    # Stage 2: stem match
    for i, cs in enumerate(cand_stems):
        if cand_map[i]:
            continue
        for j, rs in enumerate(ref_stems):
            if ref_map[j]:
                continue
            if cs == rs and cand_tokens[i] != ref_tokens[j]:
                cand_map[i] = 1
                ref_map[j] = 1
                matches += 1
                break

    return cand_map, ref_map


def _meteor_chunks(cand_map: List[int]) -> int:
    """Count matched chunks for fragmentation penalty."""
    chunks = 0
    in_chunk = False
    for m in cand_map:
        if m and not in_chunk:
            chunks += 1
            in_chunk = True
        elif not m:
            in_chunk = False
    return max(chunks, 1)


def compute_meteor(
    candidate: str,
    references: List[str],
) -> Dict[str, float]:
    if not references:
        return {"meteor": 0.0}
    best = 0.0
    for ref in references:
        cand_tokens = tokenize(candidate)
        ref_tokens = tokenize(ref)

        cand_map, ref_map = _meteor_align(cand_tokens, ref_tokens)
        m = sum(cand_map)
        if m == 0:
            continue

        precision = m / len(cand_tokens)
        recall = m / len(ref_tokens)
        if precision + recall == 0:
            continue

        fmean = (10 * precision * recall) / (9 * precision + recall)
        chunks = _meteor_chunks(cand_map)
        penalty = 0.5 * (chunks / m) if m > 0 else 0
        score = fmean * (1 - penalty)
        best = max(best, score)

    return {"meteor": round(best, 4)}


# ---------------------------------------------------------------------------
# 3. BERTScore
# ---------------------------------------------------------------------------

_bert_model = None
_bert_tokenizer = None
_BERT_MODEL_ID = "distilbert-base-uncased"


def _ensure_bert():
    global _bert_model, _bert_tokenizer
    if _bert_model is None:
        from transformers import AutoModel, AutoTokenizer
        _bert_tokenizer = AutoTokenizer.from_pretrained(_BERT_MODEL_ID)
        _bert_model = AutoModel.from_pretrained(_BERT_MODEL_ID)
        _bert_model.eval()
        if torch.cuda.is_available():
            _bert_model = _bert_model.cuda()


def _bert_encode(sentences: List[str]) -> torch.Tensor:
    _ensure_bert()
    device = next(_bert_model.parameters()).device
    encoded = _bert_tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt",
    )
    with torch.inference_mode():
        outputs = _bert_model(**{k: v.to(device) for k, v in encoded.items()})
    last_hidden = outputs.last_hidden_state  # (B, L, D)
    attention = encoded["attention_mask"].to(device)
    mask = attention.unsqueeze(-1).float()
    avg_emb = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
    avg_emb = avg_emb / avg_emb.norm(dim=-1, keepdim=True)
    return avg_emb.cpu()


def compute_bert_score(
    candidate: str,
    references: List[str],
) -> Dict[str, float]:
    if not references:
        return {"bertscore_f1": 0.0}
    cand_emb = _bert_encode([candidate])
    ref_embs = _bert_encode(references)
    sims = (cand_emb @ ref_embs.T).squeeze(0)
    best = sims.max().item()
    return {"bertscore_f1": round(best, 4)}


# ---------------------------------------------------------------------------
# 4. SPICE (simplified Python implementation)
# ---------------------------------------------------------------------------

def _extract_scene_graph(text: str) -> Dict[str, Set]:
    """Extract objects and relation tuples from a caption.

    Returns:
        {"objects": {obj1, obj2, ...}, "relations": {(subj, rel, obj), ...}}
    """
    sents = sent_tokenize(text)
    objects: Set[str] = set()
    relations: Set[Tuple[str, str, str]] = set()

    for sent in sents:
        tokens = tokenize(sent)
        if not tokens:
            continue

        # Detect COCO objects in the sentence.
        sent_lower = sent.lower()
        sent_objs = set()
        for mw in COCO_MULTI_WORD:
            if mw in sent_lower:
                sent_objs.add(mw)

        single_words = set(tokens)
        for obj in COCO_80:
            if " " not in obj and obj in single_words:
                sent_objs.add(obj)

        objects.update(sent_objs)

        # Extract simple relations via heuristics:
        # Look for patterns like "noun verb noun" or "noun prep noun"
        # where both nouns are COCO objects.
        obj_list = sorted(sent_objs, key=lambda x: -len(x))
        if len(obj_list) >= 2:
            for i in range(len(obj_list)):
                for j in range(len(obj_list)):
                    if i == j:
                        continue
                    # Try to find a verb between them.
                    subj = obj_list[i]
                    obj = obj_list[j]
                    # Find order in sentence.
                    si = sent_lower.find(subj)
                    oi = sent_lower.find(obj)
                    if si < 0 or oi < 0:
                        continue

                    between = sent_lower[si + len(subj):oi]
                    between_words = tokenize(between)

                    # Known relation verbs.
                    rel_verbs = {
                        "riding", "holding", "carrying", "wearing", "pulling",
                        "pushing", "leading", "following", "chasing",
                        "sitting", "standing", "lying", "walking", "running",
                        "looking", "playing", "eating", "drinking",
                    }
                    relation = None
                    for w in between_words:
                        if w in rel_verbs:
                            relation = w
                            break

                    if relation is None:
                        # Check for prepositional relations.
                        preps = {"on", "in", "under", "next", "near",
                                 "behind", "front", "above", "beside",
                                 "inside", "outside", "with", "by"}
                        for w in between_words:
                            if w in preps:
                                relation = w
                                break

                    if relation is not None:
                        relations.add((subj, relation, obj))

    return {"objects": objects, "relations": relations}


def _synset_match(word1: str, word2: str) -> bool:
    """Check if two words match via WordNet synonymy."""
    _ensure_nltk()
    if word1 == word2:
        return True
    # Check direct WordNet synset overlap.
    syns1 = _wn.synsets(word1, pos=_wn.NOUN)
    syns2 = _wn.synsets(word2, pos=_wn.NOUN)
    if not syns1 or not syns2:
        return _wnl.lemmatize(word1) == _wnl.lemmatize(word2)
    set1 = set(s.name() for s in syns1)
    set2 = set(s.name() for s in syns2)
    return bool(set1 & set2)


def compute_spice(
    candidate: str,
    references: List[str],
) -> Dict[str, float]:
    if not references:
        return {"spice": 0.0, "spice_precision": 0.0,
                "spice_recall": 0.0, "spice_f": 0.0}
    can_sg = _extract_scene_graph(candidate)
    can_objs = can_sg["objects"]
    can_rels = can_sg["relations"]

    if not can_objs:
        return {"spice": 0.0, "spice_precision": 0.0,
                "spice_recall": 0.0, "spice_f": 0.0}

    best_f = 0.0
    best_p = 0.0
    best_r = 0.0

    for ref in references:
        ref_sg = _extract_scene_graph(ref)
        ref_objs = ref_sg["objects"]
        ref_rels = ref_sg["relations"]

        if not ref_objs:
            continue

        # Object matching with synonym support.
        obj_match = 0
        for co in can_objs:
            for ro in ref_objs:
                if _synset_match(co, ro):
                    obj_match += 1
                    break

        # Relation matching.
        rel_match = 0
        for cs, cr, co in can_rels:
            for rs, rr, ro in ref_rels:
                if (_synset_match(cs, rs) and
                        _synset_match(cr, rr) and
                        _synset_match(co, ro)):
                    rel_match += 1
                    break

        total_can = len(can_objs) + len(can_rels)
        total_ref = len(ref_objs) + len(ref_rels)
        total_match = obj_match + rel_match

        precision = total_match / total_can if total_can > 0 else 0.0
        recall = total_match / total_ref if total_ref > 0 else 0.0
        f = (2 * precision * recall / (precision + recall)
             if precision + recall > 0 else 0.0)

        if f > best_f:
            best_f = f
            best_p = precision
            best_r = recall

    return {
        "spice": round(best_f, 4),
        "spice_precision": round(best_p, 4),
        "spice_recall": round(best_r, 4),
        "spice_f": round(best_f, 4),
    }


# ---------------------------------------------------------------------------
# 5. CHAIR
# ---------------------------------------------------------------------------

# COCO-80 category_id → name (used by instances JSON).

COCO_80_ID_TO_NAME: Dict[int, str] = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow",
    22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack",
    28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee",
    35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard",
    42: "surfboard", 43: "tennis racket", 44: "bottle", 46: "wine glass",
    47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl",
    52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli",
    57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake",
    62: "chair", 63: "couch", 64: "potted plant", 65: "bed", 67: "dining table",
    70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote",
    76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven",
    80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock",
    86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier",
    90: "toothbrush",
}


def compute_chair(
    candidate: str,
    gt_objects: Set[str],
) -> Dict[str, float]:
    """Compute CHAIR metrics.

    CHAIRi: fraction of hallucinated object mentions.
    CHAIRs: whether image has any hallucinated object mention.

    Args:
        candidate: Generated caption.
        gt_objects: Set of ground-truth COCO-80 object labels in the image.

    Returns:
        {"chair_i": ..., "chair_s": ...}
    """
    mentioned = detect_coco_objects(candidate)

    if not mentioned:
        return {
            "chair_i": 0.0,
            "chair_s": 0.0,
            "hallucinated_objects": [],
            "missed_objects": sorted(gt_objects) if gt_objects else [],
        }

    hallucinated = mentioned - gt_objects
    chair_i = len(hallucinated) / len(mentioned) if mentioned else 0.0
    chair_s = 1.0 if hallucinated else 0.0

    return {
        "chair_i": round(chair_i, 4),
        "chair_s": round(chair_s, 4),
        "hallucinated_objects": sorted(hallucinated),
        "missed_objects": sorted(gt_objects - mentioned),
    }


CHAIR_REQUIRED_KEYS: frozenset = frozenset({
    "chair_i", "chair_s", "hallucinated_objects", "missed_objects",
})


def validate_chair_schema(chair_result: Dict) -> None:
    """Validate a CHAIR result dict contains the full expected schema.

    Raises:
        KeyError: With a detailed message listing missing keys and available
                  keys, making schema drift immediately debuggable.
    """
    missing = CHAIR_REQUIRED_KEYS - set(chair_result.keys())
    if missing:
        raise KeyError(
            f"CHAIR result missing required key(s): {sorted(missing)}.\n"
            f"  Available keys: {sorted(chair_result.keys())}\n"
            f"  Expected keys:  {sorted(CHAIR_REQUIRED_KEYS)}\n"
            f"  This means the CHAIR return schema has diverged from the "
            f"contract expected by consumers (hallucination_eval.py, "
            f"evaluate.py, etc.). Fix compute_chair() to always return "
            f"the full schema, even in early-return paths."
        )


# ---------------------------------------------------------------------------
# Unified evaluation API
# ---------------------------------------------------------------------------

CaptionResult = Dict[str, float]


def evaluate_caption(
    candidate: str,
    references: List[str],
    gt_objects: Optional[Set[str]] = None,
) -> CaptionResult:
    """Compute ALL metrics for a single caption.

    Args:
        candidate:  Generated caption string.
        references: List of reference caption strings.
        gt_objects: Set of ground-truth COCO-80 objects (for CHAIR).

    Returns:
        Dict with all metric scores.
    """
    result: CaptionResult = {}

    # BLEU
    result.update(compute_bleu(candidate, references))

    # METEOR
    result.update(compute_meteor(candidate, references))

    # BERTScore
    result.update(compute_bert_score(candidate, references))

    # SPICE
    result.update(compute_spice(candidate, references))

    # CHAIR
    if gt_objects is not None:
        chair = compute_chair(candidate, gt_objects)
        validate_chair_schema(chair)
        result["chair_i"] = chair["chair_i"]
        result["chair_s"] = chair["chair_s"]

    # POPE (lazy import to avoid circular dependency)
    if gt_objects is not None:
        from utils.pope import compute_pope
        pope = compute_pope(candidate, gt_objects)
        result["pope_precision"] = pope["pope_precision"]
        result["pope_recall"] = pope["pope_recall"]
        result["pope_f1"] = pope["pope_f1"]
        result["pope_accuracy"] = pope["pope_accuracy"]
        result["pope_tp"] = pope["pope_tp"]
        result["pope_fp"] = pope["pope_fp"]
        result["pope_fn"] = pope["pope_fn"]
        result["pope_tn"] = pope["pope_tn"]
        result["pope_num_positive_probes"] = pope["pope_num_positive_probes"]
        result["pope_num_negative_probes"] = pope["pope_num_negative_probes"]
        result["pope_hallucinated_objects"] = pope["pope_hallucinated_objects"]
        result["pope_missed_objects"] = pope["pope_missed_objects"]

    return result


def evaluate_all(
    candidates: List[str],
    references_list: List[List[str]],
    objects_list: Optional[List[Set[str]]] = None,
    system_name: str = "system",
) -> Dict:
    """Evaluate a batch of captions and aggregate results.

    Args:
        candidates:      List of generated captions.
        references_list: List of reference caption lists (one per candidate).
        objects_list:    Optional list of GT object sets (for CHAIR).
        system_name:     Name for this system for display.

    Returns:
        {"system": ..., "per_image": [...], "aggregate": {...},
         "hallucination_summary": {...}}
    """
    results = []
    has_objects = objects_list is not None

    for i, cand in enumerate(candidates):
        refs = references_list[i]
        objs = objects_list[i] if has_objects else None
        r = evaluate_caption(cand, refs, objs)
        r["candidate"] = cand
        results.append(r)

    # Aggregate all numeric metrics.
    keys = ["bleu1", "bleu2", "bleu3", "bleu4", "bleu",
            "meteor", "bertscore_f1", "spice", "spice_f",
            "chair_i", "chair_s",
            "pope_precision", "pope_recall", "pope_f1", "pope_accuracy"]

    agg: Dict[str, float] = {}
    for k in keys:
        vals = [r[k] for r in results if k in r]
        if vals:
            agg[f"avg_{k}"] = round(float(np.mean(vals)), 4)
            agg[f"std_{k}"] = round(float(np.std(vals)), 4)

    # Hallucination summary.
    hall_summary = {}
    if has_objects:
        chair_i_vals = [r["chair_i"] for r in results if "chair_i" in r]
        chair_s_vals = [r["chair_s"] for r in results if "chair_s" in r]
        pope_prec_vals = [r["pope_precision"] for r in results if "pope_precision" in r]
        pope_rec_vals = [r["pope_recall"] for r in results if "pope_recall" in r]
        pope_f1_vals = [r["pope_f1"] for r in results if "pope_f1" in r]
        pope_acc_vals = [r["pope_accuracy"] for r in results if "pope_accuracy" in r]
        hall_summary = {
            "avg_chair_i": round(float(np.mean(chair_i_vals)), 4) if chair_i_vals else 0.0,
            "avg_chair_s": round(float(np.mean(chair_s_vals)), 4) if chair_s_vals else 0.0,
            "hallucination_rate": round(float(np.mean(chair_s_vals)), 4) if chair_s_vals else 0.0,
            "total_images_with_hallucination": int(sum(chair_s_vals)) if chair_s_vals else 0,
            "avg_pope_precision": round(float(np.mean(pope_prec_vals)), 4) if pope_prec_vals else 0.0,
            "avg_pope_recall": round(float(np.mean(pope_rec_vals)), 4) if pope_rec_vals else 0.0,
            "avg_pope_f1": round(float(np.mean(pope_f1_vals)), 4) if pope_f1_vals else 0.0,
            "avg_pope_accuracy": round(float(np.mean(pope_acc_vals)), 4) if pope_acc_vals else 0.0,
            "pope_total_hallucinated_assertions": int(sum(
                r.get("pope_fp", 0) for r in results if "pope_fp" in r
            )),
            "pope_total_missed_objects": int(sum(
                r.get("pope_fn", 0) for r in results if "pope_fn" in r
            )),
        }

    return {
        "system": system_name,
        "num_samples": len(results),
        "aggregate": agg,
        "hallucination_summary": hall_summary,
        "per_image": results,
    }
