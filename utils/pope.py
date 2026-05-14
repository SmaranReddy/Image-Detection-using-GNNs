"""
POPE: Polling-based Object Probing Evaluation

Evaluates object hallucination by constructing probing questions about
whether objects mentioned in a caption are actually present in the image.

POPE differs from CHAIR conceptually:
  CHAIR measures hallucination rate among mentioned objects (chair_i, chair_s)
  POPE measures precision, recall, F1, and accuracy of object assertions

POPE complements SPICE + CHAIR:
  SPICE -> relation grounding quality
  CHAIR -> hallucinated caption objects
  POPE  -> object hallucination probing behavior

Probing paradigm:
  For each candidate object from COCO-80:
    - "Is there a [object] in the image based on the caption?"
    - Positive probe: object is MENTIONED in the caption
    - Negative probe: object is NOT mentioned (sampled, balanced)
    - Answer is correct if it matches ground-truth presence

Key distinction from CHAIR:
  CHAIR only evaluates caption-output overlap with GT objects.
  POPE additionally evaluates recall (missed objects) and overall
  accuracy including true negatives via balanced probing.
"""

from __future__ import annotations

import random
from typing import Dict, Set

from utils.metrics import COCO_80, detect_coco_objects


def compute_pope(
    candidate: str,
    gt_objects: Set[str],
    seed: int = 42,
) -> Dict:
    """Compute POPE metrics for a single caption.

    Constructs probing questions by:
      1. Extracting mentioned COCO-80 objects from the caption (positive probes)
      2. Sampling non-mentioned, non-GT COCO-80 objects (negative probes)
      3. Computing the confusion matrix against GT object annotations

    All three systems are evaluated with IDENTICAL probing logic and
    object vocabulary, ensuring fair comparison.

    Args:
        candidate:  Generated caption string.
        gt_objects: Set of ground-truth COCO-80 object labels in the image.
        seed:       Random seed for deterministic negative probe sampling.

    Returns:
        Dict with POPE metrics:
            pope_precision, pope_recall, pope_f1, pope_accuracy,
            pope_tp, pope_fp, pope_fn, pope_tn,
            pope_num_positive_probes, pope_num_negative_probes,
            pope_hallucinated_objects, pope_missed_objects
    """
    mentioned = detect_coco_objects(candidate)

    # Confusion matrix building blocks.
    tp_set = mentioned & gt_objects
    fp_set = mentioned - gt_objects
    fn_set = gt_objects - mentioned

    tp = len(tp_set)
    fp = len(fp_set)
    fn = len(fn_set)

    # -- Negative probes --------------------------------------------------
    # Balanced sampling: ~equal to positive probes, minimum 10.
    # Pool = objects neither mentioned by caption NOR present in GT.
    negative_pool = sorted(COCO_80 - mentioned - gt_objects)

    if negative_pool:
        num_positives = len(mentioned)
        target_negatives = max(10, num_positives)
        num_negatives = min(len(negative_pool), target_negatives)
        num_negatives = max(1, num_negatives)

        rng = random.Random(seed)
        rng.shuffle(negative_pool)
        negative_probes = set(negative_pool[:num_negatives])
    else:
        negative_probes = set()
        num_negatives = 0

    # All negative probes are TN by construction
    # (verified they are not in gt_objects).
    tn = len(negative_probes)

    # -- Metrics ----------------------------------------------------------
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (
        (tp + tn) / (tp + fp + tn + fn)
        if (tp + fp + tn + fn) > 0
        else 0.0
    )

    return {
        "pope_precision": round(precision, 4),
        "pope_recall": round(recall, 4),
        "pope_f1": round(f1, 4),
        "pope_accuracy": round(accuracy, 4),
        "pope_tp": tp,
        "pope_fp": fp,
        "pope_fn": fn,
        "pope_tn": tn,
        "pope_num_positive_probes": num_positives,
        "pope_num_negative_probes": num_negatives,
        "pope_hallucinated_objects": sorted(fp_set),
        "pope_missed_objects": sorted(fn_set),
    }
