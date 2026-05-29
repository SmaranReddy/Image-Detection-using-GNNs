"""
Centralised console output manager.

Provides a global DEBUG flag and helper functions to produce
either a concise professional summary (DEBUG=False) or the full
research-oriented debug trace (DEBUG=True).

Usage:
    from utils.logger_utils import debug_print, section, print_clean_summary, set_debug

    set_debug(True)   # enable full verbose output
    set_debug(False)  # only clean summaries (default)

    debug_print("Raw logits:", tensor)   # only printed when DEBUG=True
    section("Image: foo.jpg")            # always printed (clean header)
    print_clean_summary(result, path)    # always printed (professional summary)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Global debug flag
# ---------------------------------------------------------------------------

DEBUG: bool = False


def set_debug(enabled: bool) -> None:
    """Enable or disable full debug console output."""
    global DEBUG
    DEBUG = enabled


# ---------------------------------------------------------------------------
# Conditional print helpers
# ---------------------------------------------------------------------------

def debug_print(*args: Any, **kwargs: Any) -> None:
    """Print *args only when DEBUG is True."""
    if DEBUG:
        print(*args, **kwargs)


def section(title: str, char: str = "=", width: int = 50) -> None:
    """Print a clean section header (visible in both modes)."""
    print()
    print(f"{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def sub_section(title: str, char: str = "-", width: int = 40) -> None:
    """Print a lighter sub-section header."""
    print(f"\n  {char * width}")
    print(f"  {title}")
    print(f"  {char * width}")


# ---------------------------------------------------------------------------
# Clean per-image summary printer
# ---------------------------------------------------------------------------

def print_clean_summary(result: Dict, img_path: str) -> None:
    """Print a concise, human-readable semantic summary for one image.

    Extracts final pipeline state (verified detections, rejected,
    filtered relations, gated caption, gate status) — no intermediate noise.
    """
    image_name = Path(img_path).name

    section(f"Image: {image_name}", width=50)

    # --- Verified Objects ---
    detections = result.get("detections", [])
    verified_labels = _unique_labels(detections)
    raw_detections = result.get("raw_detections", [])
    raw_labels = _unique_labels(raw_detections)

    print("\nVerified Objects:")
    if verified_labels:
        for label in sorted(verified_labels):
            print(f"  \u2713 {label}")
    else:
        print("  (None)")

    # --- Rejected Detections ---
    rejected_labels = sorted(raw_labels - verified_labels)
    print("\nRejected Detections:")
    if rejected_labels:
        for label in rejected_labels:
            print(f"  \u2717 {label}")
    else:
        print("  (None)")

    # --- Grounded Relations (final filtered) ---
    relations = result.get("relations", [])
    print("\nGrounded Relations:")
    if relations:
        seen: set = set()
        for r in relations:
            key = (r["subject"], r["predicate"], r["object"])
            if key not in seen:
                seen.add(key)
                print(f"  \u2713 {r['subject']} {r['predicate']} {r['object']}")
    else:
        print("  (None)")

    # --- Generated Caption ---
    caption = result.get("caption", "")
    print("\nGenerated Caption:")
    print(f'  "{caption}"')

    # --- Hallucination Status ---
    raw = result.get("raw_caption", "")
    gated = result.get("caption", "")
    is_accepted = (raw == gated)
    is_repaired = (not is_accepted) and gated and "The scene contains" not in gated
    is_fallback = "The scene contains" in gated or "No objects detected" in gated or "interaction is unclear" in gated

    print("\nHallucination Status:")
    if is_accepted:
        print("  \u2713 grounded")
        print("  \u2713 accepted")
    elif is_fallback:
        print("  \u2717 ungrounded phrases detected")
        print("  \u2717 fallback caption used")
    elif is_repaired:
        print("  \u2713 grounded")
        print("  \u2717 repaired (unsupported content removed)")
    else:
        print("  \u2717 rejected")

    print()
    print(f"{'=' * 50}")


# ---------------------------------------------------------------------------
# Final dataset summary printer
# ---------------------------------------------------------------------------

def print_dataset_summary(
    all_results: List[Dict],
    total_rels: int,
    total_sems: int,
    total_anim: int,
    total_rev: int,
    weak_spatial_rate: float = 0.0,
    reversed_direction_rate: float = 0.0,
    num_images: int = 0,
) -> None:
    """Print a professional final evaluation summary after all images."""
    total_verified = sum(len(r.get("detections", [])) for r in all_results)
    total_raw = sum(len(r.get("raw_detections", [])) for r in all_results)
    total_rejected = total_raw - total_verified
    total_hallucinations_prevented = sum(
        1 for r in all_results
        if r.get("raw_caption", "") != r.get("caption", "")
    )

    sem_ratio = total_sems / max(total_rels, 1)

    section("PIPELINE SUMMARY", width=50)
    print()
    print(f"  Images Processed:          {num_images or len(all_results)}")
    print(f"  Verified Objects:          {total_verified}")
    print(f"  Rejected Detections:       {total_rejected}")
    print(f"  Semantic Relations:        {total_rels}")
    print(f"  Hallucinations Prevented:  {total_hallucinations_prevented}")
    print()
    print(f"  Semantic Precision:        {sem_ratio:.2%}")
    print(f"  Semantic Ratio:            {sem_ratio:.2f}")
    print(f"  Weak Spatial Rate:         {weak_spatial_rate:.2%}")
    print(f"  Reversed Direction Rate:   {reversed_direction_rate:.2%}")
    print()
    print(f"{'=' * 50}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unique_labels(detections: List[Dict]) -> set:
    """Return set of unique detection labels."""
    return {d["label"] for d in detections}
