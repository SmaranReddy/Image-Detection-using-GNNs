"""
InstructBLIP-based caption generator conditioned on a structured scene graph.

Drop-in replacement for generate_causal_caption() — same call signature,
richer output via a vision-language model.

Evidence gating runs after generation:
  1. Low-confidence entity mentions are hedged with "possibly".
  2. Captions with too many unsupported entities fall back to a safe template.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import torch
from PIL import Image
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

Detection    = Dict   # {"label": str, "box": List[float], "score": float}
Relationship = Tuple[str, str, str]   # (subject, relation, object)

# ---------------------------------------------------------------------------
# Model singleton — loaded once on first call
# ---------------------------------------------------------------------------

_processor: InstructBlipProcessor | None = None
_model: InstructBlipForConditionalGeneration | None = None
_device: torch.device | None = None

_MODEL_ID = "Salesforce/instructblip-flan-t5-xl"


def _load_model() -> None:
    global _processor, _model, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype   = torch.float16 if _device.type == "cuda" else torch.float32

    print(f"[blip_captioner] Loading {_MODEL_ID} on {_device} …")
    _processor = InstructBlipProcessor.from_pretrained(_MODEL_ID)
    _model     = InstructBlipForConditionalGeneration.from_pretrained(
        _MODEL_ID, torch_dtype=dtype
    )
    _model.to(_device)
    _model.eval()
    print("[blip_captioner] Model ready.")


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(
    detections: List[Detection],
    relationships: List[Relationship],
) -> str:
    """Serialize scene graph into a strict, grounded prompt."""
    scores = {d["label"]: d.get("score", 0.5) for d in detections}

    obj_lines = [
        f"- {d['label']} ({d.get('score', 0.5):.2f})"
        for d in detections
    ]

    if relationships:
        rel_lines = [
            f"- {subj} {rel.replace('_', ' ')} {obj} "
            f"({min(scores.get(subj, 0.5), scores.get(obj, 0.5)):.2f})"
            for subj, rel, obj in relationships
        ]
    else:
        rel_lines = ["- None"]

    return (
        "You are describing an image based ONLY on verified detections.\n"
        "\n"
        "Detected objects (with confidence):\n"
        + "\n".join(obj_lines)
        + "\n"
        "\nDetected relationships:\n"
        + "\n".join(rel_lines)
        + "\n"
        "\nSTRICT RULES:\n"
        "- Do NOT mention time of day (e.g., night, day, morning, evening, afternoon).\n"
        "- Do NOT mention lighting or weather (e.g., bright, dark, sunny, shadow, rain, fog).\n"
        "- Do NOT introduce objects that are not listed above.\n"
        "- Do NOT infer context beyond the listed objects and relationships.\n"
        "- If uncertain, omit the detail rather than guessing.\n"
        "\n"
        "Write 2 concise sentences describing what is happening using only the listed objects and relationships."
    )


# ---------------------------------------------------------------------------
# Image normalisation helper
# ---------------------------------------------------------------------------

def _to_pil(image) -> Image.Image:
    """Accept a PIL Image or a CHW/HWC torch.Tensor and return a PIL Image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    t = image.float()
    if t.ndim == 3 and t.shape[0] in (1, 3, 4):   # CHW → HWC
        t = t.permute(1, 2, 0)
    t = t.cpu().numpy()

    if t.max() <= 1.0:
        t = (t * 255).clip(0, 255).astype("uint8")
    else:
        t = t.clip(0, 255).astype("uint8")

    if t.shape[2] == 1:
        t = t[:, :, 0]

    return Image.fromarray(t).convert("RGB")


# ---------------------------------------------------------------------------
# Evidence-gating helpers
# ---------------------------------------------------------------------------

# Words that are always legitimate in a scene caption regardless of detections.
# Covers verbs, prepositions, and common scene-description adjectives.
_STOP_WORDS: frozenset = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "to", "of", "in",
    "on", "at", "by", "for", "with", "from", "into", "and", "or", "but",
    "not", "no", "it", "its", "this", "that", "there", "they", "them",
    "what", "which", "who", "when", "where", "why", "how", "all", "both",
    "each", "some", "such", "than", "too", "very", "just", "also",
})

# Action / scene / relationship words — never penalised as unsupported.
_SCENE_WORDS: frozenset = frozenset({
    "riding", "walking", "running", "standing", "sitting", "holding",
    "carrying", "chasing", "playing", "looking", "parked", "near",
    "next", "possibly", "likely", "appears", "seem", "seems", "suggests",
    "suggesting", "indicating", "showing", "observed", "visible",
    "scene", "outdoor", "indoor", "background", "interaction", "unclear",
    "stationary", "moving", "using", "beside", "behind", "front",
    "left", "right",
})

# Ambient / environmental / descriptive words that add colour but are not
# object-level claims. A caption saying "on a sunny afternoon" is not
# hallucinating even if "sunny" is absent from detections.
_SAFE_CONTEXT_WORDS: frozenset = frozenset({
    # environment & weather
    "street", "road", "sidewalk", "pavement", "path", "trail", "lane",
    "park", "field", "yard", "garden", "plaza", "square", "area",
    "sky", "ground", "grass", "tree", "trees", "bush", "bushes",
    "sunny", "cloudy", "rainy", "snowy", "foggy", "bright", "dark",
    "sunny", "overcast", "clear", "windy",
    # time / light
    "day", "daytime", "morning", "afternoon", "evening", "night",
    "sunlight", "shadow", "light", "shade",
    # spatial / directional scene setters
    "outside", "inside", "nearby", "away", "forward", "toward",
    "distance", "foreground", "horizon",
    # general descriptors that never name a new object
    "busy", "quiet", "crowded", "empty", "urban", "rural", "public",
    "open", "narrow", "wide", "flat", "steep", "paved", "dirt",
    "warm", "cool", "wet", "dry",
})

_HEDGE_MARKERS: Tuple[str, ...] = ("possibly", "appears to be", "might be", "may be")


def _build_label_word_set(detections: List[Detection]) -> frozenset:
    """All individual words across every detection label, lower-cased."""
    words: set = set()
    for d in detections:
        label = d["label"].lower().replace("_", " ")
        words.add(label)          # full multi-word label
        words.update(label.split())
    return frozenset(words)


def _extract_content_words(text: str) -> List[str]:
    """
    Lower-case tokens that could represent hallucinated objects.

    Excluded from penalisation:
      - stop words (function words / grammar)
      - scene words (relationship verbs, stance words)
      - safe context words (ambient descriptors: weather, time, terrain)
    Only substantive tokens that name concrete entities remain.
    """
    tokens = re.findall(r"[a-z]+", text.lower())
    return [
        t for t in tokens
        if t not in _STOP_WORDS
        and t not in _SCENE_WORDS
        and t not in _SAFE_CONTEXT_WORDS
        and len(t) > 2
    ]


def _unsupported_fraction(caption: str, label_words: frozenset) -> float:
    """
    Fraction of content words in the caption not covered by any detection label.

    Returns 0.0 when there are no content words (avoids divide-by-zero).
    """
    content = _extract_content_words(caption)
    if not content:
        return 0.0
    unsupported = [w for w in content if w not in label_words]
    return len(unsupported) / len(content)


def _hedge_low_confidence(
    caption: str,
    detections: List[Detection],
    threshold: float = 0.5,
) -> str:
    """
    Prefix 'possibly' before article+label phrases for low-confidence detections.

    Processes longest labels first so "sports ball" is matched before "ball".
    Skips spans already preceded by a hedge marker.
    """
    low_conf = [
        d for d in sorted(detections, key=lambda x: -len(x["label"]))
        if d.get("score", 1.0) < threshold
    ]
    if not low_conf:
        return caption

    result = caption
    for d in low_conf:
        label = d["label"].replace("_", " ")
        pat   = re.compile(
            r'\b(a|an|the)\s+(' + re.escape(label) + r')\b',
            re.IGNORECASE,
        )
        matches = list(pat.finditer(result))
        # Reverse order so earlier offsets stay valid after insertions.
        for m in reversed(matches):
            prefix_chunk = result[max(0, m.start() - 30) : m.start()].lower()
            if not any(hw in prefix_chunk for hw in _HEDGE_MARKERS):
                result = result[: m.start()] + "possibly " + result[m.start():]
    return result


def _safe_fallback(detections: List[Detection]) -> str:
    """Conservative caption listing only confirmed detections."""
    labels = list(dict.fromkeys(
        d["label"].replace("_", " ") for d in detections
    ))
    if not labels:
        return "No objects detected in the scene."
    if len(labels) == 1:
        art = "an" if labels[0][0].lower() in "aeiou" else "a"
        return f"The scene contains {art} {labels[0]}."

    parts = []
    for lbl in labels:
        art = "an" if lbl[0].lower() in "aeiou" else "a"
        parts.append(f"{art} {lbl}")

    if len(parts) == 2:
        enumerated = f"{parts[0]} and {parts[1]}"
    else:
        enumerated = ", ".join(parts[:-1]) + f", and {parts[-1]}"
    return f"The scene contains {enumerated}, but the interaction is unclear."


def gate_caption(
    caption: str,
    detections: List[Detection],
    unsupported_threshold: float = 0.4,
    confidence_threshold: float = 0.5,
) -> str:
    """
    Post-process a raw BLIP caption to reduce hallucinations.

    Steps
    -----
    1. Hedge every article+label mention where detection confidence < threshold.
    2. Compute what fraction of content words are unsupported by any label.
    3. If fraction exceeds unsupported_threshold → replace with safe fallback.

    Args:
        caption:               Raw decoded model output.
        detections:            Detection list used for grounding.
        unsupported_threshold: Maximum tolerated fraction of unsupported content
                               words before the fallback fires (default 0.40).
        confidence_threshold:  Detections below this score get hedged (default 0.50).

    Returns:
        Grounded, clean caption string.
    """
    if not detections:
        return caption

    label_words = _build_label_word_set(detections)

    # Step 1 — hedge low-confidence entities.
    caption = _hedge_low_confidence(caption, detections, threshold=confidence_threshold)

    # Step 2 — fallback on too many unsupported claims.
    if _unsupported_fraction(caption, label_words) > unsupported_threshold:
        return _safe_fallback(detections)

    return caption


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_blip_caption(
    image,
    detections: List[Detection],
    relationships: List[Relationship],
    max_new_tokens: int = 128,
    num_beams: int = 4,
    unsupported_threshold: float = 0.4,
    confidence_threshold: float = 0.5,
) -> str:
    """
    Generate a grounded scene caption using InstructBLIP.

    Runs evidence gating after decoding so the returned string only contains
    claims supported by the provided detections.

    Args:
        image:                 PIL Image or CHW torch.Tensor.
        detections:            [{"label": str, "box": [...], "score": float}, ...]
        relationships:         [(subject, relation, object), ...]
        max_new_tokens:        Token budget for the generated caption.
        num_beams:             Beam-search width; 1 = greedy (fastest).
        unsupported_threshold: Passed to gate_caption (default 0.40).
        confidence_threshold:  Passed to gate_caption (default 0.50).

    Returns:
        A clean, grounded caption string.
    """
    global _processor, _model, _device

    if _model is None:
        _load_model()

    pil_image = _to_pil(image)
    prompt    = build_prompt(detections, relationships)

    inputs = _processor(
        images=pil_image,
        text=prompt,
        return_tensors="pt",
    ).to(_device)

    with torch.inference_mode():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )

    raw_caption = _processor.decode(output_ids[0], skip_special_tokens=True).strip()
    return gate_caption(
        raw_caption,
        detections,
        unsupported_threshold=unsupported_threshold,
        confidence_threshold=confidence_threshold,
    )
