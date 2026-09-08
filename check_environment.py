"""Diagnose whether this machine can run the relation experiments.

Run this first on a new machine. It touches nothing and reports one of three
states per item:

    READY     usable now
    MISSING   required for the visual experiment; fix before running it
    OPTIONAL  only needed for an arm you may not be running (pose, captioning)

    python check_environment.py
    python check_environment.py --json env.json

Exit code is 0 when nothing required is MISSING, 1 otherwise, so it can gate a
script.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJ_ROOT = Path(__file__).resolve().parent

READY, MISSING, OPTIONAL = "READY", "MISSING", "OPTIONAL"

# (import name, pip name, required?, what it is for)
PACKAGES: List[Tuple[str, str, bool, str]] = [
    ("torch",        "torch",        True,  "training and inference"),
    ("torchvision",  "torchvision",  True,  "image transforms"),
    ("numpy",        "numpy",        True,  "metrics"),
    ("PIL",          "Pillow",       True,  "image loading and cropping"),
    ("transformers", "transformers", True,  "CLIP vision encoder"),
    ("requests",     "requests",     True,  "prepare_visual_genome.py downloads"),
    ("tqdm",         "tqdm",         True,  "progress bars"),
    ("pytest",       "pytest",       True,  "the 151-test regression suite"),
    ("matplotlib",   "matplotlib",   False, "plots"),
    ("nltk",         "nltk",         False, "CHAIR / POPE caption evaluation"),
    ("ultralytics",  "ultralytics",  False, "YOLO detection (captioning arm)"),
    ("accelerate",   "accelerate",   False, "BLIP captioner"),
    ("mediapipe",    "mediapipe",    False, "pose features (--use-pose only; "
                                              "no wheel on Python 3.13)"),
]


def status_line(name: str, state: str, detail: str = "") -> None:
    mark = {READY: "  OK  ", MISSING: " MISS ", OPTIONAL: " OPT  "}[state]
    print(f"  [{mark}] {name:<26} {detail}")


def check_python() -> Dict:
    v = sys.version_info
    ok = v >= (3, 9)
    print("\nRuntime")
    status_line("Python", READY if ok else MISSING,
                f"{platform.python_version()} ({'>=3.9 required' if not ok else 'ok'})")
    status_line("platform", READY, platform.platform())
    status_line("executable", READY, sys.executable)
    return {"python": platform.python_version(), "ok": ok,
            "platform": platform.platform(), "executable": sys.executable}


def check_packages() -> Tuple[Dict, List[str]]:
    print("\nPackages")
    out: Dict = {}
    missing_required: List[str] = []
    for mod, pip_name, required, purpose in PACKAGES:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            status_line(pip_name, READY, f"{ver:<12} {purpose}")
            out[pip_name] = {"present": True, "version": ver, "required": required}
        except Exception as exc:
            state = MISSING if required else OPTIONAL
            status_line(pip_name, state,
                        f"{'REQUIRED' if required else 'optional'} - {purpose} "
                        f"({type(exc).__name__})")
            out[pip_name] = {"present": False, "required": required,
                             "error": type(exc).__name__}
            if required:
                missing_required.append(pip_name)
    return out, missing_required


def check_torch() -> Dict:
    print("\nAccelerator")
    try:
        import torch
    except Exception:
        status_line("torch", MISSING, "not importable; nothing else to report")
        return {"available": False}
    info: Dict = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    status_line("torch", READY, torch.__version__)
    status_line("CUDA build", READY if torch.version.cuda else OPTIONAL,
                torch.version.cuda or "CPU-only build of torch")
    info["torch_cuda_version"] = torch.version.cuda
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        info["device_count"] = n
        info["devices"] = []
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            mem = props.total_memory / (1024 ** 3)
            status_line(f"GPU {i}", READY,
                        f"{props.name} - {mem:.1f} GB, sm_{props.major}{props.minor}")
            info["devices"].append({"name": props.name, "total_memory_gb": round(mem, 1),
                                    "capability": f"{props.major}.{props.minor}"})
    else:
        # Not MISSING: everything runs on CPU, just slowly. The GPU is a
        # throughput concern, not a correctness one.
        status_line("GPU", OPTIONAL,
                    "no CUDA device - runs on CPU (CLIP cache build will be slow)")
    return info


def check_repo_modules() -> Tuple[Dict, List[str]]:
    print("\nProject modules")
    sys.path.insert(0, str(PROJ_ROOT))
    mods = [
        ("relation_prediction.vg_dataset", True, "dataset, geometry, predicates"),
        ("relation_prediction.model", True, "RelationMLP"),
        ("relation_prediction.relation_transformer", True, "RelationTransformer"),
        ("relation_prediction.clip_extractor", True, "CLIP crop encoder"),
        ("relation_prediction.clip_cache", True, "feature cache format"),
        ("relation_prediction.predict", False, "inference helpers"),
    ]
    out: Dict = {}
    broken: List[str] = []
    for name, required, purpose in mods:
        try:
            importlib.import_module(name)
            status_line(name.split(".")[-1], READY, purpose)
            out[name] = True
        except Exception as exc:
            status_line(name.split(".")[-1], MISSING if required else OPTIONAL,
                        f"{type(exc).__name__}: {exc}")
            out[name] = False
            if required:
                broken.append(name)
    return out, broken


def check_data(vg_root: Path, clip_cache: Optional[Path]) -> Tuple[Dict, List[str]]:
    print("\nDataset assets")
    out: Dict = {}
    missing: List[str] = []

    for fname, required, purpose in [
        ("relationships.json", True, "relation triples (~710 MB)"),
        ("image_data.json", True, "per-image URLs and sizes (~18 MB)"),
    ]:
        p = vg_root / fname
        if p.is_file():
            size = p.stat().st_size / (1024 ** 2)
            status_line(fname, READY, f"{size:,.0f} MB  {purpose}")
            out[fname] = {"present": True, "size_mb": round(size, 1)}
        else:
            status_line(fname, MISSING, f"{purpose} - run: python download_vg.py")
            out[fname] = {"present": False}
            missing.append(fname)

    manifest = PROJ_ROOT / "splits" / "e0_image_split.json"
    if manifest.is_file():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        n = sum(len(m[k]) for k in ("train_ids", "val_ids", "test_ids"))
        status_line("split manifest", READY, f"{n:,} images (frozen, seed {m.get('seed')})")
        out["split_manifest"] = {"present": True, "n_images": n}
    else:
        status_line("split manifest", MISSING, "splits/e0_image_split.json")
        out["split_manifest"] = {"present": False}
        missing.append("split manifest")

    # images
    image_dir = vg_root / "images"
    if image_dir.is_dir() and manifest.is_file():
        sys.path.insert(0, str(PROJ_ROOT))
        from prepare_visual_genome import find_local, required_image_ids
        _, combined = required_image_ids(manifest)
        present = sum(1 for i in combined if find_local(i, image_dir) is not None)
        pct = 100.0 * present / max(len(combined), 1)
        state = READY if present == len(combined) else MISSING
        status_line("VG images", state,
                    f"{present:,}/{len(combined):,} required present ({pct:.1f}%)"
                    + ("" if state == READY
                       else " - run: python prepare_visual_genome.py --download"))
        out["images"] = {"present": present, "required": len(combined),
                         "coverage_pct": round(pct, 2)}
        if state == MISSING:
            missing.append("VG images")
    else:
        status_line("VG images", MISSING, f"{image_dir} not found")
        out["images"] = {"present": 0}
        missing.append("VG images")

    # clip cache
    cache = clip_cache or (vg_root / "clip_cache_proper.pt")
    if cache.is_file():
        size = cache.stat().st_size / (1024 ** 2)
        try:
            from relation_prediction.clip_cache import ClipCache
            c = ClipCache.load(str(cache))
            status_line("CLIP cache", READY,
                        f"{len(c):,} embeddings, {size:,.0f} MB, dim "
                        f"{c.embeddings.shape[-1] if c.embeddings is not None else '?'}")
            out["clip_cache"] = {"present": True, "n": len(c), "size_mb": round(size, 1)}
        except Exception as exc:
            status_line("CLIP cache", MISSING, f"unreadable: {type(exc).__name__}")
            out["clip_cache"] = {"present": False, "error": str(exc)}
            missing.append("CLIP cache")
    else:
        status_line("CLIP cache", MISSING,
                    f"{cache} - run: python build_clip_cache.py --include-union")
        out["clip_cache"] = {"present": False}
        missing.append("CLIP cache")

    # frozen baselines that the GPU run is checked against
    for rel, label in [("results/e0/relation_mlp.json", "E0 baseline result"),
                       ("results/e2/e2_geo_ext.json", "E2 baseline result")]:
        p = PROJ_ROOT / rel
        if p.is_file():
            r = json.loads(p.read_text(encoding="utf-8"))
            top1 = r.get("metrics", {}).get("top1")
            status_line(label, READY, f"top1 {top1}")
            out[rel] = {"present": True, "top1": top1}
        else:
            status_line(label, OPTIONAL, f"{rel} not found")
            out[rel] = {"present": False}

    # checkpoints used by the baseline reproduction check
    for d, label in [("checkpoints", "E0 checkpoint"),
                     ("checkpoints_e2_geo", "E2 checkpoint")]:
        p = PROJ_ROOT / d / "relation_mlp.pt"
        state = READY if p.is_file() else OPTIONAL
        status_line(label, state,
                    f"{d}/relation_mlp.pt" + ("" if state == READY else " (gitignored; "
                                              "copy it over to reproduce the baseline)"))
        out[d] = {"present": p.is_file()}

    return out, missing


def check_nltk() -> Dict:
    """WordNet is a separate download that pip does not perform."""
    try:
        import nltk
    except Exception:
        return {"nltk": False}
    try:
        nltk.data.find("corpora/wordnet.zip")
        status_line("NLTK wordnet", READY, "caption evaluation can run")
        return {"nltk": True, "wordnet": True}
    except Exception:
        status_line("NLTK wordnet", OPTIONAL,
                    "not downloaded - python -c \"import nltk; "
                    "nltk.download('wordnet'); nltk.download('omw-1.4')\"")
        return {"nltk": True, "wordnet": False}


def check_disk(vg_root: Path) -> Dict:
    print("\nDisk")
    target = vg_root if vg_root.exists() else PROJ_ROOT
    total, used, free = shutil.disk_usage(target)
    free_gb = free / (1024 ** 3)
    # 27,145 images at ~110 KB plus a ~380 MB feature cache, with headroom.
    need_gb = 12.0
    state = READY if free_gb >= need_gb else MISSING
    status_line("free space", state,
                f"{free_gb:.1f} GB free on {target.anchor or target} "
                f"(~{need_gb:.0f} GB needed for images + cache)")
    return {"free_gb": round(free_gb, 1), "needed_gb": need_gb, "ok": state == READY}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vg-root", type=Path,
                    default=PROJ_ROOT / "data" / "visual_genome")
    ap.add_argument("--clip-cache", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None,
                    help="Write the full report as JSON")
    args = ap.parse_args()

    print("=" * 74)
    print("Environment check - relation prediction experiments")
    print("=" * 74)
    print(f"  repo: {PROJ_ROOT}")

    report: Dict = {}
    report["runtime"] = check_python()
    report["packages"], missing_pkgs = check_packages()
    report["accelerator"] = check_torch()
    report["modules"], broken_mods = check_repo_modules()
    report["data"], missing_data = check_data(args.vg_root, args.clip_cache)
    report["nltk"] = check_nltk()
    report["disk"] = check_disk(args.vg_root)

    print("\n" + "=" * 74)
    blockers = missing_pkgs + broken_mods + missing_data
    if not report["runtime"]["ok"]:
        blockers.append("Python >= 3.9")
    if not report["disk"]["ok"]:
        blockers.append("disk space")

    if blockers:
        print("STATUS: NOT READY for the visual experiment")
        print("\nBlocking items:")
        for b in blockers:
            print(f"  * {b}")
        print("\nThe geometry-only baseline may still reproduce; the visual arms "
              "need the items above.")
    else:
        print("STATUS: READY - every required item is present.")
    print("=" * 74)

    report["blockers"] = blockers
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nreport written to {args.json}")

    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
