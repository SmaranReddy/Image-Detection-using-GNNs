"""
Geometry dropout sweep: train models with geo_dropout_prob ∈ {0.0, 0.1, 0.2, 0.3, 0.4}
and evaluate each on test images.

Usage:
    python run_geo_dropout_sweep.py                    # full sweep (20 epochs each)
    python run_geo_dropout_sweep.py --quick             # fast ablation (5 epochs, small subset)
    python run_geo_dropout_sweep.py --skip-train        # re-evaluate existing checkpoints
    python run_geo_dropout_sweep.py --drops 0.0 0.3     # specific dropout values only
"""

import argparse
import json
import os
import subprocess
import sys
import time

# Same dirs used by analysis.py
EVAL_SCRIPT = os.path.join(os.path.dirname(__file__), "analysis.py")
TRAIN_MODULE = "relation_prediction.train"

VG_ROOT = "./data/visual_genome"
VG_IMAGE_DIR = "./data/visual_genome/images"
CLIP_CACHE_PATH = "./data/visual_genome/clip_cache_proper.pt"

BASE_TRAIN_ARGS = [
    sys.executable, "-m", TRAIN_MODULE,
    "--model", "transformer",
    "--use-visual",
    "--require-visual",
    "--use-pose",
    "--use-pose-object",
    "--use-union",
    "--enable-geometry-dropout",
    "--vg-root", VG_ROOT,
    "--vg-image-dir", VG_IMAGE_DIR,
    "--clip-cache-path", CLIP_CACHE_PATH,
    "--batch-size", "512",
]

SWEEP_DIR = "./checkpoints_sweep"
DROPOUT_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4]
EPOCHS_FULL = 20
EPOCHS_QUICK = 5
MAX_SAMPLES_QUICK = 5000


def train_model(dropout_prob: float, epochs: int, max_samples: int = None):
    ckpt_dir = os.path.join(SWEEP_DIR, f"geo_dropout_{dropout_prob:.1f}")
    os.makedirs(ckpt_dir, exist_ok=True)

    cmd = BASE_TRAIN_ARGS + [
        "--checkpoint-dir", ckpt_dir,
        "--epochs", str(epochs),
        "--geometry-dropout-prob", str(dropout_prob),
    ]
    if max_samples is not None:
        cmd += ["--max-samples", str(max_samples)]

    print(f"\n{'='*70}")
    print(f"  Training geo_dropout={dropout_prob} -> {ckpt_dir}")
    print(f"{'='*70}")
    print(f"  Command: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    elapsed = time.time() - start
    print(f"  Finished in {elapsed:.1f}s (exit code {result.returncode})")
    return result.returncode == 0


def evaluate_all(output_dir: str, test_image_dir: str = "test_images"):
    """Run analysis.py on all sweep checkpoints, comparing against V0/V2."""
    print(f"\n{'='*70}")
    print(f"  Evaluating all sweep models")
    print(f"{'='*70}")
    cmd = [
        sys.executable, EVAL_SCRIPT,
        "--image-dir", test_image_dir,
        "--output-dir", output_dir,
        "--skip-eval",
    ]
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    return result.returncode == 0


def show_summary(output_dir: str):
    report_path = os.path.join(output_dir, "analysis_report.json")
    if not os.path.exists(report_path):
        print(f"  No report found at {report_path}")
        return

    with open(report_path) as f:
        report = json.load(f)

    vnames = sorted(report.get("predicate_analysis", {}).keys())
    print(f"\n  {'='*60}")
    print(f"  GEOMETRY DROPOUT SWEEP SUMMARY")
    print(f"  {'='*60}")
    print(f"  {'Version':<20} {'Geo%':>6} {'Accepted':>9} {'Accept%':>9} {'Conf Mean':>10}")
    print(f"  {'-'*20} {'-'*6} {'-'*9} {'-'*9} {'-'*10}")
    for vn in vnames:
        pa = report["predicate_analysis"].get(vn, {})
        fu = report["feature_utilization"].get(vn, {})
        geo_pct = fu.get("percentages", {}).get("geo", 0)
        total = pa.get("total_pairs_evaluated", 0)
        accepted = pa.get("total_accepted", 0)
        accept_pct = accepted / total * 100 if total > 0 else 0
        ch = report["confidence_histograms"].get(vn, {})
        acc_conf = ch.get("accepted_conf", {})
        conf_mean = acc_conf.get("mean", 0)
        print(f"  {vn:<20} {geo_pct:>5.1f}% {accepted:>4}/{total:<4} {accept_pct:>7.1f}% {conf_mean:>10.4f}")

    print(f"\n  {'='*60}")
    print(f"  FAILURE CATEGORIES")
    print(f"  {'='*60}")
    print(f"  {'Version':<20} {'A':>5} {'B':>5} {'C':>5} {'D':>5}")
    print(f"  {'-'*20} {'-'*5} {'-'*5} {'-'*5} {'-'*5}")
    for vn in vnames:
        fa = report["failure_analysis"].get(vn, {})
        cats = fa.get("category_counts", {})
        print(f"  {vn:<20} {cats.get('A',0):>5} {cats.get('B',0):>5} {cats.get('C',0):>5} {cats.get('D',0):>5}")


def main():
    parser = argparse.ArgumentParser(description="Geometry dropout sweep")
    parser.add_argument("--quick", action="store_true", help="Quick ablation (5 epochs, 5k samples)")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, just evaluate existing")
    parser.add_argument("--drops", type=float, nargs="*", default=None,
                        help="Specific dropout values (default: 0.0 0.1 0.2 0.3 0.4)")
    parser.add_argument("--test-dir", default="test_images")
    parser.add_argument("--output-dir", default="analysis_output")
    args = parser.parse_args()

    dropout_values = args.drops if args.drops else DROPOUT_VALUES

    # Ensure test images exist
    if not os.path.isdir(args.test_dir):
        print(f"ERROR: test image directory not found: {args.test_dir}")
        sys.exit(1)

    if not args.skip_train:
        epochs = EPOCHS_QUICK if args.quick else EPOCHS_FULL
        max_samples = MAX_SAMPLES_QUICK if args.quick else None

        for dp in dropout_values:
            success = train_model(dp, epochs, max_samples)
            if not success:
                print(f"  WARNING: Training failed for geo_dropout={dp}, continuing...")
    else:
        print("Skipping training (--skip-train).")

    # Evaluate: analysis.py merges sweep dirs with existing V0/V2 via CHECKPOINT_DIRS
    print(f"\nSweep complete. Checkpoints saved under {SWEEP_DIR}/")
    print(f"Run `python analysis.py --image-dir {args.test_dir} --output-dir {args.output_dir}` to evaluate.")


if __name__ == "__main__":
    main()
