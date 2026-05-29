"""
Diagnose V2's regression on sitting.jpg.
"""
import os, sys, gc, torch
from PIL import Image

os.environ["REL_CKPT_DIR"] = "./checkpoints_v2"
sys.path.insert(0, ".")

# Clear cached models
import relation_prediction.predict as rp
from utils.yolo_detector import load_model, run_inference, format_detections
from utils.detection_verifier import verify_detections

image = Image.open("test_images/sitting.jpg").convert("RGB")
img_w, img_h = image.size

# Detect
yolo = load_model()
raw = run_inference(yolo, image)
raw_detections = format_detections(raw, conf_thres=0.5)
detections = verify_detections(raw_detections, image, debug=False)

print(f"Image: {img_w}x{img_h}")
print(f"Detections ({len(detections)}):")
for d in detections:
    print(f"  {d['label']}: box={d['box']}")

def run_diagnostic(checkpoint_dir, label):
    """Run detailed diagnostic for a checkpoint."""
    rp._model = None
    rp._label_vocab = None
    rp._pred_vocab = None
    rp._clip_model = None
    rp._pose_model = None
    rp._model_clip_dim = 0
    rp._model_pose_dim = 0
    rp._model_pose_object_dim = 0
    rp._model_union_dim = 0
    rp._model_type = "mlp"
    gc.collect()
    torch.cuda.empty_cache()
    
    rp.load_relation_model(checkpoint_dir)
    
    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC: {label}")
    print(f"{'='*70}")
    
    # For each pair, get detailed logits
    n = len(detections)
    results = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = detections[i], detections[j]
            subj, obj = a["label"], b["label"]
            
            logits, pred_tokens, _, _ = rp._get_raw_logits(
                subj, obj, a["box"], b["box"],
                img_w=img_w, img_h=img_h, image=image,
            )
            if logits is None:
                continue
            
            # Get top-5 predicates by adjusted logit
            top5_idx = logits.argsort(descending=True)[:5]
            top5 = [(pred_tokens[idx.item()], logits[idx].item()) for idx in top5_idx]
            
            # Temperature-calibrated softmax
            import torch.nn.functional as F
            calibrated = F.softmax(logits / 2.0, dim=-1)
            
            # Get top-5 by calibrated score
            calib_top5_idx = calibrated.argsort(descending=True)[:5]
            calib_top5 = [(pred_tokens[idx.item()], calibrated[idx].item()) for idx in calib_top5_idx]
            
            results.append({
                "pair": f"{subj} -> {obj}",
                "top5_logits": top5,
                "top5_calibrated": calib_top5,
            })
            
            print(f"\n  Pair: {subj} -> {obj}")
            print(f"  Top-5 by adjusted logit:")
            for pname, rval in top5:
                marker = " [SEM]" if pname in rp.SEMANTIC_PREDS else ""
                print(f"    {pname:20s}: {rval:+8.4f}{marker}")
            print(f"  Top-5 by calibrated (T=2.0):")
            for pname, pval in calib_top5:
                marker = " [SEM]" if pname in rp.SEMANTIC_PREDS else ""
                print(f"    {pname:20s}: {pval:8.4f}{marker}")
    
    return results

# Run V0
v0_results = run_diagnostic("./checkpoints", "V0")

# Run V2
v2_results = run_diagnostic("./checkpoints_v2", "V2")

# Compare key pairs
print(f"\n{'='*70}")
print(f"  KEY COMPARISON: person + chair (should be 'sitting on')")
print(f"{'='*70}")

for label, results in [("V0", v0_results), ("V2", v2_results)]:
    for r in results:
        if r["pair"] == "person -> chair":
            print(f"\n  {label}:")
            for pname, rval in r["top5_logits"]:
                cval = None
                for pn2, cv in r["top5_calibrated"]:
                    if pn2 == pname:
                        cval = cv
                        break
                marker = " [SEM]" if pname in rp.SEMANTIC_PREDS else ""
                print(f"    {pname:20s}: logit={rval:+8.4f}  calibrated={cval:.4f}{marker}")

print(f"\n{'='*70}")
print(f"  KEY COMPARISON: person + potted plant (distractor)")
print(f"{'='*70}")

for label, results in [("V0", v0_results), ("V2", v2_results)]:
    for r in results:
        if r["pair"] == "person -> potted plant":
            print(f"\n  {label}:")
            for pname, rval in r["top5_logits"]:
                cval = None
                for pn2, cv in r["top5_calibrated"]:
                    if pn2 == pname:
                        cval = cv
                        break
                marker = " [SEM]" if pname in rp.SEMANTIC_PREDS else ""
                print(f"    {pname:20s}: logit={rval:+8.4f}  calibrated={cval:.4f}{marker}")
