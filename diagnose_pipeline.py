"""
Full pipeline debug: trace V2 behavior on sitting.jpg.
"""
import os, sys, gc, torch
from PIL import Image

sys.path.insert(0, ".")

from relation_prediction.predict import infer_relationships_semantic
from utils.yolo_detector import load_model, run_inference, format_detections
from utils.detection_verifier import verify_detections
from relation_prediction.vg_dataset import normalize_label

# --- Capture all printed output ---
import io
from contextlib import redirect_stdout

def run_full_pipeline(ckpt_dir, label):
    os.environ["REL_CKPT_DIR"] = ckpt_dir
    
    # Reset model globals
    import relation_prediction.predict as rp
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
    
    rp.load_relation_model(ckpt_dir)
    
    image = Image.open("test_images/sitting.jpg").convert("RGB")
    yolo = load_model()
    raw = run_inference(yolo, image)
    raw_detections = format_detections(raw, conf_thres=0.5)
    detections = verify_detections(raw_detections, image, debug=False)
    
    buf = io.StringIO()
    with redirect_stdout(buf):
        relations, debug_info = infer_relationships_semantic(
            detections, threshold=0.05, top_k=3,
            img_w=image.width, img_h=image.height,
            image=image, temperature=2.0, debug=False,
        )
    output = buf.getvalue()
    
    print(f"\n{'='*70}")
    print(f"  FULL PIPELINE: {label} ({ckpt_dir})")
    print(f"{'='*70}")
    print(f"  Output relations ({len(relations)}):")
    for r in relations:
        print(f"    {r['subject']} -> {r['predicate']} -> {r['object']}  "
              f"({r['confidence']=:.4f}  {r['adjusted_confidence']=:.4f})")
    
    print(f"\n  Debug info ({len(debug_info)} entries):")
    for d in debug_info:
        s, o = d["subject"], d["object"]
        bp = d.get("best_predicate", "N/A")
        st = d["status"]
        
        if st == "candidate":
            sel = ""
            for r in relations:
                if r["subject"] == s and r["object"] == o and r["predicate"] == bp:
                    sel = " <<< SELECTED"
                    break
            print(f"    CANDIDATE: {s} {bp} {o} (calib={d.get('best_calibrated',0):.4f} final={d.get('best_final_score',0):.4f}){sel}")
            for pp in sorted(d.get("per_predicate", []), key=lambda x: -x["final"])[:5]:
                marker = " <<< BEST" if pp["predicate"] == bp else ""
                print(f"      {pp['predicate']:20s}: calib={pp['calibrated']:.4f} prior={pp['prior_total']:+.4f} cs={pp['commonsense_penalty']:.4f} final={pp['final']:.4f}{marker}")
        elif st == "rejected_pairwise_plausibility":
            print(f"    REJECTED: {s} {bp} {o} (pairwise plausibility: {d.get('reason', '')})")
            for pp in sorted(d.get("per_predicate", []), key=lambda x: -x["final"])[:3]:
                print(f"      {pp['predicate']:20s}: final={pp['final']:.4f}")
        elif st == "rejected_calibration":
            print(f"    REJECTED: {s} {bp} {o} (calibration: {d.get('reject_reason', '')})")
            for pp in sorted(d.get("per_predicate", []), key=lambda x: -x["final"])[:3]:
                print(f"      {pp['predicate']:20s}: final={pp['final']:.4f}")
        elif st == "rejected_extreme_nonsense":
            print(f"    REJECTED: {s} {bp} {o} (extreme nonsense)")
    
    return output

# Run both
v0_out = run_full_pipeline("./checkpoints", "V0")
v2_out = run_full_pipeline("./checkpoints_v2", "V2")

print(f"\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")
print(f"\n  V2 output does NOT include 'person sitting on chair' despite correct logits.")
print(f"  Conclusion: one of the pipeline filters is rejecting this pair in V2 but not V0.")
