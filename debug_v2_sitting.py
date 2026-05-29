"""
Minimal isolated V2 debug on sitting.jpg — match evaluate_versions.py exactly.
Prints ALL debug output to see where person->sitting on->chair gets dropped.
"""
import os, sys, gc, torch
sys.path.insert(0, ".")

# Do NOT set REL_CKPT_DIR — match evaluate_versions.py behavior
from utils.logger_utils import set_debug
set_debug(True)

from relation_prediction.predict import infer_relationships_semantic
from utils.yolo_detector import load_model, run_inference, format_detections
from utils.detection_verifier import verify_detections

# Reset
import relation_prediction.predict as rp
rp._model = None
rp._label_vocab = None
rp._pred_vocab = None
rp._device = None
rp._clip_model = None
rp._pose_model = None
rp._model_clip_dim = 0
rp._model_pose_dim = 0
rp._model_pose_object_dim = 0
rp._model_union_dim = 0
rp._model_type = "mlp"
gc.collect()
torch.cuda.empty_cache()

# Load V2
ckpt_dir = "./checkpoints_v2"
rp.load_relation_model(ckpt_dir)

# Detect
from PIL import Image
image = Image.open("test_images/sitting.jpg").convert("RGB")
yolo = load_model()
raw = run_inference(yolo, image)
raw_detections = format_detections(raw, conf_thres=0.5)
detections = verify_detections(raw_detections, image, debug=False)

print(f"\n{'='*60}")
print(f"RUNNING INFERENCE: V2 on sitting.jpg")
print(f"{'='*60}\n")

relations, debug_info = infer_relationships_semantic(
    detections, threshold=0.05, top_k=3,
    img_w=image.width, img_h=image.height,
    image=image, temperature=2.0, debug=True,
)

print(f"\n{'='*60}")
print(f"FINAL RELATIONS ({len(relations)}):")
print(f"{'='*60}")
for r in relations:
    print(f"  {r['subject']} -> {r['predicate']} -> {r['object']}  "
          f"(conf={r['confidence']:.4f}, adjusted={r.get('adjusted_confidence',0):.4f})")
print()

if not relations or relations[0]["predicate"] != "sitting on":
    print("BUG CONFIRMED: V2 did NOT produce 'sitting on'")
    print("\nCandidate trace:")
    for d in debug_info:
        s, o = d["subject"], d["object"]
        bp = d.get("best_predicate", "N/A")
        st = d["status"]
        if st == "candidate":
            sel = " <<<" if any(r["subject"]==s and r["object"]==o and r["predicate"]==bp for r in relations) else ""
            print(f"  CANDIDATE: {s} {bp} {o} final={d.get('best_final_score',0):.4f}{sel}")
        elif st != "candidate":
            print(f"  {st}: {s} {bp} {o} ({d.get('reject_reason', d.get('reason', ''))})")
else:
    print("V2 correctly produces 'person -> sitting on -> chair'")
