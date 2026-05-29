import os, sys, gc, torch
os.environ["REL_CKPT_DIR"] = "./checkpoints_v2"
sys.path.insert(0, ".")

from PIL import Image
import relation_prediction.predict as rp
from utils.yolo_detector import load_model, run_inference, format_detections
from utils.detection_verifier import verify_detections

rp._model = None; rp._label_vocab = None; rp._pred_vocab = None
rp._device = None; rp._clip_model = None; rp._pose_model = None
rp._model_clip_dim = 0; rp._model_pose_dim = 0; rp._model_pose_object_dim = 0
rp._model_union_dim = 0; rp._model_type = "mlp"
gc.collect(); torch.cuda.empty_cache()
rp.load_relation_model("./checkpoints_v2")

yolo = load_model()
image = Image.open("test_images/sitting.jpg").convert("RGB")
raw = run_inference(yolo, image)
raw_detections = format_detections(raw, conf_thres=0.5)
detections = verify_detections(raw_detections, image, debug=False)

relations, debug = rp.infer_relationships_semantic(
    detections, threshold=0.05, top_k=3, image=image, temperature=2.0, debug=False)
print(f"Relations ({len(relations)}):")
for r in relations:
    print(f"  {r['subject']} -> {r['predicate']} -> {r['object']}  (adj_conf={r.get('adjusted_confidence',0):.4f})")
