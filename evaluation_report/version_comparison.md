# V0 vs V1 vs V2 Evaluation Report
Generated: 2026-05-27 17:14:07
## Configuration
- **V0**: `./checkpoints`- **V2**: `./checkpoints_v2`- **Image directory**: `test_images`
## Aggregate Predicate Distribution
| Metric | V0 | V2 ||--------|---|---|| total_relations | 3 | 3 || semantic_count | 3 | 3 || weak_spatial_count | 0 | 0 || neutral_spatial_count | 0 | 0 |### Per-Predicate Distribution
| Predicate | V0 | V2 | Type ||-----------|---|---|------|| carrying | 1 (33.3%) | 1 (33.3%) | semantic || riding | 1 (33.3%) | 0 (0.0%) | semantic || sitting on | 1 (33.3%) | 1 (33.3%) | semantic || wearing | 0 (0.0%) | 1 (33.3%) | semantic |## Qualitative Comparison
| Image | Detections | V0 | V2 ||-------|-----------|---|---|| bicycle | person, bicycle, backpack | riding | wearing || car | car | (none) | (none) || dog | person, dog | carrying | carrying || multiobject | cup, laptop, book | (none) | (none) || sitting | person, potted plant, chair, c | sitting on | sitting on |## Detailed Per-Image Output
### V0
### V2
## Feature Utilization
### V0
- subj_clip: 22.1%
- union_clip: 21.9%
- obj_clip: 21.0%
- obj_label: 13.8%
- subj_label: 10.5%
- geo: 6.5%
- pose: 4.2%
### V2
- subj_clip: 20.7%
- union_clip: 20.2%
- obj_clip: 17.5%
- obj_label: 15.3%
- subj_label: 14.3%
- geo: 4.6%
- pose_object: 3.9%
- pose: 3.4%
## Version Delta
**V2 vs V0:**
- wearing: delta=+33.3%
- sitting on: delta=+0.0%
- carrying: delta=+0.0%
- riding: delta=-33.3%
