# Read this before citing anything in `checkpoints/`

`checkpoints/relation_mlp.pt` is the **E0 baseline**: a geometry-only 5-D MLP,
bare `state_dict`, 75,029 parameters, `input_dim = 133`. It reproduces
`results/e0/relation_mlp.json` (Top-1 **0.553437**) exactly, and
`eval_gt_relations.py` reconstructs it from the weights themselves.

The four JSON files sitting beside it **do not describe that checkpoint**:

| file | describes |
|---|---|
| `training_meta.json` | a **visual transformer** run: `clip_dim=768`, pose + union features, `dataset_size=37658` |
| `training_logs.json` | the per-epoch curve of that same other run |
| `validation_metrics.json` | that run's per-predicate validation metrics |
| `confusion_analysis.json` | that run's confusion counts |

## How they drifted apart

`.gitignore` tracked `checkpoints/*.json` while excluding `*.pt`. So each new
training run overwrote the JSONs in place, but the weights they were written
alongside never entered version control and were eventually replaced on disk by
a different model. The JSONs are the residue of the last run that wrote them,
not a description of the `.pt` that is here now.

## What this does and does not invalidate

* **The E0 number is fine.** `results/e0/relation_mlp.json` was produced by
  evaluating the actual weights on the frozen split. Nothing in it comes from
  these JSONs, and the audit reproduced it bit-for-bit.
* **The training provenance of E0 is unknown.** Do not cite the hyperparameters,
  epoch count, learning rate, or `dataset_size` in `training_meta.json` as the
  configuration that produced the E0 checkpoint. They belong to a different run.

These files are kept, unmodified, because overwriting or deleting a historical
artefact to tidy up a record is worse than labelling it. If you need a
provenance-complete run, use `run_visual_experiment.py`, which writes a
`run.json` recording the commit, split hash, seed and full configuration
alongside every result.

`checkpoints_e2_geo/` does not have this problem: its `training_meta.json` was
written by the run that produced its `relation_mlp.pt` (E2, 19-D geometry,
Top-1 **0.638897**).
