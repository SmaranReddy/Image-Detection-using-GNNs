# Stable Baseline

## Best Configuration
- Model: geo_dropout_0.1
- Calibration: adaptive predicate-aware calibration
- Status: stable baseline

## Key Achievements
- riding recovered successfully
- hallucination suppression preserved
- weak spatial predicates remain rejected
- strong semantic predicates recover correctly

## Important Findings
- Previous bottleneck was overconservative calibration
- Current bottleneck is representation quality
- Adaptive margins solved semantic false negatives

## Frozen Rules
DO NOT MODIFY:
- calibration constants
- predicate families
- adaptive margin logic
- geo dropout setting

All future experiments must branch from this baseline.

## Next Research Direction
Cross-attention relation encoder:
- subject attends to object
- object attends to union
- geometry attends to semantic features
- richer interaction reasoning
