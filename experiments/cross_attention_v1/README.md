markdown

\# Cross Attention V1



\## Goal

Improve representation quality beyond adaptive calibration.



\## Hypothesis

Current bottleneck is weak interaction modeling between:

\- subject features

\- object features

\- union features

\- geometry features



The current MLP/Transformer mostly concatenates features instead of reasoning over them relationally.



\## Planned Architecture

Introduce cross-attention:

\- subject attends to object

\- object attends to union

\- geometry attends to semantic embeddings



\## Frozen Constraints

Must outperform stable\_baseline on:

\- riding

\- sitting on

\- carrying



WITHOUT:

\- increasing weak spatial false positives

\- relaxing calibration thresholds further



\## Evaluation Rules

Always compare against:

stable\_baseline/best\_analysis



\## Success Criteria

\- higher semantic recall

\- preserved hallucination suppression

\- improved geometry-heavy reasoning





After that, your next real engineering task is:



text

Replace feature concatenation with tokenized relation attention.

