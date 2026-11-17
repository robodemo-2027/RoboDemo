# Robot References

The C15-wo evaluation uses a query-conditioned Refreward model. The query is
the instruction and the initial RGB observation. The candidate bank contains
160-165 eligible robot training demonstrations for each evaluation task. The selected
reference remains fixed for the episode.

## Reproducing The Reported Episodes

The default evaluation command reads `configs/robot_reference_selections.json`.
It contains the selected reference and all candidate scores for each of the
400 published reset/policy seeds. The policy then encodes that reference video
and performs the closed-loop rollout. It does not need the candidate feature
cache to replay these reference selections.

## New Initial Observations

`evaluation/retrieval.py` provides `RobotReferences`, with the same scoring
and reference-encoding operations used by the recorded experiment.
It expects the selected step-6,000 Refreward checkpoint and precomputed
candidate features:

```bash
export ROBODEMO_RETRIEVER=/models/refreward/retriever.safetensors
export ROBODEMO_RETRIEVAL_QWEN=/models/Qwen3-VL-Embedding-2B
export ROBODEMO_RETRIEVAL_DINO=/models/dinov3-vith16plus-pretrain-lvd1689m
export ROBODEMO_CANDIDATE_FEATURES=/data/robot_candidate_features
export ROBODEMO_DATA_ROOT=/data/IG-10K-Dataset
```

Candidate feature filenames are `<candidate_id>.safetensors`; the tensors
follow `models/robot/refreward/features.py`. Feature extraction code is in
`extractors.py` and `preprocessing.py`. The candidate cache and auxiliary
retriever weight are not contained in the GitHub code tree.

Unseen tasks are held out from policy training, not from the complete robot
reference bank. The retrieval supervision and policy supervision therefore
have different task coverage.
