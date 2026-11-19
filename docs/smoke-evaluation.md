# Simulator Smoke Evaluation

All five released checkpoints completed one 500-step Imitator Game episode
on NVIDIA H20 using the public evaluator on November 18, 2026.
This checks the evaluation pipeline; the full benchmark results remain in
[results](../results/README.md).

| Model | Checkpoint | Pipeline | Task Success | First Success Step |
| --- | ---: | --- | --- | ---: |
| full50 | 240,000 | Passed | Yes | 345 |
| c15 | 90,000 | Passed | Yes | 341 |
| c30 | 160,000 | Passed | Yes | 341 |
| c45 | 260,000 | Passed | Yes | 344 |
| c15-wo | 100,000 | Passed | Yes | 339 |

The test case was selected before running the models:
`L0_TwoRobotFoldTowel-v1_trial00`, reset seed `20261407`, policy seed
`202650070`. Every run loaded its checkpoint strictly, completed 500 policy
calls and 499 execution-memory updates, and saved a 501-frame video with a
complete trajectory. Actions were finite and within the controller bounds.
C15-wo used its published recorded robot-reference selection.

The environment used Python 3.10.20, PyTorch 2.6.0 / CUDA 12.4,
Transformers 4.57.6, SAPIEN 3.0.3 and Gymnasium 0.29.1.
The human-reference and robot-reference CPU suites each passed 51 tests.

To run the same trial after [installation and data setup](../README.md):

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate.py \
  --model c15 \
  --checkpoint /models/RoboDemo-c15/pytorch_model.pt \
  --data-root "$ROBODEMO_DATA_ROOT" \
  --task-level L0_TwoRobotFoldTowel-v1 \
  --max-episodes 1 \
  --output outputs/c15-smoke
```

Select the other model names and their corresponding checkpoints to repeat
the remaining runs. Use a new output directory for each run.
[Machine-readable records](smoke-evaluation.json) include per-run outcomes
and timings. These five episodes are separate from the published benchmark
statistics and do not measure fresh-query retrieval for C15-wo.

## H200 Follow-Up

On November 19, all five checkpoints also completed the same 500-step trial
on H200 with NVIDIA driver 570.195.03. Every checkpoint loaded strictly and
completed 500 policy calls and 499 execution-memory updates. The measured
total process peaks were 21.38 GiB for Full50/C15/C30/C45 and 21.37 GiB for
C15-wo, including rendering. All five trials reached the success condition.

The human-reference and robot-reference CPU suites each passed 100 tests.
[H200 records](validation/20261119-five-model-memory.json) and
[24/32/80-GB sizing guidance](evaluation-scaling.md) accompany these runs.

A separate Full50 integration run completed four episodes on two H200s
with two workers per GPU. The launcher merged all four results, and the
peak combined process memory was 42.77 GiB per GPU.
[Concurrent-run record](validation/20261119-concurrency.json).
