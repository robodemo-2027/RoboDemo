# Evaluation Scale And GPU Memory

All five checkpoints use one environment and policy batch size 1 per worker.
Parallelism is **independent episodes**, not DDP or model sharding. Every
worker holds its own model and simulator. Each completed episode runs the
same 500-control-step protocol.

## One Or Eight GPUs

Every checkpoint has two launchers:

| Model | One GPU | Eight GPUs |
| --- | --- | --- |
| Full50 240k | `scripts/evaluate/full50-single.sh` | `scripts/evaluate/full50-8gpu.sh` |
| C15 90k | `scripts/evaluate/c15-single.sh` | `scripts/evaluate/c15-8gpu.sh` |
| C30 160k | `scripts/evaluate/c30-single.sh` | `scripts/evaluate/c30-8gpu.sh` |
| C45 260k | `scripts/evaluate/c45-single.sh` | `scripts/evaluate/c45-8gpu.sh` |
| C15-wo 100k | `scripts/evaluate/c15-wo-single.sh` | `scripts/evaluate/c15-wo-8gpu.sh` |

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate/c15-single.sh \
  --checkpoint checkpoints/RoboDemo-c15/pytorch_model.pt \
  --data-root "$ROBODEMO_DATA_ROOT" \
  --workers-per-gpu 1 --max-episodes 10 --output outputs/c15-small

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/evaluate/c15-8gpu.sh \
  --checkpoint checkpoints/RoboDemo-c15/pytorch_model.pt \
  --data-root "$ROBODEMO_DATA_ROOT" \
  --workers-per-gpu 1 --output outputs/c15-full
```

Replace GPU IDs with the devices allocated to your job. Omit `--max-episodes`
for the complete plan: 2,000 episodes for Full50 or 400 for each other model.
`--max-episodes` is a **global total**, not a per-GPU limit. Use `--split seen`,
`--split unseen`, or `--task-level L0_TwoRobotFoldTowel-v1` to narrow the plan.
`--dry-run` prints assignments without creating workers or loading weights.

## Memory Sizing

| GPU capacity | `--workers-per-gpu` | Suggested first run |
| --- | ---: | --- |
| 24 GB | **1** | `--max-episodes 1`; lower-capacity option for the measured setup |
| 32 GB | **1** | `--max-episodes 1`; recommended entry configuration |
| 80 GB | **2** | `--max-episodes 2`; budget about 43 GiB for two measured workers |

These are capacity recommendations derived from **H200 process-memory
measurements**, including the renderer, rather than native 24/32/80-GB
card tests. Use 24 GB as the minimum practical target for the measured
configuration and **32 GB or more for comfortable evaluation**. Rendering
and reference-video differences can change the peak across tasks.
The two-worker configuration also completed a real concurrent Full50 run
on H200, described below.

### Policy Example

The Full50 human-video example completed both policy calls on H200
under **18, 32 and 80 GiB PyTorch allocation budgets**. Budgets of 16 and
17 GiB failed with allocator OOM; all failures are retained. Its peak
PyTorch allocation was **17.48 GiB**, with sampled process memory up to
**18.23 GiB**.

### Complete Simulator Episodes

All five released checkpoints completed a **500-step** episode on H200,
with strict loading and the unchanged renderer and policy settings:

| Checkpoint | Peak PyTorch allocation (GiB) | Peak total process (GiB) | Control steps |
| --- | ---: | ---: | ---: |
| Full50 240k | 17.62 | 21.38 | 500 |
| C15 90k | 17.62 | 21.38 | 500 |
| C30 160k | 17.62 | 21.38 | 500 |
| C45 260k | 17.62 | 21.38 | 500 |
| C15-wo 100k | 17.62 | 21.37 | 500 |

The scene was `L0_TwoRobotFoldTowel-v1_trial00`, batch size 1, 224-by-224 RGB,
PhysX CPU simulation, and `rt-fast` with OptiX denoising. Each run completed
500 policy calls and 499 progress-memory updates. Process memory was sampled
every 0.5 seconds. [Download the measurements](validation/20261119-five-model-memory.json).

Earlier complete Full50 simulator episodes also passed under both
24 and 32 GiB PyTorch budgets: **17.62 GiB** peak allocated and up to
**21.39 GiB** sampled total process memory including the renderer. See the
[validation record](validation/20261119.json).

The allocator budget does not cap Vulkan or driver allocations; the
**total process peak** is the sizing quantity. An 18-GiB allocator budget
does not mean full evaluation fits on an 18-GB card.

### Concurrent Workers

Full50 completed **four 500-step episodes on two H200s, with two workers
per GPU**, using the published launcher and result merge. All four episodes
completed 500 policy calls and 499 progress-memory updates. The peak
combined process memory was **42.77 GiB per GPU**, including both renderers.
This supports the **two-worker starting configuration for 80 GB**.

[Concurrent-run measurements](validation/20261119-concurrency.json).

### Profile An Episode

To reproduce a measured complete-episode profile:

```bash
CUDA_VISIBLE_DEVICES=0 python environments/profile_memory.py \
  --budget-gib 18 --output outputs/full50-memory.json -- \
  evaluation/evaluate.py --model full50 \
  --checkpoint checkpoints/RoboDemo-full50/pytorch_model.pt \
  --data-root "$ROBODEMO_DATA_ROOT" \
  --task-level L0_TwoRobotFoldTowel-v1 --max-episodes 1 \
  --output outputs/full50-profiled
```

### Choose The Evaluation Scale

Start with enough *free* VRAM, not merely the card's nominal capacity.
On a shared 80-GB card, use one worker if two copies and a safety margin
do not fit in the available memory. Reducing the number of episodes changes
runtime, **not per-worker VRAM**. Do not change token counts, action horizon,
image resolution or precision to reproduce the reported protocol.

The eight-GPU launcher uses exactly the same independent worker per device.
Its sharding and merge logic have CPU tests and the two-GPU integration
run above. End-to-end execution on eight physical GPUs is not yet validated;
the available H200 host has two GPUs.

## Restart And Merge

Append `--resume` to the same command and output directory to continue after
an interruption. The saved plan, checkpoint file identity, data path and GPU
assignment must match. Both completed successes and completed task failures
are retained. An incomplete technical run is archived and rerun with its
original seeds; it is never counted as a successful episode.

Each worker writes its own log, `run.json`, incremental `summary.json` and
per-episode outputs. After every worker exits successfully, the launcher
checks for missing/duplicate trials and writes combined `summary.json` and
`episodes.jsonl` at the output root. Empty shards load no model.
