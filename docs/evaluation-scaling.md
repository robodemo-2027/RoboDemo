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
| 32 GB | **1** | `--max-episodes 1` |
| 80 GB | **1 initially; 2 after a one-worker check** | `--max-episodes 2` |

**32 GB is the recommended entry configuration, not a measured hardware
minimum.** The completed full-horizon runs used H20 GPUs with 97,871 MiB
reported capacity. An offline two-call replay of each checkpoint measured
about **17.5 GiB of peak PyTorch allocated memory**. That figure excludes
some CUDA, Vulkan and OptiX allocations and is not a total-VRAM requirement.
A native 32-GB/80-GB validation and an eight-GPU end-to-end run are not yet
recorded. We do not currently claim support below 32 GB.

Start with enough *free* VRAM, not merely the card's nominal capacity. On
an 80-GB card, increase to two workers only after checking the full episode's
observed memory use and leaving room for the renderer. If memory is tight,
reduce `--workers-per-gpu` to 1. Reducing the number of episodes changes
runtime, **not per-worker VRAM**. Do not change token counts, action horizon,
image resolution or precision to reproduce the reported protocol.

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
