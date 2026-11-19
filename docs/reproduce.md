# Reproduce RoboDemo

Choose the shortest path for what you want to try.

| Goal | Start here | Required resources |
| --- | --- | --- |
| Explore the method's inputs | `python examples/predict_actions.py --inspect-only` | Python and NumPy; included sample |
| Predict actions from a human video | [Policy example](../examples/README.md#1-human-demonstration-to-actions) | Full50 checkpoint and policy encoders; NVIDIA GPU |
| Retrieve a robot demonstration | [Retrieval example](../examples/README.md#2-retrieve-a-robot-reference) | Included scorer; benchmark media and retrieval encoders; CPU supported |
| Evaluate a checkpoint | [Quickstart](quickstart.md) | Linux, simulator, assets and one policy checkpoint |
| Run parallel episodes | [GPU sizing and launchers](evaluation-scaling.md) | One model copy and simulator per worker |
| Recompute published numbers | `python evaluation/summarize.py results/*/episodes.jsonl` | Included logs; no weights or GPU |

All five checkpoints share the [evaluation environment](../environments/README.md).
The standard C15-wo evaluation uses the included reference selections; it
does not require running retrieval or calling a labeling service.

## From Installation To One Episode

After completing the [data and encoder setup](quickstart.md#3-prepare-the-benchmark-and-encoders):

```bash
# Choose full50, c15, c30, c45 or c15-wo.
export MODEL=c15
export CUDA_VISIBLE_DEVICES=0

hf download "mxk1998/RoboDemo-${MODEL}" \
  pytorch_model.pt policy.yaml normalization.json \
  --local-dir "checkpoints/RoboDemo-${MODEL}"

python environments/check_environment.py --assets \
  --output "outputs/${MODEL}-environment.json"

bash "scripts/evaluate/${MODEL}-single.sh" \
  --checkpoint "checkpoints/RoboDemo-${MODEL}/pytorch_model.pt" \
  --data-root "$ROBODEMO_DATA_ROOT" \
  --workers-per-gpu 1 --max-episodes 1 \
  --output "outputs/${MODEL}-smoke"
```

Use a GPU allocated to your job and a new output directory. A smoke run
completes the full 500-control-step horizon. Omit `--max-episodes` to run
the complete plan, or increase it for a smaller study.

**Use one worker on 32 GB, or two workers on 80 GB.** These starting
configurations use H200 process-memory measurements. A 24-GB card is the
minimum practical target for the measured setup; 32 GB leaves more rendering
headroom. Episode count changes runtime, not per-worker memory.

## Validation Status

Last reviewed: November 19, 2026.

| Item | Completed | Next validation |
| --- | --- | --- |
| Five policy checkpoints | Strict loading and one complete simulator episode per checkpoint on H20 and H200 | [Recorded runs](smoke-evaluation.md) and [H200 measurements](validation/20261119-five-model-memory.json) |
| Human-video policy example | Two real policy calls; memory ages `[0, 1]` | [Example outputs](validation/20261119.json) |
| Raw-media retrieval example | Both encoders and scorer ran on CPU; selected episode 3836 | [Inputs and ranking](../examples/README.md#2-retrieve-a-robot-reference) |
| Memory sizing | H200 process measurements and 24/32/80-GB configuration guidance | [Peaks and worker counts](evaluation-scaling.md) |
| Eight-GPU evaluation | Ten per-model launchers and CPU sharding tests released | Eight physical GPUs running end to end |
| Docker | Image built, CPU import/package checks passed, archive published | [Download and use the image](https://huggingface.co/mxk1998/RoboDemo-Docker); GPU simulator validation |
| CI | GitHub Actions runs offline code tests and website tests | [Code](https://github.com/robodemo-2027/RoboDemo/actions) and [website](https://github.com/robodemo-2027/robodemo-2027.github.io/actions/workflows/site-checks.yml) |

The complete CPU suite passed **100 tests for each policy variant**. Hosted
code CI runs a 30-case offline subset plus inspection of the included sample;
the static site suite has 17 cases. These counts describe different suites,
not independent benchmark experiments.

The [memory guide](evaluation-scaling.md) uses measured total process peaks,
including rendering. Hardware-specific performance and eight-device execution
are separate from this capacity guidance.

## Common Setup Problems

| Symptom | Check |
| --- | --- |
| `mani_skill` cannot be imported | Use the recorded simulator revision and put its root on `PYTHONPATH`. |
| Renderer reports a missing Vulkan or OptiX runtime | Expose the host's matching NVIDIA graphics/OptiX libraries. Follow the [driver checks](../environments/README.md#host-driver-and-diagnostics). |
| CUDA out of memory | Use `--workers-per-gpu 1`, check free VRAM, and [profile a complete run](../environments/README.md#measure-your-inference-process). |
| Asset or encoder path not found | Run `check_environment.py --assets` and correct the reported environment variable. |
| Output directory is already populated | Use a new directory, or follow the launcher's [resume instructions](evaluation-scaling.md#restart-and-merge). |

Do not change reference sampling, image resolution or action horizon when
reproducing the reported results.
