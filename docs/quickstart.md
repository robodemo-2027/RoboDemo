# Evaluate A Released Checkpoint

Run the evaluator on Linux with Python 3.10 and an NVIDIA GPU.
All five checkpoints use the same entry point.
For a smaller first step, see the [reproduction navigator](reproduce.md):
inspect the included inputs or try a policy/retrieval example without a simulator.

## 1. Install The Code

For a ready-made Linux amd64 environment, use the
[Docker image and mounted-data commands](../environments/README.md#docker-image).
For a native Python environment:

```bash
git clone https://github.com/robodemo-2027/RoboDemo.git
cd RoboDemo
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r environments/requirements-evaluation.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

The [environment guide](../environments/README.md) includes the simulator
revision, Vulkan and OptiX setup, and the recorded package versions.

## 2. Download One Model

Choose one row and set `MODEL` to its evaluator name:

| Evaluator name | Model repository | Updates | Reference | Evaluation episodes |
| --- | --- | ---: | --- | ---: |
| `full50` | [RoboDemo-full50](https://huggingface.co/mxk1998/RoboDemo-full50) | 240,000 | Human | 2,000 |
| `c15` | [RoboDemo-c15](https://huggingface.co/mxk1998/RoboDemo-c15) | 90,000 | Human | 400 |
| `c30` | [RoboDemo-c30](https://huggingface.co/mxk1998/RoboDemo-c30) | 160,000 | Human | 400 |
| `c45` | [RoboDemo-c45](https://huggingface.co/mxk1998/RoboDemo-c45) | 260,000 | Human | 400 |
| `c15-wo` | [RoboDemo-c15-wo](https://huggingface.co/mxk1998/RoboDemo-c15-wo) | 100,000 | Retrieved robot | 400 |

```bash
export MODEL=c15
hf download "mxk1998/RoboDemo-${MODEL}" \
  pytorch_model.pt policy.yaml normalization.json \
  --local-dir "checkpoints/RoboDemo-${MODEL}"
```

Each checkpoint is approximately 16.5 GB. Download only the models you plan
to evaluate. The checkpoint is a complete PyTorch policy state dictionary,
loaded by the RoboDemo evaluator. The evaluator selects its matching
configuration and normalization from this repository.

## 3. Prepare The Benchmark And Encoders

Follow the [Imitator Game project](https://imitator-game.github.io/) to obtain
the IG-10K dataset, simulator source and simulation assets. Keep the
benchmark's dataset directory layout, including human and robot videos.

Prepare the local configuration, tokenizer and processor files from
[Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), and the
[V-JEPA2 source and `vjepa2-ac-vitg.pt` encoder](https://github.com/facebookresearch/vjepa2).
Set the following paths to your installation:

```bash
export ROBODEMO_DATA_ROOT=/data/IG-10K-Dataset
export IMITATOR_GAME_ROOT=/path/to/ImitationGame
export MS_ASSET_DIR=/data/maniskill
export ROBODEMO_QWEN=/models/Qwen3-VL-4B-Instruct
export ROBODEMO_VJEPA2_SOURCE=/path/to/vjepa2
export ROBODEMO_VJEPA2_WEIGHTS=/models/vjepa2-ac-vitg.pt
export PYTHONPATH="$IMITATOR_GAME_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

Reference identities, normalization and trial seeds are already included in
`configs/`. C15-wo uses the published robot-reference selections; no separate
retrieval run is needed to evaluate its released trial plan. See
[reference retrieval](retrieval.md) for the scorer and candidate-bank interface.

## 4. Test One Episode

From the repository root, check the environment and asset paths first:

```bash
CUDA_VISIBLE_DEVICES=0 python environments/check_environment.py --assets \
  --output "outputs/${MODEL}-environment.json"
```

Then run one episode:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate.py \
  --model "$MODEL" \
  --checkpoint "checkpoints/RoboDemo-${MODEL}/pytorch_model.pt" \
  --data-root "$ROBODEMO_DATA_ROOT" \
  --task-level L0_TwoRobotFoldTowel-v1 \
  --max-episodes 1 \
  --output "outputs/${MODEL}-smoke"
```

Replace GPU `0` with a GPU allocated to your job. Each test runs one complete
500-control-step episode. Use a new output directory for each run.
[Recorded smoke tests](smoke-evaluation.md) cover all five checkpoints.

## 5. Evaluate The Complete Trial Plan

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate.py \
  --model "$MODEL" \
  --checkpoint "checkpoints/RoboDemo-${MODEL}/pytorch_model.pt" \
  --data-root "$ROBODEMO_DATA_ROOT" \
  --output "outputs/${MODEL}-evaluation"
```

Full50 covers all 50 tasks. C15, C30, C45 and C15-wo each cover five Seen and
five Unseen tasks. For those four models, append `--split seen` or
`--split unseen` to evaluate only that split. The default evaluates both.

## Outputs And Published Results

| Output | Contents |
| --- | --- |
| `model_load.json` | Checkpoint step and strict-loading result |
| `summary.json` | Completed episode count, successes and success rate |
| `<trial_id>/trial.json` | Task, seeds and selected reference |
| `<trial_id>/result.json` | Per-episode metrics and timings |
| `<trial_id>/trajectory.npz` | Recorded actions, states and execution-memory ages |
| `<trial_id>/rollout.mp4` | Rendered episode |

The [complete results directory](../results/README.md) provides all published
numbers, per-model tables, CSV/JSON downloads and 3,600 episode logs.
You can recompute their summaries without downloading weights:

```bash
python evaluation/summarize.py results/*/episodes.jsonl
python evaluation/export_results.py --output outputs/result-tables
```

Model architecture and evaluation are released. Training code and launch
recipes remain on the [release roadmap](../README.md#release-status).

## Docker, Parallel Evaluation And Examples

The [shared Docker recipe](../environments/README.md#docker-recipe) uses the
same environment for all five checkpoints. Check the host driver and assets
with `python environments/check_environment.py --assets`.

Use the [per-model one/eight-GPU launchers](evaluation-scaling.md) to choose
episode count, workers per GPU, and restart an interrupted run. For learning
the interfaces without a simulator, start with the
[minimal policy and retrieval examples](../examples/README.md).
