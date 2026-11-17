# RoboDemo

## Release Status

- [x] Model architecture and dual-memory inference.
- [x] Imitator Game evaluation code and environment specifications.
- [x] Evaluation references, random seeds and per-model normalization.
- [x] Episode logs and complete task-level results for all five models.
- [ ] Publish the five model checkpoints.
- [ ] Training code, data preparation and launch recipes.

RoboDemo separates demonstration intent from execution progress with
dual-cascaded test-time memory. A reference video initializes Retrieval-TTT;
completed interaction transitions update Execution-TTT. The learned policy
parameters remain fixed during evaluation.

## Models And Results

| Model | Checkpoint | Reference | Seen SR (%) | Unseen SR (%) |
| --- | ---: | --- | ---: | ---: |
| RoboDemo-c15 | 90,000 | Human | 89.5 | 0.0 |
| RoboDemo-c30 | 160,000 | Human | 94.0 | 0.5 |
| RoboDemo-c45 | 260,000 | Human | 88.5 | 11.5 |
| RoboDemo-c15-wo | 100,000 | Retrieved robot | 90.0 | 0.0 |

RoboDemo-full50 at **240,000** updates achieves **84.6% SR** across all 50
tasks (1,692 / 2,000 episodes). The three-scale Seen mean is **90.7%**.
Each nested-corpus model is evaluated on 200 Seen and 200 Unseen episodes.
For C15-wo, Unseen denotes tasks held out from policy training; task-matched
robot references are supplied from the benchmark training bank.

[Complete results](results/README.md) include all 360 task-level cells.
The selected checkpoint for each model is the one used for the reported
results. Checkpoints are distributed separately from this code repository.

## Installation

The evaluation environment uses Linux, Python 3.10, PyTorch 2.6.0 / CUDA
12.4, Transformers 4.57.6, SAPIEN 3.0.3 and Gymnasium 0.29.1.
See [environment setup](environments/README.md) for installation and rendering.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r environments/requirements-evaluation.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## Data Preparation

Follow the [Imitator Game project](https://imitator-game.github.io/) to obtain
the IG-10K dataset, simulator and assets. Use the paired human-reference and
simulation subsets. Dataset media are distributed by the benchmark.

Configure the dataset, simulator and pretrained encoder locations:

```bash
export ROBODEMO_DATA_ROOT=/data/IG-10K-Dataset
export IMITATOR_GAME_ROOT=/path/to/ImitationGame
export MS_ASSET_DIR=/data/maniskill
export ROBODEMO_QWEN=/models/Qwen3-VL-4B-Instruct
export ROBODEMO_XR1=/models/Xiaomi-Robotics-1-5B
export ROBODEMO_VJEPA2_SOURCE=/path/to/vjepa2
export ROBODEMO_VJEPA2_WEIGHTS=/models/vjepa2-ac-vitg.pt
export PYTHONPATH="$IMITATOR_GAME_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

The pretrained components are
[Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct),
[Xiaomi-Robotics-1](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-5B)
and [V-JEPA2](https://github.com/facebookresearch/vjepa2).
Their original distribution terms apply.

`configs/*_trials.json` contains the reference identities and seeds used by the
reported experiments. Paths inside those files are relative to the IG-10K
dataset root. C15-wo also includes its 400 selected robot-reference identities
and candidate scores.

## Evaluation

Use a downloaded policy checkpoint and a GPU allocated to your job:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate.py \
  --model c15 \
  --checkpoint /models/RoboDemo-c15/pytorch_model.pt \
  --data-root "$ROBODEMO_DATA_ROOT" \
  --output outputs/c15
```

Choose `full50`, `c15`, `c30`, `c45` or `c15-wo`. Add `--split seen` or
`--split unseen` for a nested-corpus split, or `--task-level
L0_TwoRobotFoldTowel-v1` for a task-level subset. Use a new output directory
for each evaluation.

The evaluator queries the policy every control step, retains a new action
chunk every four steps, and combines up to four overlapping chunks.
Each applied, bounded action is fed back into execution memory.
An episode is successful if the benchmark success condition is reached
within the 500-control-step horizon.

For the reported C15-wo episodes, the evaluator reuses the published
initial-observation retrieval selections. `evaluation/retrieval.py` also
provides the original query-conditioned scorer for a prepared candidate
feature bank; see [reference retrieval](docs/retrieval.md).

## Evaluation Logs

`results/<model>/episodes.jsonl` contains the completed episode records with
task, level, seed, reference, success and timing fields. Recompute the
published summary without loading a model:

```bash
python evaluation/summarize.py results/*/episodes.jsonl
python -m pytest tests
ROBODEMO_TEST_VARIANT=robot python -m pytest tests
```

The result package covers 3,600 episodes.

## Code Layout

| Directory | Contents |
| --- | --- |
| `models/human/demottt` | Human-reference policy, reference encoder and dual-memory modules |
| `models/robot/demottt` | Robot-reference policy and video adapter |
| `models/robot/refreward` | Retrieval model and frozen feature extractors |
| `evaluation` | Rollout, action aggregation, reference selection and summarization |
| `configs` | Inference configurations, normalization and trial manifests |
| `environments` | Evaluation dependency versions and simulator setup |
| `results` | Episode logs, summary and per-task tables |

The two policy source trees preserve their respective evaluation
implementations. The original `demottt` module name is retained for checkpoint
compatibility. Training launchers, optimizers and checkpoint-resume tooling
will be released in the next stage.

## Acknowledgments

This implementation builds on Imitator Game, StarVLA, Xiaomi-Robotics-1,
Qwen3-VL, V-JEPA2, LeRobot and OpenVLA. Upstream source notices are preserved
in the code and [licenses](licenses/). See [NOTICE](NOTICE.md).

RoboDemo code is released under the [MIT License](LICENSE). Third-party
components retain their respective licenses.

## Anonymous Citation

```bibtex
@misc{anonymous2027robodemo,
  title  = {RoboDemo},
  author = {Anonymous Authors},
  year   = {2027},
  note   = {Manuscript under review}
}
```
