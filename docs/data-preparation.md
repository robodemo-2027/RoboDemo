# Prepare Evaluation Assets

Run the downloader on the Linux evaluation machine. Downloads are resumable;
repository versions are recorded in `environments/assets.json`.

```bash
pip install huggingface-hub==0.36.2
export ROBODEMO_HOST_ASSETS=/data/robodemo-assets
```

| Profile | Downloads |
| --- | --- |
| `policy-example` | Full50 weights, processor files and temporal encoder; sample media are already in Git |
| `retrieval-example` | Two retrieval encoders and the single video containing the four example episodes |
| `smoke` | One policy, encoders, one reference video, simulator and simulation assets |
| `evaluation` | Same as smoke, with all reference videos in that model's complete trial plan |

Policy weights are about 16.5 GB; the original temporal checkpoint is about
11.8 GB. Evaluation needs Qwen configuration/tokenizer/processor files,
not an additional copy of its weights. The evaluated V-JEPA2 encoder source
is included in `vendor/vjepa2`.

## Minimal Examples

```bash
python scripts/prepare_assets.py --root "$ROBODEMO_HOST_ASSETS" \
  --profile policy-example --model full50
source "$ROBODEMO_HOST_ASSETS/robodemo-env.sh"
CUDA_VISIBLE_DEVICES=0 python examples/predict_actions.py \
  --checkpoint "$ROBODEMO_HOST_ASSETS/checkpoints/RoboDemo-full50/pytorch_model.pt" \
  --output outputs/policy-example
```

For retrieval, first request access to
[DINOv3 ViT-H+](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m)
and run `hf auth login` on your own machine. This upstream approval is not
needed to download the five RoboDemo checkpoints.

```bash
python scripts/prepare_assets.py --root "$ROBODEMO_HOST_ASSETS" --profile retrieval-example
source "$ROBODEMO_HOST_ASSETS/robodemo-env.sh"
CUDA_VISIBLE_DEVICES="" python examples/retrieve_reference.py \
  --data-root "$ROBODEMO_DATA_ROOT" --device cpu \
  --output outputs/retrieval-example.json
```

## Simulator Evaluation

Install `git`, `curl`, GNU `tar` and `zstd`. Choose any of the five models:

```bash
export MODEL=c15
python scripts/prepare_assets.py --root "$ROBODEMO_HOST_ASSETS" \
  --profile smoke --model "$MODEL" --extract-assets
source "$ROBODEMO_HOST_ASSETS/robodemo-env.sh"
CUDA_VISIBLE_DEVICES=0 bash "scripts/evaluate/$MODEL-single.sh" \
  --checkpoint "$ROBODEMO_HOST_ASSETS/checkpoints/RoboDemo-$MODEL/pytorch_model.pt" \
  --data-root "$ROBODEMO_DATA_ROOT" --task-level L0_TwoRobotFoldTowel-v1 \
  --workers-per-gpu 1 --max-episodes 1 --output "outputs/$MODEL-smoke"
```

The smoke profile prepares the towel task specifically. Keep `--task-level`
in its evaluation command. The four shared simulator asset archives still
need space for both download and extraction; use a dedicated asset directory.
Completed extraction is recorded beside each archive; interrupted extraction
is rerun.

For the complete plan, prepare with `--profile evaluation`, then omit
`--task-level` and `--max-episodes` from evaluation. Use `--dry-run` to inspect
any download plan without writes or downloads. Completed downloads are reused.

Original media and simulator assets come from
[Imitator Game](https://imitator-game.github.io/).
Large encoder features are reconstructed locally, not redistributed.
