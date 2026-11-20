# Run Without Docker Engine

The released image can run through Apptainer without Docker Engine or NVIDIA
Container Toolkit. This is the GPU-tested container route for the current
release. The host still needs NVIDIA graphics/OptiX drivers and permission
to create user namespaces. Use Apptainer 1.5.5 or a compatible installation.
Do not start a nested Docker daemon inside a managed notebook container.

## Import The Published Image

Run on the Linux server with space for an unpacked image and conversion files:

```bash
mkdir -p containers
curl -fL https://huggingface.co/mxk1998/RoboDemo-Docker/resolve/main/robodemo-evaluation-cu124-amd64.tar.gz \
  | gzip -d > containers/robodemo-evaluation.tar
apptainer build --sandbox containers/robodemo-evaluation \
  "docker-archive://$PWD/containers/robodemo-evaluation.tar"
```

This imports the published image, without rebuilding dependencies.
Keep the archive until import succeeds. The sandbox can be reused.

## Evaluate

First [prepare the model and data assets](data-preparation.md).
Code and model/data inputs are mounted read-only; outputs and compiler
caches go into the selected output directory.

```bash
export ROBODEMO_HOST_ASSETS=/data/robodemo-assets
export ROBODEMO_HOST_OUTPUTS="$PWD/outputs/apptainer"
export CUDA_VISIBLE_DEVICES=0

bash environments/run-apptainer.sh containers/robodemo-evaluation \
  python environments/check_environment.py --assets

bash environments/run-apptainer.sh containers/robodemo-evaluation \
  bash scripts/evaluate/c15-single.sh \
  --checkpoint /checkpoints/RoboDemo-c15/pytorch_model.pt \
  --data-root /assets/IG-10K-Dataset --task-level L0_TwoRobotFoldTowel-v1 \
  --workers-per-gpu 1 --max-episodes 1 --output /outputs/c15-smoke
```

For separately supplied OptiX libraries, set `ROBODEMO_HOST_OPTIX=/path/to/optix`.
Keep `libnvoptix.so.1` and its matching `nvoptix.bin` together there.

## Minimal Examples

After preparing the corresponding example profile:

```bash
bash environments/run-apptainer.sh containers/robodemo-evaluation \
  python examples/predict_actions.py \
  --checkpoint /checkpoints/RoboDemo-full50/pytorch_model.pt \
  --output /outputs/policy-example

CUDA_VISIBLE_DEVICES="" bash environments/run-apptainer.sh containers/robodemo-evaluation \
  python examples/retrieve_reference.py --data-root /assets/IG-10K-Dataset \
  --device cpu --cache /outputs/retrieval-features --output /outputs/retrieval.json
```

Use allocated device IDs. An empty `CUDA_VISIBLE_DEVICES` selects CPU-only
execution. `APPTAINER` optionally names a platform-specific executable or
wrapper; `ROBODEMO_CHECKPOINTS` selects a separate directory mounted at
`/checkpoints`. Eight-device hardware validation remains a future check.
