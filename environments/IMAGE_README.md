---
license: other
license_name: upstream-component-licenses
license_link: https://github.com/robodemo-2027/RoboDemo/blob/main/NOTICE.md
tags:
- robotics
- docker
- evaluation
---
# RoboDemo Evaluation Image

A Linux **amd64** Docker image shared by all five RoboDemo checkpoints.
It contains the pinned Python/CUDA/PyTorch environment and Imitator Game
simulator source. Dataset assets and model weights are mounted separately.
The runtime image omits build-stage downloads and CUDA compiler tools;
the C compiler needed by Triton remains included.

## 1. Load The Image

On a Linux x86-64 GPU host with Docker and NVIDIA Container Toolkit:

```bash
curl -fL https://huggingface.co/mxk1998/RoboDemo-Docker/resolve/main/robodemo-evaluation-cu124-amd64.tar.gz \
  | gzip -d | docker load
docker run --rm --gpus all robodemo-evaluation:cu124
```

The streaming command avoids keeping a second copy of the image archive.
Docker still needs disk space for the unpacked image.

## 2. Mount Data And Models

Clone [RoboDemo](https://github.com/robodemo-2027/RoboDemo) and follow its
[data and encoder setup](https://github.com/robodemo-2027/RoboDemo/blob/main/docs/quickstart.md#3-prepare-the-benchmark-and-encoders).
The shared asset directory contains `IG-10K-Dataset/`, `maniskill/`,
`Qwen3-VL-4B-Instruct/`, `vjepa2/`, and `vjepa2-ac-vitg.pt`.
Put a released checkpoint in `checkpoints/RoboDemo-c15/pytorch_model.pt`.

```bash
cd RoboDemo
export ROBODEMO_HOST_ASSETS=/data/robodemo-assets
export ROBODEMO_HOST_OUTPUTS="$PWD/outputs"
export ROBODEMO_GPU=0
mkdir -p "$ROBODEMO_HOST_OUTPUTS"

docker compose -f environments/docker-compose.yml run --rm evaluation \
  python environments/check_environment.py --assets
```

## 3. Evaluate A Checkpoint

```bash
docker compose -f environments/docker-compose.yml run --rm evaluation \
  bash scripts/evaluate/c15-single.sh \
  --checkpoint checkpoints/RoboDemo-c15/pytorch_model.pt \
  --data-root /assets/IG-10K-Dataset --workers-per-gpu 1 --max-episodes 1 \
  --output /outputs/c15-smoke
```

Replace `c15` with `full50`, `c30`, `c45` or `c15-wo` and its checkpoint.
Remove `--max-episodes` for the complete trial plan.
[GPU sizing](https://github.com/robodemo-2027/RoboDemo/blob/main/docs/evaluation-scaling.md)
recommends one worker on 32 GB or two on 80 GB, based on H200 process-memory
measurements. The Compose example allocates one GPU.

For driver setup and diagnostics, use the
[environment guide](https://github.com/robodemo-2027/RoboDemo/tree/main/environments).
The reproducible [Dockerfile](https://github.com/robodemo-2027/RoboDemo/blob/main/environments/Dockerfile)
is maintained in GitHub.

## Validation

The image passed a clean build, CPU package checks and **100 regression
tests for each policy variant**. It also completed a **500-step Full50
episode on H200**, with Vulkan/OptiX rendering and **21.38 GiB** peak process
memory. It completed 500 policy calls and 499 memory updates, with task success.

The GPU run loaded this exported image with **Apptainer 1.5.5-1**, NVIDIA
driver 570.195.03 and the image's own Python packages. Docker Engine/NVIDIA
Container Toolkit execution was not tested on that host. Reports:
[CPU environment](https://huggingface.co/mxk1998/RoboDemo-Docker/blob/main/container-check.json),
[CPU tests](https://huggingface.co/mxk1998/RoboDemo-Docker/blob/main/container-regression.json) and
[GPU validation](https://huggingface.co/mxk1998/RoboDemo-Docker/blob/main/container-gpu-validation.json).

The [two minimal examples](https://github.com/robodemo-2027/RoboDemo/tree/main/examples#run-the-examples-in-docker)
also passed inside the image: human-video action prediction on H200 and
raw-media reference retrieval on CPU. The GPU report includes both outputs.

The host supplies its NVIDIA driver, NVIDIA Container Toolkit, Vulkan
discovery and matching OptiX runtime. Do not install a kernel driver inside
the image.

If the host exposes OptiX separately, keep `libnvoptix.so.1` and `nvoptix.bin`
together and use the documented
[OptiX mount override](https://github.com/robodemo-2027/RoboDemo/tree/main/environments#standalone-optix-runtime).

## Licenses

Bundled CUDA, Python, simulator and dependency packages retain their upstream
licenses. RoboDemo code is distributed separately under the MIT license.
