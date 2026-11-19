# Evaluation Environment

The recorded environment uses Python 3.10.20, PyTorch 2.6.0, torchvision
0.21.0, CUDA 12.4, Triton 3.2.0 and FlashAttention 2.7.4.post1.
`requirements-evaluation.txt` lists the policy and rendering dependencies.
`evaluation-packages.txt` preserves the full package-version inventory.

## Simulator

Obtain Imitator Game from its [project site](https://imitator-game.github.io/).
The experiments used simulator revision
`d6d16ec511bc389e0a207692730c137bc022ef14`, with SAPIEN 3.0.3.
Use that source tree on `PYTHONPATH`; the benchmark's current development
installation may target a different Python environment.

Install the benchmark assets following its instructions and set
`MS_ASSET_DIR`. Rendering requires an NVIDIA driver with Vulkan and the
OptiX denoiser runtime. Make the runtime available through `LD_LIBRARY_PATH`
if your driver installation does not already expose it.

Evaluation uses PhysX CPU simulation, `pd_joint_pos` control, the `zed2i`
224-by-224 RGB camera and the `rt-fast` shader with OptiX denoising.
Control and physics rates are 20 Hz and 100 Hz in simulation time.
These are simulator rates, not a claim of 20-Hz wall-clock policy inference.

## Local Verification

The tests use small synthetic tensors and saved episode records.
They do not load pretrained weights, run the simulator, make network requests
or control a physical robot. A new GPU rollout is separate from those tests.

## Docker Image

The build files, package inventory and diagnostics are maintained here in
GitHub. The shared Linux **amd64** image is available from
[RoboDemo-Docker on Hugging Face](https://huggingface.co/mxk1998/RoboDemo-Docker).
It serves all five policy checkpoints. The **5.9-GB download** passed a clean
build, CPU package/simulator-import checks and **100 CPU tests for each
policy variant inside the image** in
[GitHub Actions](https://github.com/robodemo-2027/RoboDemo/actions/runs/44473741985).

The same exported image also passed a **500-step Full50 simulator episode
on H200**, including Vulkan/OptiX rendering, with **21.38 GiB** peak process
memory. This GPU run used Apptainer 1.5.5-1 and the image's own Python
environment; Docker Engine/NVIDIA Container Toolkit execution was not tested
on that host. See the [container validation record](../docs/validation/20261119-container.json).

The image fixes Python 3.10.20, the CUDA 12.4 PyTorch build, simulator source,
and package versions. Dataset assets and model files are mounted separately.
It deliberately does not install the simulator repository's current training
extras, which target a different Python/PyTorch stack.

```bash
# Stream into Docker without storing a second archive copy.
curl -fL https://huggingface.co/mxk1998/RoboDemo-Docker/resolve/main/robodemo-evaluation-cu124-amd64.tar.gz \
  | gzip -d | docker load

# Run from the RoboDemo repository; /data/robodemo-assets is an example path.
export ROBODEMO_HOST_ASSETS=/data/robodemo-assets
export ROBODEMO_HOST_OUTPUTS="$PWD/outputs"
export ROBODEMO_GPU=0
mkdir -p "$ROBODEMO_HOST_OUTPUTS"
docker compose -f environments/docker-compose.yml run --rm evaluation \
  python environments/check_environment.py --assets
```

To build from source instead, run
`docker build -t robodemo-evaluation:cu124 environments`.
The image does not contain model weights or benchmark assets.

The asset mount should contain `IG-10K-Dataset/`, `maniskill/`,
`Qwen3-VL-4B-Instruct/`, `vjepa2/`, and `vjepa2-ac-vitg.pt`.
Place policy weights in the repository's ignored `checkpoints/` directory.
For the raw-media retrieval example, optionally add `Qwen3-VL-Embedding-2B/`
and `dinov3-vith16plus-pretrain-lvd1689m/` under the same asset directory.
These extra encoders are not needed for standard C15-wo evaluation with
the included reference map.
Run one complete episode:

```bash
docker compose -f environments/docker-compose.yml run --rm evaluation \
  bash scripts/evaluate/c15-single.sh \
  --checkpoint checkpoints/RoboDemo-c15/pytorch_model.pt \
  --data-root /assets/IG-10K-Dataset --max-episodes 1 \
  --output /outputs/c15-smoke
```

For eight GPUs, use `docker run --gpus '"device=0,1,2,3,4,5,6,7"'`
with the same mounts/environment, and set container `CUDA_VISIBLE_DEVICES`
to its eight visible device indices. The supplied Compose example allocates
one GPU; it is not an eight-GPU reservation.

## Host Driver And Diagnostics

Two native driver configurations have completed simulator runs:

| NVIDIA driver | Recorded validation |
| --- | --- |
| **555.42.06** | Original recorded evaluation environment |
| **570.195.03** | [Five-model 500-step memory runs on H200](../docs/validation/20261119-five-model-memory.json) and [minimal examples](../docs/validation/20261119.json) |

CUDA 12.4 wheels require a compatible Linux driver (CUDA 12.x minor-version
compatibility starts at 525.60.13), but that CUDA floor alone does not certify
SAPIEN/Vulkan/OptiX compatibility. Use a supported host driver and run the
checks below; downgrading to 555.42.06 is not required. Do not install a
kernel driver inside the container.

`nvidia-smi`'s "CUDA Version" is the driver's supported CUDA level, not the
PyTorch build's toolkit version. Use `torch.version.cuda` to check the latter.
The host needs NVIDIA Container Toolkit, Vulkan driver discovery, and
`libnvoptix.so.1` from its matching NVIDIA driver. The container requests
`compute,utility,graphics` driver capabilities. If OptiX is absent, install
its matching host driver component rather than substituting a different
shader or turning off denoising.

### Standalone OptiX Runtime

On hosts that provide OptiX separately, keep **`libnvoptix.so.1` and
`nvoptix.bin` together** in a dedicated directory from the matching NVIDIA
driver. The resource file must be accessible beside the loaded library.
Use the optional Compose override to mount that directory and put it first
on the library search path:

```bash
export ROBODEMO_HOST_OPTIX=/data/nvidia-optix
docker compose -f environments/docker-compose.yml \
  -f environments/docker-compose.optix.yml run --rm evaluation \
  python environments/check_environment.py --assets

docker compose -f environments/docker-compose.yml \
  -f environments/docker-compose.optix.yml run --rm evaluation \
  bash scripts/evaluate/c15-single.sh \
  --checkpoint checkpoints/RoboDemo-c15/pytorch_model.pt \
  --data-root /assets/IG-10K-Dataset --max-episodes 1 \
  --output /outputs/c15-optix-smoke
```

Use the same override for other models. The directory should contain the
OptiX runtime files, not a host operating system's entire library directory.
This is a host-driver mount; the model, shader and denoiser settings are unchanged.

### Check The Environment

```bash
nvidia-smi
python environments/check_environment.py --assets \
  --output outputs/environment.json
# Package and simulator-import checks without initializing CUDA:
python environments/check_environment.py --cpu-only
```

The diagnostics do not run an episode. Follow them with a one-episode
simulator smoke test before starting a larger evaluation. See
[memory and parallelism](../docs/evaluation-scaling.md) for 32-GB/80-GB
starting configurations and the currently measured scope.

## Measure Your Inference Process

The profiler records both the PyTorch allocator peak and sampled total
process memory reported by `nvidia-smi`. It can also apply a PyTorch-only
allocation budget, without filling the card or reserving unused memory:

```bash
CUDA_VISIBLE_DEVICES=0 python environments/profile_memory.py \
  --budget-gib 32 --output outputs/memory.json -- \
  examples/predict_actions.py \
  --checkpoint checkpoints/RoboDemo-full50/pytorch_model.pt \
  --output outputs/profiled-example
```

A budget is not a physical GPU partition: the driver and renderer allocate
memory outside the PyTorch limit. Use the process-memory reading and a
complete simulator episode when deciding how many workers fit.

The active manual [image-build workflow](../.github/workflows/build-image.yml) builds a
clean Linux amd64 image, checks its imports/package versions and both complete
CPU test suites, and publishes
a `docker load` archive to `mxk1998/RoboDemo-Docker` only after those checks
pass. The publication step uses the repository's `HF_TOKEN` secret; users
do not need a token to download the public image. `ci/build-image.yml` is
the reusable workflow copy.
