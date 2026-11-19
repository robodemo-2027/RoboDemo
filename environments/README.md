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
It serves all five policy checkpoints and has passed a clean build and CPU
package/simulator-import checks in
[GitHub Actions](https://github.com/robodemo-2027/RoboDemo/actions/runs/44467850247).

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
clean Linux amd64 image, checks its imports/package versions, and publishes
a `docker load` archive to `mxk1998/RoboDemo-Docker` only after those checks
pass. The publication step uses the repository's `HF_TOKEN` secret; users
do not need a token to download the public image. `ci/build-image.yml` is
the reusable workflow copy.
