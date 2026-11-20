#!/usr/bin/env bash
set -euo pipefail
if (( $# < 2 )); then
  echo "Usage: CUDA_VISIBLE_DEVICES=0 ROBODEMO_HOST_ASSETS=/data/assets bash $0 IMAGE COMMAND [ARGS...]" >&2
  exit 2
fi
: "${CUDA_VISIBLE_DEVICES?Set allocated GPU IDs, or an empty value for CPU}"
: "${ROBODEMO_HOST_ASSETS:?Set the prepared asset directory}"
image=$(realpath "$1")
shift
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
assets=$(realpath "$ROBODEMO_HOST_ASSETS")
outputs=${ROBODEMO_HOST_OUTPUTS:-"$repo/outputs"}
mkdir -p "$outputs"
outputs=$(realpath "$outputs")
cache="$outputs/.container-cache"
mkdir -p "$cache"/{tmp,hf,torch,triton,inductor,cuda}
args=(exec --userns --cleanenv --contain --no-home --no-mount cwd,hostfs,bind-paths
      --pwd /workspace/RoboDemo)
if [[ -d "$image" ]]; then
  args+=(--writable)
  # Writable sandboxes need mount destinations before user-namespace startup.
  mkdir -p "$image"/{.singularity.d/libs,workspace/RoboDemo,assets,checkpoints,outputs,cache,opt/nvidia-optix}
  for file in /usr/bin/nvidia-smi /usr/bin/nvidia-debugdump /usr/bin/nvidia-persistenced \
    /usr/bin/nvidia-cuda-mps-control /usr/bin/nvidia-cuda-mps-server \
    /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    /usr/share/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_layers.json \
    /usr/share/vulkan/implicit_layer.d/nvidia_layers.json \
    /usr/share/egl/egl_external_platform.d/15_nvidia_gbm.json \
    /usr/share/egl/egl_external_platform.d/10_nvidia_wayland.json \
    /usr/share/nvidia/nvoptix.bin; do
    if [[ -e "$file" && ! -e "$image$file" ]]; then
      mkdir -p "$(dirname "$image$file")"
      touch "$image$file"
    fi
  done
fi
for name in IG-10K-Dataset maniskill Qwen3-VL-4B-Instruct \
  Qwen3-VL-Embedding-2B dinov3-vith16plus-pretrain-lvd1689m vjepa2-ac-vitg.pt; do
  [[ -e "$assets/$name" ]] || continue
  if [[ -d "$image" ]]; then
    if [[ -d "$assets/$name" ]]; then
      mkdir -p "$image/assets/$name"
    elif [[ ! -e "$image/assets/$name" ]]; then
      touch "$image/assets/$name"
    fi
  fi
  args+=(--bind "$(realpath "$assets/$name"):/assets/$name:ro")
done
checkpoints=${ROBODEMO_CHECKPOINTS:-"$assets/checkpoints"}
[[ ! -d "$checkpoints" ]] || args+=(--bind "$(realpath "$checkpoints"):/checkpoints:ro")
if [[ -n "${ROBODEMO_HOST_OPTIX:-}" ]]; then
  args+=(--bind "$(realpath "$ROBODEMO_HOST_OPTIX"):/opt/nvidia-optix:ro")
fi
[[ -z "$CUDA_VISIBLE_DEVICES" ]] || args+=(--nv)
args+=(--bind "$repo:/workspace/RoboDemo:ro" --bind "$outputs:/outputs" --bind "$cache:/cache")
export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export APPTAINERENV_ROBODEMO_DATA_ROOT=/assets/IG-10K-Dataset
export APPTAINERENV_IMITATOR_GAME_ROOT=/opt/ImitatorGame APPTAINERENV_MS_ASSET_DIR=/assets/maniskill
export APPTAINERENV_ROBODEMO_QWEN=/assets/Qwen3-VL-4B-Instruct
export APPTAINERENV_ROBODEMO_VJEPA2_SOURCE=/workspace/RoboDemo/vendor/vjepa2
export APPTAINERENV_ROBODEMO_VJEPA2_WEIGHTS=/assets/vjepa2-ac-vitg.pt
export APPTAINERENV_ROBODEMO_RETRIEVAL_QWEN=/assets/Qwen3-VL-Embedding-2B
export APPTAINERENV_ROBODEMO_RETRIEVAL_DINO=/assets/dinov3-vith16plus-pretrain-lvd1689m
export APPTAINERENV_PYTHONPATH=/opt/ImitatorGame
export APPTAINERENV_PATH=/opt/python/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export APPTAINERENV_LD_LIBRARY_PATH=/opt/nvidia-optix:/.singularity.d/libs:/usr/local/cuda/lib64
export APPTAINERENV_HF_HUB_OFFLINE=1 APPTAINERENV_TRANSFORMERS_OFFLINE=1
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1 APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_TOKENIZERS_PARALLELISM=false APPTAINERENV_OMP_NUM_THREADS=4
export APPTAINERENV_OPENBLAS_NUM_THREADS=1 APPTAINERENV_MKL_NUM_THREADS=1
export APPTAINERENV_TORCHINDUCTOR_COMPILE_THREADS=1 APPTAINERENV_TMPDIR=/cache/tmp
export APPTAINERENV_HF_HOME=/cache/hf APPTAINERENV_TORCH_HOME=/cache/torch
export APPTAINERENV_TRITON_CACHE_DIR=/cache/triton APPTAINERENV_TORCHINDUCTOR_CACHE_DIR=/cache/inductor
export APPTAINERENV_XDG_CACHE_HOME=/cache APPTAINERENV_CUDA_CACHE_PATH=/cache/cuda
exec "${APPTAINER:-apptainer}" "${args[@]}" "$image" "$@"
