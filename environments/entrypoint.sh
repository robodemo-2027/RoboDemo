#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${IMITATOR_GAME_ROOT:-/opt/ImitatorGame}${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "${VK_ICD_FILENAMES:-}" ]]; then
  for path in /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json; do
    if [[ -f "$path" ]]; then export VK_ICD_FILENAMES="$path"; break; fi
  done
fi
exec "$@"
