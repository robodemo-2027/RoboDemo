#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec "${PYTHON:-python}" evaluation/launch.py --model c45 --gpus 8 "$@"
