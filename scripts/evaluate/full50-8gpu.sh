#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec "${PYTHON:-python}" evaluation/launch.py --model full50 --gpus 8 "$@"
