#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-/home/juchanlee/lerobot_custom/3rdparty/cosmos-reason2/.venv/bin/python}"

"${PYTHON_BIN}" overlap_experiment.py \
  --vla-tpc-start 0 --vla-tpc-count 1 \
  --llm-tpc-start 1 --llm-tpc-count 1 \
  --case vla-encoder-only \
  --iterations 1 \
  --warmup 0 \
  "$@"
