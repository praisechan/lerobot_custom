#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-/home/juchanlee/lerobot_custom/3rdparty/cosmos-reason2/.venv/bin/python}"

"${PYTHON_BIN}" - <<'PY'
import torch
from libsmctrl_extension import load_libsmctrl_extension

print("python extension prebuild")
print("torch", torch.__version__, "cuda", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")

torch.cuda.init()
ext = load_libsmctrl_extension(verbose=True)
print(ext.get_tpc_info(0))
PY
