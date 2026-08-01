#!/usr/bin/env bash
# =============================================================================
# Environment for the record-#40 NanoGPT runs: a venv with a torch that Flash
# Attention 3 has a prebuilt for.
#
#   bash setup_env.sh
#   source .venv/bin/activate
#   bash run_all.sh
#
# Why a venv and not the system python: apt-installed packages have no pip RECORD,
# so pip cannot uninstall them ("Cannot uninstall cryptography 41.0.7").
#
# Why its own torch and not the container's: `kernels` downloads a PREBUILT FA3
# binary, and the hub has variants for cu126 / cu128 / cu130 only. NGC containers
# ship CUDA 13.1 builds (torch 2.10.0a0+...nv26.01, needs a cu131 variant that does
# not exist), so the container torch cannot resolve FA3. PyPI's torch==2.10.0 is a
# cu128 build -> variant torch210-cxx11-cu128-x86_64-linux, which does exist.
# cu128 runs fine on the newer driver.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$HERE/.venv}"
PYTHON="${PYTHON:-python3}"

echo "== venv -> $VENV"
if [ -x "$VENV/bin/python" ] &&
   grep -qi 'include-system-site-packages *= *true' "$VENV/pyvenv.cfg" 2>/dev/null; then
  echo "  this venv inherits system site-packages, so it still sees the container's"
  echo "  CUDA 13.1 torch, which has no FA3 prebuilt. Recreate it clean:"
  echo "      rm -rf $VENV && bash setup_env.sh"
  exit 1
fi
if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON" -m venv "$VENV" || {
    echo "'python -m venv' failed; on Debian/Ubuntu: sudo apt-get install -y python3-venv"
    exit 1
  }
fi
VPY="$VENV/bin/python"

echo "== packages"
"$VPY" -m pip install --quiet --upgrade pip
"$VPY" -m pip install -r "$HERE/requirements.txt"

echo "== verify"
"$VPY" - <<'PY'
import sys, torch
print(f"  python  {sys.version.split()[0]}")
print(f"  torch   {torch.__version__} (CUDA {torch.version.cuda}), "
      f"{torch.cuda.device_count()} GPU(s)")
from kernels import get_kernel
get_kernel("varunneal/flash-attention-3")     # real fetch; cached afterwards
print("  FA3     resolved and cached")
PY

echo "== ready"
echo "  source $VENV/bin/activate"
echo "  python data/cached_fineweb10B.py 9    # if you have no shards yet"
echo "  NANOGPT_ITERS=200 bash run_all.sh     # smoke pass, then: bash run_all.sh"
