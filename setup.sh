#!/usr/bin/env bash
# One-click initialization for a Linux GPU server: venv + dependencies + environment variables.
# Usage: bash setup.sh
# Note: on GPU machines use Python 3.10-3.12 (torch/transformers may not fully support 3.13+).
set -euo pipefail

cd "$(dirname "$0")"

PYBIN="${PYBIN:-python3}"
PYVER="$("$PYBIN" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
case "$PYVER" in
  3.10|3.11|3.12) ;;
  *)
    echo "==> Warning: $PYBIN version is $PYVER; 3.10-3.12 is recommended (torch CUDA wheels)."
    echo "    Retry with a specific interpreter: PYBIN=/path/to/python3.11 bash setup.sh"
    echo "    If you are sure this version works, set ALLOW_PY=1 to continue."
    [ "${ALLOW_PY:-0}" = "1" ] || exit 1
    ;;
esac

echo "==> Creating venv ($PYBIN $PYVER)"
"$PYBIN" -m venv .venv
source .venv/bin/activate

echo "==> Upgrading pip and installing dependencies"
pip install --upgrade pip
pip install -r requirements/vlm.txt

# Environment variables (load .env if present)
if [ -f .env ]; then
  set -a; source .env; set +a
  echo "==> Loaded .env"
fi

echo "==> Verifying torch + CUDA"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "==> Done. Next step: smoke-test the four experimental models (see docs/RUNBOOK.md P0)"
