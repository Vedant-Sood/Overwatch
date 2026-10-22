#!/usr/bin/env bash
# BigBoss setup for Ubuntu / WSL2. Run from the repository root:  bash scripts/setup.sh
set -euo pipefail
PY=${PYTHON:-python3}
$PY -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,xgb,tune]"
# PyTorch is optional (MLP model). For the RTX 4060 (CUDA 12.x wheels):
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
if command -v npm >/dev/null; then
  (cd frontend && npm ci && npm run build)
else
  echo "npm not found - install Node.js 20+ to build the dashboard (the API still works)."
fi
echo
echo "Done. Next:"
echo "  source .venv/bin/activate"
echo "  bigboss ingest --groups us_etfs us_equities   # real data (needs internet)"
echo "  bigboss train --horizon 20"
echo "  bigboss backtest --horizon 20"
echo "  bigboss serve                                  # http://127.0.0.1:8000"
