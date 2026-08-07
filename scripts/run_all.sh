#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"

echo "[1/3] Installing dependencies (if needed)..."
"$PYTHON" -m pip install -e ".[dev]" || "$PYTHON" -m pip install -r requirements.txt

echo "[2/3] Starting edge orchestrator on :8080 ..."
"$PYTHON" scripts/run_edge.py &
EDGE_PID=$!
trap "kill $EDGE_PID" EXIT

sleep 3
echo "[3/3] Starting operator dashboard on :8501 ..."
"$PYTHON" -m streamlit run src/isip/dashboard/app.py --server.port 8501
