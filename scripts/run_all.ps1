# ISIP — Windows quick start
$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\.."

Write-Host "[1/3] Installing dependencies (if needed)..."
python -m pip install -e ".[dev]"

Write-Host "[2/3] Starting edge orchestrator on :8080 ..."
Start-Process -NoNewWindow python -ArgumentList "scripts/run_edge.py"
Start-Sleep -Seconds 3

Write-Host "[3/3] Starting operator dashboard on :8501 ..."
python -m streamlit run src/isip/dashboard/app.py --server.port 8501
