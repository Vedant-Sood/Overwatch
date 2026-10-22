# BigBoss setup for Windows 11 PowerShell. Run from the repository root:  .\scripts\setup.ps1
$ErrorActionPreference = "Stop"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,xgb,tune]"
# Optional GPU PyTorch for the MLP model (RTX 4060):
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
if (Get-Command npm -ErrorAction SilentlyContinue) {
  Push-Location frontend; npm ci; npm run build; Pop-Location
} else { Write-Host "npm not found - install Node.js 20+ to build the dashboard." }
Write-Host "Done. Activate with .\.venv\Scripts\Activate.ps1 then: bigboss ingest; bigboss train; bigboss serve"
