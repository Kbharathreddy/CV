$ErrorActionPreference = "Stop"

py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host "Environment created. Install PyTorch/CUDA and COLMAP according to README.md if needed."
