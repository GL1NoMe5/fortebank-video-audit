$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'runtime_env.ps1')
Push-Location $ProjectRoot
try {
    if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
        uv venv .venv --python 3.10
        if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
    }
    uv pip install --python .venv/Scripts/python.exe torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed' }
    $RequirementsFile = 'requirements-runtime.txt'
    if (Test-Path -LiteralPath 'requirements-runtime.lock.txt') {
        $RequirementsFile = 'requirements-runtime.lock.txt'
    }
    uv pip install --python .venv/Scripts/python.exe --requirement $RequirementsFile
    if ($LASTEXITCODE -ne 0) { throw 'Runtime dependency installation failed' }
    & .venv/Scripts/python.exe -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
    if ($LASTEXITCODE -ne 0) { throw 'CUDA validation failed' }
} finally {
    Pop-Location
}
