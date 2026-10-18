$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$SecretFolder = Join-Path $ProjectRoot 'runtime\local_demo'
New-Item -ItemType Directory -Force $SecretFolder | Out-Null
$KeyPath = Join-Path $SecretFolder 'api_key.txt'
if (-not (Test-Path $KeyPath)) {
    ([guid]::NewGuid().ToString('N') + [guid]::NewGuid().ToString('N')) | Set-Content -Encoding ASCII $KeyPath
}
$env:FORTE_API_KEY = (Get-Content -Raw $KeyPath).Trim()
$env:FORTE_RUNTIME_DIR = $SecretFolder
$env:FORTE_MODEL_PATH = Join-Path $ProjectRoot 'models\Qwen3-VL-8B-Instruct'
$env:FORTE_GROUNDING_ENABLED = '0'
$env:FORTE_OVERVIEW_FRAMES = '32'
$env:FORTE_OVERVIEW_PIXELS = '262144'
$env:FORTE_PRIVATE_DIR = Join-Path $ProjectRoot 'runtime\private\local_demo'
$env:PYTHONUTF8 = '1'
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$server = Start-Process -FilePath $Python -ArgumentList '-m uvicorn forte.api:app --host 127.0.0.1 --port 8002 --no-access-log' -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $SecretFolder 'api.stdout.log') -RedirectStandardError (Join-Path $SecretFolder 'api.stderr.log') -PassThru
$server.Id | Set-Content (Join-Path $SecretFolder 'api.pid')
$reportRoot = Join-Path $ProjectRoot 'runtime\local_visual'
if (-not (Test-Path (Join-Path $SecretFolder 'report.pid'))) {
$static = Start-Process -FilePath $Python -ArgumentList @('-m','http.server','8767','--bind','127.0.0.1','--directory',('"'+$reportRoot+'"')) -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $SecretFolder 'report.stdout.log') -RedirectStandardError (Join-Path $SecretFolder 'report.stderr.log') -PassThru
$static.Id | Set-Content (Join-Path $SecretFolder 'report.pid')
}
Write-Output 'Local API: http://127.0.0.1:8002 ; local report: http://127.0.0.1:8767/visual_report.html'
