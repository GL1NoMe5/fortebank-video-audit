param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$RunFolder = Join-Path $ProjectRoot 'runtime\local_working'
New-Item -ItemType Directory -Force $RunFolder | Out-Null
$KeyPath = Join-Path $RunFolder 'api_key.txt'
if (-not (Test-Path -LiteralPath $KeyPath)) {
    ([guid]::NewGuid().ToString('N') + [guid]::NewGuid().ToString('N')) | Set-Content -LiteralPath $KeyPath -Encoding ASCII
}
$env:FORTE_API_KEY = (Get-Content -Raw -LiteralPath $KeyPath).Trim()
$env:FORTE_RUNTIME_DIR = $RunFolder
$env:FORTE_PRIVATE_DIR = Join-Path $RunFolder 'private'
$env:FORTE_MODEL_PATH = Join-Path $ProjectRoot 'models\Qwen3-VL-8B-Instruct'
$env:FORTE_GROUNDING_ENABLED = '0'
$env:FORTE_ANALYSIS_TIMEOUT = '285'
$env:PYTHONUTF8 = '1'
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$listener = Get-NetTCPConnection -State Listen -LocalPort 8002 -ErrorAction SilentlyContinue
if ($listener) { throw 'Port 8002 is occupied. No existing process was stopped.' }
$server = Start-Process -FilePath $Python -ArgumentList '-m uvicorn forte.api:app --host 127.0.0.1 --port 8002 --no-access-log' -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $RunFolder 'api.stdout.log') -RedirectStandardError (Join-Path $RunFolder 'api.stderr.log') -PassThru
$server.Id | Set-Content -LiteralPath (Join-Path $RunFolder 'api.pid')
for ($attempt = 0; $attempt -lt 15; $attempt++) {
    try {
        $response = Invoke-WebRequest -Uri 'http://127.0.0.1:8002/' -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            if (-not $NoBrowser) { Start-Process ('http://127.0.0.1:8002/#key=' + $env:FORTE_API_KEY) }
            Write-Output 'Local interface started: http://127.0.0.1:8002/ ; model warms up in background.'
            exit 0
        }
    } catch { Start-Sleep -Milliseconds 500 }
}
throw 'Server started but page availability was not confirmed; inspect runtime/local_working/api.stderr.log.'
