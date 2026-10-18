$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$Supervisor = Join-Path $PSScriptRoot 'serve.py'
$Cloudflared = Join-Path $ProjectRoot 'tools\cloudflared\cloudflared.exe'
foreach ($RequiredFile in @($Python, $Supervisor, $Cloudflared)) {
    if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
        throw "Required service component is missing: $RequiredFile"
    }
}

& $Python $Supervisor --status
if ($LASTEXITCODE -eq 0) {
    Write-Output 'The service already has a validated supervisor.'
    return
}

. (Join-Path $PSScriptRoot 'runtime_env.ps1')
$LogDirectory = Join-Path $ProjectRoot 'runtime\logs'
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$LaunchId = (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
$StdoutLog = Join-Path $LogDirectory "launch-$LaunchId.stdout.log"
$StderrLog = Join-Path $LogDirectory "launch-$LaunchId.stderr.log"
$Arguments = '-u "{0}"' -f $Supervisor
$ServiceProcess = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog -PassThru
Start-Sleep -Seconds 2
$ServiceProcess.Refresh()
if ($ServiceProcess.HasExited) {
    & $Python $Supervisor --status
    if ($LASTEXITCODE -eq 0) { return }
    throw "Supervisor exited during startup. See $StderrLog and runtime\logs\supervisor.log."
}
Write-Output "Supervisor started in the background (PID $($ServiceProcess.Id))."
Write-Output 'GPU warmup may take 120 seconds or longer. The tunnel starts only after /healthz reports ready=true.'
Write-Output 'Current temporary URL: runtime\public_endpoint.json. API key: runtime\secrets\api_key.txt (keep private).'
