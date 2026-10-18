$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$Supervisor = Join-Path $PSScriptRoot 'serve.py'
$StateFile = Join-Path $ProjectRoot 'runtime\service_state.json'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Project venv Python is missing.' }
try {
    $RequestedState = Get-Content -LiteralPath $StateFile -Raw | ConvertFrom-Json
    $RequestedInstance = $RequestedState.instance
} catch { $RequestedInstance = $null }
if (-not $RequestedInstance) {
    Write-Output 'No supervisor instance could be identified; no stop request was written.'
    return
}
& $Python $Supervisor --stop --instance $RequestedInstance
if ($LASTEXITCODE -ne 0) { return }

$WaitUntil = [DateTime]::UtcNow.AddSeconds(45)
while ([DateTime]::UtcNow -lt $WaitUntil) {
    Start-Sleep -Seconds 1
    try {
        $ServiceState = Get-Content -LiteralPath $StateFile -Raw | ConvertFrom-Json
        if ($ServiceState.instance -eq $RequestedInstance -and $ServiceState.status -eq 'stopped') {
            Write-Output 'Service and its owned tunnel stopped. Local review service is unaffected.'
            return
        }
        if ($RequestedInstance -and $ServiceState.instance -ne $RequestedInstance) {
            Write-Output 'The requested supervisor was replaced by another instance; this helper will not stop the new instance.'
            return
        }
    } catch {
        # An atomic state-file replacement may briefly race with this read.
    }
}
Write-Output 'Stop was requested and is still pending. Inspect runtime\logs\supervisor.log; no unrelated process was terminated.'
