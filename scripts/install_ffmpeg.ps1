$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'runtime_env.ps1')
$DownloadRoot = Join-Path $ProjectRoot '.cache/downloads'
$ArchivePath = Join-Path $DownloadRoot 'ffmpeg-release-essentials.7z'
$ArchiveSha256 = '4705843ccaaf54257c16ad90f3e952ece33c17df964ecf7bfdbb0f49c7171077'
$ArchiveBytes = 35430500
$DownloadUrl = 'https://github.com/GyanD/codexffmpeg/releases/download/9.0.2/ffmpeg-9.0.2-essentials_build.7z'
$InstallRoot = Join-Path $ProjectRoot 'tools/ffmpeg'
$StageRoot = Join-Path $DownloadRoot 'ffmpeg-9.0.2-stage'
New-Item -ItemType Directory -Path $DownloadRoot -Force | Out-Null
if (-not (Test-Path -LiteralPath $ArchivePath) -or (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ArchiveSha256) {
    if ((Test-Path -LiteralPath $ArchivePath) -and (Get-Item -LiteralPath $ArchivePath).Length -ge $ArchiveBytes) {
        # An invalid complete archive cannot be repaired by appending bytes.
        Remove-Item -LiteralPath $ArchivePath
    }
    curl.exe --fail --location --silent --show-error --connect-timeout 15 --retry 3 --continue-at - --output $ArchivePath $DownloadUrl
    if ($LASTEXITCODE -ne 0) { throw 'FFmpeg download failed; re-run to resume' }
}
if ((Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ArchiveSha256) {
    throw 'FFmpeg archive checksum mismatch; refusing to execute extracted programs'
}
New-Item -ItemType Directory -Path $StageRoot -Force | Out-Null
tar.exe -xf $ArchivePath -C $StageRoot
if ($LASTEXITCODE -ne 0) { throw 'FFmpeg archive extraction failed' }
$ExtractedRoot = Join-Path $StageRoot 'ffmpeg-9.0.2-essentials_build'
if (-not (Test-Path -LiteralPath (Join-Path $ExtractedRoot 'bin/ffprobe.exe'))) {
    throw 'Unexpected FFmpeg archive layout'
}
New-Item -ItemType Directory -Path $InstallRoot -Force | Out-Null
Get-ChildItem -LiteralPath $ExtractedRoot | Copy-Item -Destination $InstallRoot -Recurse -Force
@{
    version = '9.0.2'; source = $DownloadUrl; sha256 = $ArchiveSha256
    license = 'GPL-3.0'; verified_at = [DateTime]::UtcNow.ToString('o')
    official_download_page = 'https://ffmpeg.org/download.html'
    provider_page = 'https://www.gyan.dev/ffmpeg/builds/'
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $InstallRoot 'provenance.json') -Encoding utf8
& (Join-Path $InstallRoot 'bin/ffmpeg.exe') -version | Select-Object -First 1
& (Join-Path $InstallRoot 'bin/ffprobe.exe') -version | Select-Object -First 1
