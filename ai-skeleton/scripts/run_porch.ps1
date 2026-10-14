# Run the porch, restarting it if it ever crashes. Ctrl+C stops it.
#   powershell -ExecutionPolicy Bypass -File scripts\run_porch.ps1 [-SaveCaptures] [-Preview]
# -SaveCaptures writes photos + clips to porch\debug\ (living-room tests only, never Halloween night).
# Keep this file ASCII-only.
param([switch]$SaveCaptures, [switch]$Preview)

$root = Split-Path -Parent $PSScriptRoot
$porch = Join-Path $root "porch"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $HOME ".uv\python"
$env:UV_CACHE_DIR = Join-Path $HOME ".uv\cache"

$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    $f = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter uv.exe -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($f) { $uv = $f.FullName }
}
if (-not $uv) { Write-Host "uv not found - run scripts\setup_porch.ps1 first" -ForegroundColor Red; exit 1 }

$a = @("run", "--project", $porch, "--frozen", "--offline", "skeleton-porch", "run")
if ($SaveCaptures) { $a += "--debug" }
if ($Preview) { $a += "--preview" }

while ($true) {
    & $uv @a
    if ($LASTEXITCODE -eq 0) { break }
    Write-Host "porch exited with code $LASTEXITCODE - restarting in 3 s (Ctrl+C to stop)" -ForegroundColor Yellow
    Start-Sleep -Seconds 3
}
