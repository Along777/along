# One-time setup for the porch laptop (Windows). Safe to re-run.
# From the ai-skeleton folder:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_porch.ps1 -BrainUrl http://192.168.1.50:8080
# Keep this file ASCII-only: Windows PowerShell 5.1 misreads other characters.
param([string]$BrainUrl = "")

$root = Split-Path -Parent $PSScriptRoot
$porch = Join-Path $root "porch"
$brain = Join-Path $root "brain"

function Say($m)  { Write-Host ""; Write-Host "== $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "   ok: $m" -ForegroundColor Green }
function Warn($m) { Write-Host " warn: $m" -ForegroundColor Yellow }
function Fail($m) { Write-Host " fail: $m" -ForegroundColor Red }

# uv keeps Python + its cache under ~/.uv, not AppData. Packaged apps (the Claude
# desktop app, for one) redirect AppData writes, which breaks uv's Python links.
$env:UV_PYTHON_INSTALL_DIR = Join-Path $HOME ".uv\python"
$env:UV_CACHE_DIR = Join-Path $HOME ".uv\cache"

function Find-Uv {
    $c = Get-Command uv -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    $f = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter uv.exe -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($f) { return $f.FullName }
    return $null
}

Say "uv (Python manager)"
$uv = Find-Uv
if (-not $uv) {
    winget install --id astral-sh.uv -e --source winget --accept-package-agreements
    $uv = Find-Uv
}
if (-not $uv) { Fail "uv not found - install it: winget install astral-sh.uv"; exit 1 }
Ok "$uv"

Say "Python environments (Python 3.12, managed by uv)"
& $uv sync --project $porch
if ($LASTEXITCODE -ne 0) { Fail "porch environment"; exit 1 }
Ok "porch"
& $uv sync --project $brain
if ($LASTEXITCODE -ne 0) { Warn "brain environment (only needed on the laptop for tests)" } else { Ok "brain (for tests)" }

Say "Person detector weights"
& $uv run --project $porch skeleton-porch fetch-models
if ($LASTEXITCODE -ne 0) { Fail "could not download yolo26n.pt"; exit 1 }

Say "Local config (porch\config.local.toml, never committed)"
$local = Join-Path $porch "config.local.toml"
if (-not (Test-Path $local)) {
    $url = $BrainUrl
    if (-not $url) { $url = "http://127.0.0.1:8080" }
    $text = "# This laptop's settings. Overrides porch\config.toml; never committed.`r`n" +
            "[brain]`r`nurl = `"$url`"`r`n`r`n" +
            "[audio]`r`n# speaker = `"SoundCore`"   # part of the speaker's name; empty = Windows default`r`n" +
            "# tail_margin_s = 0.5        # use 0.5 for a Bluetooth speaker`r`n"
    [System.IO.File]::WriteAllText($local, $text)
    Ok "wrote $local (brain = $url)"
} else {
    Ok "$local already exists"
    if ($BrainUrl) { Warn "edit [brain] url in $local yourself to change it to $BrainUrl" }
}

Say "Windows checks (warnings only)"
$public = Get-NetConnectionProfile -ErrorAction SilentlyContinue | Where-Object { $_.NetworkCategory -eq "Public" }
foreach ($p in $public) {
    Warn "network '$($p.Name)' is Public: the phone mute page is blocked. Settings > Network & internet > Wi-Fi > $($p.Name) > Private"
}
if (-not $public) { Ok "no Public networks" }
$vpn = Get-NetAdapter -ErrorAction SilentlyContinue | Where-Object {
    $_.Status -eq "Up" -and ($_.InterfaceDescription -match "VPN|WireGuard|Wintun|TAP|OpenVPN" -or $_.Name -match "VPN")
}
foreach ($v in $vpn) { Warn "VPN '$($v.Name)' is up: allow LAN connections in it, or disconnect for the night" }
if (-not $vpn) { Ok "no VPN up" }

function Get-AcIndex($sub, $setting) {
    $out = powercfg /query SCHEME_CURRENT $sub $setting 2>$null | Out-String
    $m = [regex]::Match($out, "Current AC Power Setting Index:\s+0x([0-9a-fA-F]+)")
    if ($m.Success) { return [Convert]::ToInt32($m.Groups[1].Value, 16) }
    return $null
}
$lid = Get-AcIndex "SUB_BUTTONS" "LIDACTION"
if ($lid -eq 0) { Ok "closing the lid does nothing on mains power" }
elseif ($lid -ne $null) { Warn "closing the lid sleeps the laptop: Control Panel > Power Options > Choose what closing the lid does > Plugged in: Do nothing" }
$idle = Get-AcIndex "SUB_SLEEP" "STANDBYIDLE"
if ($idle -eq 0) { Ok "never sleeps on mains power" }
elseif ($idle -ne $null) { Warn "sleeps after $([math]::Round($idle/60)) min on mains power (the porch asks Windows to stay awake while it runs, but Never is safer for the night)" }

foreach ($cap in @("microphone", "webcam")) {
    $key = "HKCU:\Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\$cap"
    $v = (Get-ItemProperty -Path $key -Name Value -ErrorAction SilentlyContinue).Value
    $nv = (Get-ItemProperty -Path "$key\NonPackaged" -Name Value -ErrorAction SilentlyContinue).Value
    if ($v -eq "Deny" -or $nv -eq "Deny") { Warn "$cap access is off for desktop apps: Settings > Privacy & security > $cap" }
    else { Ok "$cap access allowed" }
}

Say "Doctor"
& $uv run --project $porch skeleton-porch doctor
exit $LASTEXITCODE
