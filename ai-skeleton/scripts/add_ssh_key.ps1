# Give this laptop key-based SSH access to the brain, so checks and logs can be read
# from the laptop. Run once; it asks for the Linux password one time.
#   powershell -ExecutionPolicy Bypass -File scripts\add_ssh_key.ps1 -Target user@192.168.1.50
# Keep this file ASCII-only.
param([Parameter(Mandatory = $true)][string]$Target)

$pub = Join-Path $HOME ".ssh\id_ed25519.pub"
if (-not (Test-Path $pub)) {
    Write-Host "No key yet. Create one first:  ssh-keygen -t ed25519" -ForegroundColor Red
    exit 1
}
# accept-new: trust the brain's host key on this first connection (home network).
# tr -d removes the CR that Windows PowerShell adds to piped lines.
Get-Content $pub | ssh -o StrictHostKeyChecking=accept-new $Target "umask 077; mkdir -p ~/.ssh && tr -d '\r' >> ~/.ssh/authorized_keys && echo KEY-INSTALLED"
if ($LASTEXITCODE -eq 0) {
    Write-Host "Done. Test it:  ssh -o BatchMode=yes $Target hostname" -ForegroundColor Green
} else {
    Write-Host "Copy failed (exit $LASTEXITCODE). Wrong password, or SSH not running on the brain (step 3 of the guide)." -ForegroundColor Red
}
