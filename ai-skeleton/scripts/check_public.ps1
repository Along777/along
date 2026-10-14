# The along repo is public and GitHub Pages publishes every file in it.
# Run before every push: fails if anything private is staged or tracked under ai-skeleton/.
# Keep this file ASCII-only.
$repo = git rev-parse --show-toplevel
if ($LASTEXITCODE -ne 0) { Write-Host "not in a git repo" -ForegroundColor Red; exit 2 }
Push-Location $repo
try {
    $paths = @(git diff --cached --name-only) + @(git ls-files ai-skeleton)
    $bad = $paths | Sort-Object -Unique | Where-Object {
        $_ -match '\.(wav|onnx|pt|bin|mp3|flac)$' -or
        $_ -match '(^|/)(voices|debug|logs|models|cache|\.venv)/' -or
        $_ -match 'config\.local\.toml$'
    }
    if ($bad) {
        Write-Host "NOT SAFE TO PUSH - private or bulky files are staged/tracked:" -ForegroundColor Red
        $bad | ForEach-Object { Write-Host "  $_" }
        exit 1
    }
    Write-Host "ok: nothing private staged or tracked under ai-skeleton/" -ForegroundColor Green
    exit 0
} finally {
    Pop-Location
}
