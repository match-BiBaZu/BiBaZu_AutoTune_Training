param()
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$autotuneSource = Join-Path (Split-Path -Parent $PSScriptRoot) 'BiBaZu_AutoTune'
if (-not (Test-Path -LiteralPath (Join-Path $autotuneSource 'pyproject.toml'))) {
    throw 'Das gemeinsame Repo BiBaZu_AutoTune muss neben BiBaZu_AutoTune_Training liegen.'
}
if (Get-Command uv -ErrorAction SilentlyContinue) {
    & uv sync --extra dev
    if ($LASTEXITCODE -ne 0) { throw 'Installation mit uv fehlgeschlagen.' }
} else {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        if (Get-Command py -ErrorAction SilentlyContinue) { & py -3 -m venv .venv }
        elseif (Get-Command python -ErrorAction SilentlyContinue) { & python -m venv .venv }
        else { throw 'Python 3.11+ oder uv installieren und Setup erneut starten.' }
        if ($LASTEXITCODE -ne 0) { throw 'Virtuelle Umgebung konnte nicht erstellt werden.' }
    }
    & .\.venv\Scripts\python.exe -m pip install -e $autotuneSource -e '.[dev]'
    if ($LASTEXITCODE -ne 0) { throw 'Paketinstallation fehlgeschlagen.' }
}
Write-Host 'Bereit. Start.cmd öffnet die Trainings-GUI mit eigenem SPS-Dienst.'
Write-Host 'Start.cmd --demo simuliert SPS und Kamera ohne Zugriff auf die Anlage.'
