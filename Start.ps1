param([Parameter(ValueFromRemainingArguments=$true)][string[]]$AppArguments)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    throw 'Zuerst Setup.ps1 im Training-Repo ausführen.'
}
& .\.venv\Scripts\python.exe -m bibazu_autotune_training @AppArguments
exit $LASTEXITCODE
