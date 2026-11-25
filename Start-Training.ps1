param([Parameter(ValueFromRemainingArguments=$true)][string[]]$AppArguments)
& (Join-Path $PSScriptRoot 'Start.ps1') @AppArguments
exit $LASTEXITCODE
