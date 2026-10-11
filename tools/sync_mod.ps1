param(
    [string]$Destination = 'C:\Users\Sfilya\Documents\Paradox Interactive\Hearts of Iron IV\mod\smeshariki'
)
$ErrorActionPreference = 'Stop'
$taskSource = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$taskDestination = (Resolve-Path -LiteralPath $Destination).Path
if ($taskSource -eq $taskDestination) { throw 'Source and destination must differ.' }
$taskReport = Get-Content -LiteralPath (Join-Path $taskSource 'build\validation_report.json') -Raw | ConvertFrom-Json
if ($taskReport.failed -ne 0) { throw 'Only a validated build may be installed.' }
$taskFiles = @(Get-Content -LiteralPath (Join-Path $taskSource 'build\generated_files.json') -Raw | ConvertFrom-Json)
$taskFiles += @('README.md','build/generated_files.json','build/validation_report.json','build/preserved_user_files.json','build/game_verification.json')
$taskFiles += @(Get-ChildItem -LiteralPath (Join-Path $taskSource 'tools') -File | ForEach-Object { 'tools/' + $_.Name })
$taskFiles = @($taskFiles | Sort-Object -Unique)
$taskChanges = @()
foreach ($taskRelative in $taskFiles) {
    if ([System.IO.Path]::IsPathRooted($taskRelative) -or ($taskRelative -split '[\\/]') -contains '..') { throw 'Invalid manifest path.' }
    $taskInput = [System.IO.Path]::GetFullPath((Join-Path $taskSource $taskRelative))
    $taskOutput = [System.IO.Path]::GetFullPath((Join-Path $taskDestination $taskRelative))
    if (-not $taskInput.StartsWith($taskSource + '\',[System.StringComparison]::OrdinalIgnoreCase) -or -not $taskOutput.StartsWith($taskDestination + '\',[System.StringComparison]::OrdinalIgnoreCase)) { throw 'Path left its intended mod directory.' }
    if (-not (Test-Path -LiteralPath $taskInput -PathType Leaf)) { throw ('Missing generated file: ' + $taskRelative) }
    $taskIsDifferent = -not (Test-Path -LiteralPath $taskOutput -PathType Leaf)
    if (-not $taskIsDifferent) { $taskIsDifferent = (Get-FileHash -LiteralPath $taskInput).Hash -ne (Get-FileHash -LiteralPath $taskOutput).Hash }
    if ($taskIsDifferent) { $taskChanges += [pscustomobject]@{ Relative=$taskRelative; Input=$taskInput; Output=$taskOutput } }
}
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$taskBackup = Join-Path $taskSource ('build\installed_before_locations_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.zip')
$taskArchive = [System.IO.Compression.ZipFile]::Open($taskBackup,[System.IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($taskChange in $taskChanges) {
        if (Test-Path -LiteralPath $taskChange.Output -PathType Leaf) {
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($taskArchive,$taskChange.Output,$taskChange.Relative,[System.IO.Compression.CompressionLevel]::Optimal) | Out-Null
        }
    }
} finally { $taskArchive.Dispose() }
foreach ($taskChange in $taskChanges) {
    $taskParent = Split-Path -Parent $taskChange.Output
    if (-not (Test-Path -LiteralPath $taskParent -PathType Container)) { New-Item -ItemType Directory -Path $taskParent -Force | Out-Null }
    Copy-Item -LiteralPath $taskChange.Input -Destination $taskChange.Output -Force
    if ((Get-FileHash -LiteralPath $taskChange.Input).Hash -ne (Get-FileHash -LiteralPath $taskChange.Output).Hash) { throw ('Copy verification failed: ' + $taskChange.Relative) }
}
$taskRecord = [ordered]@{ source=$taskSource; destination=$taskDestination; backup=$taskBackup; copied=$taskChanges.Count; verified=$true; installed_at=(Get-Date -Format 'o'); files=@($taskChanges | ForEach-Object { $_.Relative }) }
$taskRecord | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $taskSource 'build\installed_sync.json') -Encoding utf8
Write-Host ('Installed and hash-verified ' + $taskChanges.Count + ' changed files.')
Write-Host ('Backup: ' + $taskBackup)
