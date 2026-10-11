param(
    [string]$GameDirectory = 'C:\Program Files (x86)\Steam\steamapps\common\Hearts of Iron IV',
    [string]$PythonPath = ''
)
$ErrorActionPreference = 'Stop'
if (-not $PythonPath) {
    $taskBundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (Test-Path -LiteralPath $taskBundledPython) {
        $PythonPath = $taskBundledPython
    } else {
        $PythonPath = (Get-Command python.exe -ErrorAction Stop).Source
    }
}
& $PythonPath (Join-Path $PSScriptRoot 'generate_world.py') --game $GameDirectory
if ($LASTEXITCODE -ne 0) { throw 'Map generation failed.' }
& $PythonPath (Join-Path $PSScriptRoot 'validate_world.py')
if ($LASTEXITCODE -ne 0) { throw 'Map validation failed.' }
Write-Host 'Map rebuilt and validated. Restart HOI4 and start a new game.'
