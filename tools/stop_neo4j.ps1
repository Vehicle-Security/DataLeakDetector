<#
Stops the project-local Neo4j runtime started by tools/start_neo4j.ps1.

The script first uses the recorded pid file, then falls back to finding a
process launched from this repository's .runtime/neo4j directory. It exists so
developers can cleanly stop the optional graph service without affecting other
Neo4j installations.
#>

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Runtime = Join-Path $Root ".runtime"
$Neo4jDir = Join-Path $Runtime "neo4j"
$PidFile = Join-Path $Runtime "neo4j.pid"

$Stopped = $false
if (Test-Path -LiteralPath $PidFile) {
    $PidValue = Get-Content -LiteralPath $PidFile | Select-Object -First 1
    if ($PidValue) {
        $Process = Get-Process -Id ([int]$PidValue) -ErrorAction SilentlyContinue
        if ($Process) {
            Stop-Process -Id $Process.Id -Force
            $Stopped = $true
        }
    }
    Remove-Item -LiteralPath $PidFile -Force
}

if (-not $Stopped -and (Test-Path -LiteralPath $Neo4jDir)) {
    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -like "*$Neo4jDir*" -and $_.CommandLine -like "*org.neo4j*" } |
        ForEach-Object {
            Stop-Process -Id $_.ProcessId -Force
            $Stopped = $true
        }
}

if ($Stopped) {
    Write-Output "Neo4j local runtime stopped."
} else {
    Write-Output "No Neo4j local runtime process found."
}
