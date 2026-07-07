<#
Starts the optional local Neo4j runtime used by DataLeakDetector.

The script downloads a project-local JRE and Neo4j Community distribution into
.runtime, writes the DLD_NEO4J_* values into .env when missing, and starts the
database without requiring a system-wide Neo4j installation. It is necessary
for Windows development environments where Docker or a preinstalled Neo4j
service may not be available.
#>

param(
    [string]$Neo4jVersion = "2026.05.0",
    [string]$Password = "data-leak-detector"
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Runtime = Join-Path $Root ".runtime"
$JreDir = Join-Path $Runtime "jre"
$Neo4jDir = Join-Path $Runtime "neo4j"
$PidFile = Join-Path $Runtime "neo4j.pid"

New-Item -ItemType Directory -Path $Runtime -Force | Out-Null

function Install-Jre {
    if (Test-Path -LiteralPath $JreDir) { return }
    $Zip = Join-Path $Runtime "temurin-jre.zip"
    if (-not (Test-Path -LiteralPath $Zip)) {
        Invoke-WebRequest `
            -Uri "https://api.adoptium.net/v3/binary/latest/21/ga/windows/x64/jre/hotspot/normal/eclipse" `
            -OutFile $Zip
    }
    $Extract = Join-Path $Runtime "jre_extract"
    if (Test-Path -LiteralPath $Extract) { Remove-Item -LiteralPath $Extract -Recurse -Force }
    Expand-Archive -LiteralPath $Zip -DestinationPath $Extract -Force
    $HomeDir = Get-ChildItem -LiteralPath $Extract -Directory | Select-Object -First 1
    Move-Item -LiteralPath $HomeDir.FullName -Destination $JreDir
    Remove-Item -LiteralPath $Extract -Recurse -Force
}

function Install-Neo4j {
    if (Test-Path -LiteralPath $Neo4jDir) { return }
    $Zip = Join-Path $Runtime "neo4j-community-$Neo4jVersion-windows.zip"
    if (-not (Test-Path -LiteralPath $Zip)) {
        Invoke-WebRequest `
            -Uri "https://dist.neo4j.org/neo4j-community-$Neo4jVersion-windows.zip" `
            -OutFile $Zip
    }
    $Extract = Join-Path $Runtime "neo4j_extract"
    if (Test-Path -LiteralPath $Extract) { Remove-Item -LiteralPath $Extract -Recurse -Force }
    Expand-Archive -LiteralPath $Zip -DestinationPath $Extract -Force
    $HomeDir = Get-ChildItem -LiteralPath $Extract -Directory | Select-Object -First 1
    Move-Item -LiteralPath $HomeDir.FullName -Destination $Neo4jDir
    Remove-Item -LiteralPath $Extract -Recurse -Force
}

function Configure-Neo4j {
    $Conf = Join-Path $Neo4jDir "conf\neo4j.conf"
    $Marker = "DataLeakDetector local runtime"
    $Text = Get-Content -Raw -LiteralPath $Conf
    if ($Text -notmatch $Marker) {
        Add-Content -LiteralPath $Conf -Value @"

# $Marker
server.default_listen_address=127.0.0.1
server.bolt.listen_address=:7687
server.http.listen_address=:7474
server.jvm.additional=-Dfile.encoding=UTF-8
"@
    }
}

function Set-LocalEnvFile {
    $EnvFile = Join-Path $Root ".env"
    $Lines = @(
        "DLD_NEO4J_ENABLED=1",
        "DLD_NEO4J_URI=bolt://localhost:7687",
        "DLD_NEO4J_USER=neo4j",
        "DLD_NEO4J_PASSWORD=$Password",
        "DLD_NEO4J_DATABASE=neo4j"
    )
    $Existing = if (Test-Path -LiteralPath $EnvFile) { Get-Content -LiteralPath $EnvFile } else { @() }
    foreach ($Line in $Lines) {
        $Key = $Line.Split("=")[0]
        if (-not ($Existing -match "^$Key=")) {
            Add-Content -LiteralPath $EnvFile -Value $Line
        }
    }
}

Install-Jre
Install-Neo4j

$env:JAVA_HOME = $JreDir
$env:PATH = (Join-Path $JreDir "bin") + ";" + $env:PATH

Configure-Neo4j
Set-LocalEnvFile

$Process = Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like "*$Neo4jDir*" -and $_.CommandLine -like "*org.neo4j*" } |
    Select-Object -First 1

if (-not $Process) {
    & (Join-Path $Neo4jDir "bin\neo4j-admin.bat") dbms set-initial-password $Password | Out-Null
    $Started = Start-Process `
        -FilePath (Join-Path $Neo4jDir "bin\neo4j.bat") `
        -ArgumentList "console" `
        -WorkingDirectory $Neo4jDir `
        -PassThru `
        -WindowStyle Hidden
    Set-Content -LiteralPath $PidFile -Value $Started.Id
}

Write-Output "Neo4j local runtime is starting on bolt://localhost:7687"
