param(
  [string]$CaseRoot = "spec\data\nas_samples",
  [string]$RunDir = "",
  [int]$PrecomputeWorkers = 2,
  [int]$VlmCaseWorkers = 4,
  [int]$VlmWorkers = 4,
  [string]$VlmGridSizes = "2,3",
  [switch]$UseNeo4jLogMinerForPrecompute
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

function Invoke-PythonLogged {
  param(
    [string[]]$Arguments,
    [string]$LogPath
  )

  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = "python"
  $psi.UseShellExecute = $false
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
  $psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
  $psi.Arguments = Join-CommandArguments $Arguments

  $process = New-Object System.Diagnostics.Process
  $process.StartInfo = $psi
  [void]$process.Start()
  $stdout = $process.StandardOutput.ReadToEnd()
  $stderr = $process.StandardError.ReadToEnd()
  $process.WaitForExit()

  $logText = $stdout
  if ($stderr) {
    if ($logText -and -not $logText.EndsWith("`n")) {
      $logText += "`n"
    }
    $logText += $stderr
  }
  [System.IO.File]::WriteAllText((Resolve-LogPath $LogPath), $logText, [System.Text.UTF8Encoding]::new($false))
  $script:LastPythonExitCode = $process.ExitCode
}

function Join-CommandArguments {
  param([string[]]$Arguments)

  return ($Arguments | ForEach-Object { Quote-CommandArgument $_ }) -join " "
}

function Quote-CommandArgument {
  param([string]$Value)

  if ($Value -notmatch '[\s"]') {
    return $Value
  }
  return '"' + ($Value -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
}

function Resolve-LogPath {
  param([string]$Path)

  $parent = Split-Path -Parent $Path
  if ($parent) {
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
  }
  return [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Path))
}

if (-not $RunDir) {
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $RunDir = "artifacts\full_release_direct_$stamp"
}

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$precomputeLog = Join-Path $RunDir "precompute.log"
$gridSizes = @(
  $VlmGridSizes -split "," |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ } |
    ForEach-Object { [int]$_ }
)
if (-not $gridSizes) {
  throw "At least one VLM grid size is required."
}

Write-Host "Run directory: $RunDir"
Write-Host "Precompute log: $precomputeLog"
Write-Host "VLM grids: $($gridSizes -join ', ')"

$precomputeArgs = @(
  "main/run_e2e.py",
  "--case-root", $CaseRoot,
  "--case-workers", "$PrecomputeWorkers",
  "--release",
  "--release-precompute-only",
  "--output-dir", $RunDir
)

if ($UseNeo4jLogMinerForPrecompute) {
  $precomputeArgs += "--release-precompute-neo4j-log-miner"
}

Write-Host "Starting release precompute..."
Invoke-PythonLogged -Arguments $precomputeArgs -LogPath $precomputeLog
if ($script:LastPythonExitCode -ne 0) {
  throw "Release precompute failed. See $precomputeLog"
}

foreach ($gridSize in $gridSizes) {
  if ($gridSize -lt 1) {
    throw "VLM grid size must be >= 1, got $gridSize"
  }

  $vlmLog = Join-Path $RunDir "vlm_grid$gridSize.log"
  $vlmArgs = @(
    "main/run_e2e.py",
    "--case-root", $CaseRoot,
    "--case-workers", "$VlmCaseWorkers",
    "--release",
    "--output-dir", $RunDir,
    "--vlm-grid-size", "$gridSize",
    "--vlm-workers", "$VlmWorkers",
    "--vlm-fast-dispatch"
  )

  Write-Host "Starting release VLM grid=$gridSize..."
  Invoke-PythonLogged -Arguments $vlmArgs -LogPath $vlmLog
  if ($script:LastPythonExitCode -ne 0) {
    throw "Release VLM grid=$gridSize failed. See $vlmLog"
  }

  Copy-Item -Force (Join-Path $RunDir "release_report.json") (Join-Path $RunDir "release_report_grid$gridSize.json")
  Copy-Item -Force (Join-Path $RunDir "release_comparison.json") (Join-Path $RunDir "release_comparison_grid$gridSize.json")
  Copy-Item -Force (Join-Path $RunDir "release_progress.json") (Join-Path $RunDir "release_progress_grid$gridSize.json")
}

Write-Host "Done."
foreach ($gridSize in $gridSizes) {
  Write-Host "Grid $gridSize release report: $(Join-Path $RunDir "release_report_grid$gridSize.json")"
  Write-Host "Grid $gridSize comparison: $(Join-Path $RunDir "release_comparison_grid$gridSize.json")"
}
