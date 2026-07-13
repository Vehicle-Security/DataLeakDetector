param(
  [string]$CaseRoot = "spec\data\nas_samples",
  [string]$RunDir = "",
  [int]$PrecomputeWorkers = 8,
  [int]$VlmCaseWorkers = 4,
  [int]$VlmWorkers = 4,
  [string]$VlmGridLayout = "4x1",
  [int]$VlmRetryAttempts = 6,
  [double]$VlmRetryBackoffSeconds = 2,
  [switch]$UseCodingPlan,
  [switch]$UseNeo4jLogMinerForPrecompute
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

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

function Invoke-PythonLogged {
  param(
    [string[]]$Arguments,
    [string]$LogPath
  )

  $stdoutPath = Resolve-LogPath $LogPath
  $stderrPath = Resolve-LogPath "$LogPath.stderr.log"
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = "python"
  $psi.WorkingDirectory = (Get-Location).Path
  $psi.UseShellExecute = $false
  $psi.CreateNoWindow = $true
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  $psi.Arguments = Join-CommandArguments $Arguments

  $process = New-Object System.Diagnostics.Process
  $process.StartInfo = $psi
  $stdoutStream = [System.IO.File]::Open($stdoutPath, "Create", "Write", "ReadWrite")
  $stderrStream = [System.IO.File]::Open($stderrPath, "Create", "Write", "ReadWrite")
  try {
    [void]$process.Start()
    $stdoutTask = $process.StandardOutput.BaseStream.CopyToAsync($stdoutStream)
    $stderrTask = $process.StandardError.BaseStream.CopyToAsync($stderrStream)
    $process.WaitForExit()
    [void]$stdoutTask.GetAwaiter().GetResult()
    [void]$stderrTask.GetAwaiter().GetResult()
    $script:LastPythonExitCode = $process.ExitCode
  } finally {
    $stdoutStream.Dispose()
    $stderrStream.Dispose()
    $process.Dispose()
  }

  $stderr = [System.IO.File]::ReadAllText($stderrPath, [System.Text.Encoding]::UTF8)
  if ($stderr) {
    [System.IO.File]::AppendAllText($stdoutPath, $stderr, [System.Text.UTF8Encoding]::new($false))
  }
}

function Start-PythonLogged {
  param(
    [string[]]$Arguments,
    [string]$LogPrefix
  )

  $stdoutPath = Resolve-LogPath "$LogPrefix.stdout.log"
  $stderrPath = Resolve-LogPath "$LogPrefix.stderr.log"
  return Start-Process `
    -FilePath "python" `
    -ArgumentList (Join-CommandArguments $Arguments) `
    -WorkingDirectory (Get-Location).Path `
    -NoNewWindow `
    -PassThru `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath
}

function Merge-ProcessLogs {
  param([string]$LogPrefix)

  $stdoutPath = "$LogPrefix.stdout.log"
  $stderrPath = "$LogPrefix.stderr.log"
  $combined = ""
  if (Test-Path $stdoutPath) {
    $combined = [System.IO.File]::ReadAllText((Resolve-Path $stdoutPath), [System.Text.Encoding]::UTF8)
  }
  if (Test-Path $stderrPath) {
    $stderr = [System.IO.File]::ReadAllText((Resolve-Path $stderrPath), [System.Text.Encoding]::UTF8)
    if ($stderr) {
      if ($combined -and -not $combined.EndsWith("`n")) {
        $combined += "`n"
      }
      $combined += $stderr
    }
  }
  [System.IO.File]::WriteAllText((Resolve-LogPath "$LogPrefix.log"), $combined, [System.Text.UTF8Encoding]::new($false))
}

if (-not $RunDir) {
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $RunDir = "artifacts\full_release_grid4x1_$stamp"
}
if ($VlmGridLayout -notmatch '^\d+x\d+$') {
  throw "VlmGridLayout must use ROWSxCOLUMNS, for example 4x1."
}

$env:DLD_VLM_RETRY_ATTEMPTS = "$VlmRetryAttempts"
$env:DLD_VLM_RETRY_BACKOFF_SECONDS = "$VlmRetryBackoffSeconds"
if ($UseCodingPlan) {
  $env:DLD_VLM_USE_CODING_PLAN = "1"
}
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$precomputeLog = Join-Path $RunDir "precompute.log"
$vlmLogPrefix = Join-Path $RunDir "vlm"
$vlmPreflightLog = Join-Path $RunDir "vlm_preflight.log"

Write-Host "Run directory: $RunDir"
Write-Host "Case root: $CaseRoot"
Write-Host "VLM grid layout: $VlmGridLayout"
Write-Host "Coding endpoint override: $(if ($UseCodingPlan) { 'enabled' } else { 'use .env/default' })"

$vlmPreflightCode = @"
from data_leak_detector.frame_analyzer.config import VisionConfig
c = VisionConfig.from_env()
endpoints = c.effective_vlm_endpoints()
if c.vlm_dry_run:
    raise SystemExit('VLM preflight failed: DLD_VLM_DRY_RUN is enabled')
if not endpoints:
    raise SystemExit('VLM preflight failed: no enabled endpoint/API key pair')
print(f'VLM preflight passed: model={c.vlm_model} endpoints={len(endpoints)} workers={c.vlm_workers}')
"@
Write-Host "Checking VLM configuration before precompute..."
Invoke-PythonLogged -Arguments @("-c", $vlmPreflightCode) -LogPath $vlmPreflightLog
if ($script:LastPythonExitCode -ne 0) {
  throw "VLM preflight failed. See $vlmPreflightLog"
}

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

Write-Host "Starting full release precompute..."
Invoke-PythonLogged -Arguments $precomputeArgs -LogPath $precomputeLog
if ($script:LastPythonExitCode -ne 0) {
  throw "Release precompute failed. See $precomputeLog"
}

$vlmArgs = @(
  "main/run_e2e.py",
  "--case-root", $CaseRoot,
  "--case-workers", "$VlmCaseWorkers",
  "--release",
  "--release-debug-artifacts",
  "--output-dir", $RunDir,
  "--vlm-grid-layout", $VlmGridLayout,
  "--vlm-workers", "$VlmWorkers",
  "--vlm-fast-dispatch"
)
$vlmProcess = $null
try {
  Write-Host "Starting full release VLM..."
  $vlmProcess = Start-PythonLogged -Arguments $vlmArgs -LogPrefix $vlmLogPrefix
  $vlmProcess.WaitForExit()
  Merge-ProcessLogs -LogPrefix $vlmLogPrefix
  if ($vlmProcess.ExitCode -ne 0) {
    throw "Release VLM failed. See $vlmLogPrefix.log"
  }
} finally {
  if ($vlmProcess -and -not $vlmProcess.HasExited) {
    Stop-Process -Id $vlmProcess.Id -Force
  }
}

Write-Host "Done."
Write-Host "Release report: $(Join-Path $RunDir "release_report.json")"
Write-Host "Comparison: $(Join-Path $RunDir "release_comparison.json")"
