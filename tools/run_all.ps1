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
$env:DLD_VLM_USE_CODING_PLAN = if ($UseCodingPlan) { "1" } else { "0" }
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$precomputeLog = Join-Path $RunDir "precompute.log"
$vlmLogPrefix = Join-Path $RunDir "vlm"

Write-Host "Run directory: $RunDir"
Write-Host "Case root: $CaseRoot"
Write-Host "VLM grid layout: $VlmGridLayout"
Write-Host "Coding plan enabled: $($UseCodingPlan.IsPresent)"

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
