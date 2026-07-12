param(
  [string]$CaseRoot = "spec\data\nas_samples",
  [string]$RunDir = "",
  [int]$PrecomputeWorkers = 2,
  [int]$VlmCaseWorkers = 4,
  [int]$VlmWorkers = 4,
  [string]$VlmGridLayout = "4x1",
  [int]$VlmRetryAttempts = 6,
  [double]$VlmRetryBackoffSeconds = 2,
  [string]$JudgeBaseUrl = "https://api.deepseek.com",
  [string]$JudgeModel = "deepseek-v4-pro",
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

function Wait-ForReleaseStart {
  param(
    [string]$ProgressPath,
    [System.Diagnostics.Process]$VlmProcess
  )

  while ($true) {
    if ($VlmProcess.HasExited) {
      throw "Release VLM exited before release_progress.json entered the running state."
    }
    if (Test-Path $ProgressPath) {
      try {
        $progress = Get-Content -Raw $ProgressPath | ConvertFrom-Json
        if ($progress.state -in @("starting", "running")) {
          return
        }
      } catch {
        # The progress file is replaced atomically; retry if it is temporarily unavailable.
      }
    }
    Start-Sleep -Seconds 1
  }
}

if (-not $RunDir) {
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $RunDir = "artifacts\full_release_grid4x1_judged_$stamp"
}

if (-not $env:DLD_JUDGE_API_KEY) {
  $env:DLD_JUDGE_API_KEY = [Environment]::GetEnvironmentVariable("DLD_JUDGE_API_KEY", "User")
}
if (-not $env:DLD_JUDGE_API_KEY) {
  throw "Missing DLD_JUDGE_API_KEY. Set the user environment variable before starting the full run."
}
if ($VlmGridLayout -notmatch '^\d+x\d+$') {
  throw "VlmGridLayout must use ROWSxCOLUMNS, for example 4x1."
}

$env:DLD_VLM_RETRY_ATTEMPTS = "$VlmRetryAttempts"
$env:DLD_VLM_RETRY_BACKOFF_SECONDS = "$VlmRetryBackoffSeconds"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$precomputeLog = Join-Path $RunDir "precompute.log"
$vlmLogPrefix = Join-Path $RunDir "vlm"
$judgeLogPrefix = Join-Path $RunDir "judge"
$progressPath = Join-Path $RunDir "release_progress.json"
$judgeOutput = Join-Path $RunDir "llm_adjudication_live.json"

Write-Host "Run directory: $RunDir"
Write-Host "Case root: $CaseRoot"
Write-Host "VLM grid layout: $VlmGridLayout"

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
$judgeArgs = @(
  "tools/watch_release_adjudication.py",
  "--release-progress", $progressPath,
  "--case-debug-root", (Join-Path $RunDir "case_debug"),
  "--output", $judgeOutput,
  "--base-url", $JudgeBaseUrl,
  "--model", $JudgeModel
)

$vlmProcess = $null
$judgeProcess = $null
try {
  Write-Host "Starting full release VLM..."
  $vlmProcess = Start-PythonLogged -Arguments $vlmArgs -LogPrefix $vlmLogPrefix
  Wait-ForReleaseStart -ProgressPath $progressPath -VlmProcess $vlmProcess

  Write-Host "Starting live LLM adjudication..."
  $judgeProcess = Start-PythonLogged -Arguments $judgeArgs -LogPrefix $judgeLogPrefix
  $vlmProcess.WaitForExit()
  Merge-ProcessLogs -LogPrefix $vlmLogPrefix
  if ($vlmProcess.ExitCode -ne 0) {
    throw "Release VLM failed. See $vlmLogPrefix.log"
  }

  $judgeProcess.WaitForExit()
  Merge-ProcessLogs -LogPrefix $judgeLogPrefix
  if ($judgeProcess.ExitCode -ne 0) {
    throw "LLM adjudication failed. See $judgeLogPrefix.log"
  }
} finally {
  if ($judgeProcess -and -not $judgeProcess.HasExited) {
    Stop-Process -Id $judgeProcess.Id -Force
  }
  if ($vlmProcess -and -not $vlmProcess.HasExited) {
    Stop-Process -Id $vlmProcess.Id -Force
  }
}

Write-Host "Done."
Write-Host "Release report: $(Join-Path $RunDir "release_report.json")"
Write-Host "Comparison: $(Join-Path $RunDir "release_comparison.json")"
Write-Host "Live adjudication: $judgeOutput"
