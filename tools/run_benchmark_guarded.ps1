param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $BenchmarkArgs
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$runner = Join-Path $repoRoot "tools\run_benchmark_guarded.py"
& python $runner @BenchmarkArgs
exit $LASTEXITCODE
