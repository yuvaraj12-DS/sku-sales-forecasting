
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Pipeline = Join-Path $ProjectRoot "src\run_forecast_pipeline.py"
$ReportsDir = Join-Path $ProjectRoot "reports"
$LogPath = Join-Path $ReportsDir "pipeline_runner.log"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment not found: $Python"
}

if (-not (Test-Path -LiteralPath $Pipeline -PathType Leaf)) {
    throw "Pipeline script not found: $Pipeline"
}

New-Item -ItemType Directory -Path $ReportsDir -Force | Out-Null

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Pipeline started." |
    Tee-Object -FilePath $LogPath -Append

& $Python $Pipeline *>&1 |
    Tee-Object -FilePath $LogPath -Append

$ExitCode = $LASTEXITCODE

if ($ExitCode -ne 0) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Pipeline failed with exit code $ExitCode." |
        Tee-Object -FilePath $LogPath -Append
    exit $ExitCode
}

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Pipeline completed successfully." |
    Tee-Object -FilePath $LogPath -Append

exit 0
