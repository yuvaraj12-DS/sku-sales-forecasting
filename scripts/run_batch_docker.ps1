
param(
    [ValidateSet("batch", "forward", "monitor")]
    [string]$Mode = "batch"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ImageName = "sku-sales-forecasting:latest"

$FeaturesPath = Join-Path $ProjectRoot "data\processed\sales_model_features.csv"
$FinalModelPath = Join-Path $ProjectRoot "models\random_forest_final.joblib"
$ForecastPath = Join-Path $ProjectRoot "reports\next_week_forecast.csv"

$ProcessedDir = Join-Path $ProjectRoot "data\processed"
$ModelDir = Join-Path $ProjectRoot "models"
$ReportsDir = Join-Path $ProjectRoot "reports"

# Validate required inputs.
if (-not (Test-Path -LiteralPath $FeaturesPath -PathType Leaf)) {
    throw "Feature file not found: $FeaturesPath"
}

if (
    $Mode -eq "batch" -and
    -not (Test-Path -LiteralPath $FinalModelPath -PathType Leaf)
) {
    throw "Final test model not found: $FinalModelPath"
}

if (
    $Mode -eq "monitor" -and
    -not (Test-Path -LiteralPath $ForecastPath -PathType Leaf)
) {
    throw "Forward forecast file not found: $ForecastPath. Run forward mode first."
}

# Ensure output directories exist.
foreach ($Directory in @($ModelDir, $ReportsDir)) {
    if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
        New-Item -ItemType Directory -Path $Directory -Force |
            Out-Null
    }
}

# Build the image.
Write-Host "Building Docker image: $ImageName"

& docker build --tag $ImageName $ProjectRoot

if ($LASTEXITCODE -ne 0) {
    throw "Docker image build failed."
}

# Set up the container command and mounts.
$DockerArgs = @(
    "run",
    "--rm",
    "--mount",
    "type=bind,source=$ProcessedDir,target=/app/data/processed,readonly"
)

# Batch mode reads the existing final model.
# Forward mode trains and saves a separate model.
# Monitor mode does not need a model mount.
if ($Mode -eq "batch") {
    $DockerArgs += @(
        "--mount",
        "type=bind,source=$ModelDir,target=/app/models,readonly"
    )
}
elseif ($Mode -eq "forward") {
    $DockerArgs += @(
        "--mount",
        "type=bind,source=$ModelDir,target=/app/models"
    )
}

# All modes may write reports.
$DockerArgs += @(
    "--mount",
    "type=bind,source=$ReportsDir,target=/app/reports",
    $ImageName
)

# Select the script to run inside the container.
if ($Mode -eq "forward") {
    $DockerArgs += @(
        "python",
        "src/forecast_next_week.py"
    )
}
elseif ($Mode -eq "monitor") {
    $DockerArgs += @(
        "python",
        "src/monitor_forecast.py"
    )
}

# Execute the container.
Write-Host "Running Docker mode: $Mode"

& docker @DockerArgs

if ($LASTEXITCODE -ne 0) {
    throw "Docker execution failed in mode: $Mode"
}

# Report the expected output file.
switch ($Mode) {
    "forward" {
        $OutputPath = Join-Path $ReportsDir "next_week_forecast.csv"
    }
    "monitor" {
        $OutputPath = Join-Path $ReportsDir "forecast_monitoring_summary.csv"
    }
    default {
        $OutputPath = Join-Path $ReportsDir "batch_predictions_test.csv"
    }
}

Write-Host ""
Write-Host "Prediction workflow completed successfully."
Write-Host "Mode: $Mode"
Write-Host "Output: $OutputPath"
