param(
    [ValidateSet("batch", "forward")]
    [string]$Mode = "batch"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ImageName = "sku-sales-forecasting:latest"

$FeaturesPath = Join-Path $ProjectRoot "data\processed\sales_model_features.csv"
$FinalModelPath = Join-Path $ProjectRoot "models\random_forest_final.joblib"
$ProcessedDir = Join-Path $ProjectRoot "data\processed"
$ModelDir = Join-Path $ProjectRoot "models"
$ReportsDir = Join-Path $ProjectRoot "reports"

if (-not (Test-Path -LiteralPath $FeaturesPath -PathType Leaf)) {
    throw "Feature file not found: $FeaturesPath"
}

if ($Mode -eq "batch" -and
    -not (Test-Path -LiteralPath $FinalModelPath -PathType Leaf)) {
    throw "Final test model not found: $FinalModelPath"
}

foreach ($Directory in @($ModelDir, $ReportsDir)) {
    if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
        New-Item -ItemType Directory -Path $Directory -Force |
            Out-Null
    }
}

Write-Host "Building Docker image: $ImageName"

& docker build --tag $ImageName $ProjectRoot

if ($LASTEXITCODE -ne 0) {
    throw "Docker image build failed."
}

$DockerArgs = @(
    "run",
    "--rm",
    "--mount",
    "type=bind,source=$ProcessedDir,target=/app/data/processed,readonly"
)

if ($Mode -eq "batch") {
    $DockerArgs += @(
        "--mount",
        "type=bind,source=$ModelDir,target=/app/models,readonly"
    )
}
else {
    $DockerArgs += @(
        "--mount",
        "type=bind,source=$ModelDir,target=/app/models"
    )
}

$DockerArgs += @(
    "--mount",
    "type=bind,source=$ReportsDir,target=/app/reports",
    $ImageName
)

if ($Mode -eq "forward") {
    $DockerArgs += @(
        "python",
        "src/forecast_next_week.py"
    )
}

Write-Host "Running Docker prediction mode: $Mode"

& docker @DockerArgs

if ($LASTEXITCODE -ne 0) {
    throw "Docker prediction failed in mode: $Mode"
}

if ($Mode -eq "forward") {
    $OutputPath = Join-Path $ReportsDir "next_week_forecast.csv"
}
else {
    $OutputPath = Join-Path $ReportsDir "batch_predictions_test.csv"
}

Write-Host "Prediction completed successfully."
Write-Host "Output: $OutputPath"