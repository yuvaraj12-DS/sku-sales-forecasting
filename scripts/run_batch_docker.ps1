
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ImageName = "sku-sales-forecasting:latest"

$FeaturesPath = Join-Path $ProjectRoot "data\processed\sales_model_features.csv"
$ModelPath = Join-Path $ProjectRoot "models\random_forest_final.joblib"
$ProcessedDir = Join-Path $ProjectRoot "data\processed"
$ModelDir = Join-Path $ProjectRoot "models"
$ReportsDir = Join-Path $ProjectRoot "reports"

if (-not (Test-Path -LiteralPath $FeaturesPath -PathType Leaf)) {
    throw "Feature file not found: $FeaturesPath"
}

if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) {
    throw "Trained model not found: $ModelPath"
}

if (-not (Test-Path -LiteralPath $ReportsDir -PathType Container)) {
    New-Item -ItemType Directory -Path $ReportsDir -Force |
        Out-Null
}

Write-Host "Building Docker image: $ImageName"

& docker build --tag $ImageName $ProjectRoot

if ($LASTEXITCODE -ne 0) {
    throw "Docker image build failed."
}

$DockerArgs = @(
    "run",
    "--rm",
    "--mount", "type=bind,source=$ProcessedDir,target=/app/data/processed,readonly",
    "--mount", "type=bind,source=$ModelDir,target=/app/models,readonly",
    "--mount", "type=bind,source=$ReportsDir,target=/app/reports",
    $ImageName
)

Write-Host "Running batch prediction in Docker..."

& docker @DockerArgs

if ($LASTEXITCODE -ne 0) {
    throw "Docker batch prediction failed."
}

Write-Host "Batch prediction completed successfully."
Write-Host "Output: $(Join-Path $ReportsDir 'batch_predictions_test.csv')"
