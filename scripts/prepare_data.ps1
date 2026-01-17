param(
  [string]$ZipPath = "",
  [string]$OutDir = ""
)

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

if ([string]::IsNullOrWhiteSpace($ZipPath)) {
  $ZipPath = (Join-Path $root 'data.zip')
}
if ([string]::IsNullOrWhiteSpace($OutDir)) {
  $OutDir = (Join-Path $root 'baseline_icpr_2026\data')
}

$ZipPath = [System.IO.Path]::GetFullPath($ZipPath)
$OutDir = [System.IO.Path]::GetFullPath($OutDir)

Write-Output "ZipPath: $ZipPath"
Write-Output "OutDir : $OutDir"

if (-not (Test-Path $ZipPath)) {
  throw "data.zip not found at: $ZipPath"
}

if (-not (Test-Path $OutDir)) {
  New-Item -ItemType Directory -Path $OutDir | Out-Null
}

$trainDir = Join-Path $OutDir 'train'
if (Test-Path $trainDir) {
  Write-Output "Train dir already exists: $trainDir"
  Write-Output "Skipping extraction. Delete it if you want a clean re-extract."
  exit 0
}

Write-Output "Extracting... this may take a while (many files)."
Expand-Archive -Path $ZipPath -DestinationPath $OutDir -Force

if (-not (Test-Path $trainDir)) {
  throw "Extraction finished, but train dir not found at: $trainDir"
}

Write-Output "Done. Extracted dataset to: $trainDir"
