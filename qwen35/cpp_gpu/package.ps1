# Package qwen35 GPU-only inference into a distributable zip
# Usage: .\qwen35\cpp_gpu\package.ps1 [-ModelDir path] [-OutputName name]

param(
    [string]$ModelDir = "",
    [string]$OutputName = "qwen35-gpu-paro"
)

$root = "C:\Apps\translatorle"
$exe = "$root\qwen35\cpp_gpu\build\Release\qwen35_gpu.exe"
$ovLibDir = "$root\.venv\Lib\site-packages\openvino\libs"
$tokLibDir = "$root\.venv\Lib\site-packages\openvino_tokenizers\lib"

if ($ModelDir -eq "") {
    $ModelDir = "$root\models\qwen35\Qwen3.5-0.8B-paro-ov-int4sym"
}

if (-not (Test-Path $exe)) {
    Write-Host "ERROR: $exe not found. Run build.ps1 first." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $ModelDir)) {
    Write-Host "ERROR: Model dir not found: $ModelDir" -ForegroundColor Red
    exit 1
}

# Create staging directory
$stageDir = "$root\dist\$OutputName"
if (Test-Path $stageDir) {
    Remove-Item -Recurse -Force $stageDir
}
New-Item -ItemType Directory -Path "$stageDir\model" | Out-Null

Write-Host "=== Packaging $OutputName ==="

# 1. Copy executable
Copy-Item $exe "$stageDir\qwen35_gpu.exe"
Write-Host "  Copied qwen35_gpu.exe"

# 2. Copy OpenVINO runtime DLLs
$ovDlls = @(
    "openvino.dll",
    "openvino_c.dll",
    "openvino_intel_gpu_plugin.dll",
    "openvino_auto_plugin.dll",
    "openvino_auto_batch_plugin.dll",
    "openvino_ir_frontend.dll",
    "plugins.xml"
)

foreach ($dll in $ovDlls) {
    $src = "$ovLibDir\$dll"
    if (Test-Path $src) {
        Copy-Item $src "$stageDir\$dll"
        Write-Host "  Copied $dll"
    }
}

# Copy TBB DLLs (OpenVINO runtime dependency, skip .lib debug files)
$tbbDlls = Get-ChildItem "$ovLibDir\tbb*.dll" -ErrorAction SilentlyContinue
foreach ($dll in $tbbDlls) {
    Copy-Item $dll.FullName "$stageDir\$($dll.Name)"
    Write-Host "  Copied $($dll.Name)"
}

# 3. Copy openvino_tokenizers extension
$tokDll = "$tokLibDir\openvino_tokenizers.dll"
if (Test-Path $tokDll) {
    Copy-Item $tokDll "$stageDir\openvino_tokenizers.dll"
    Write-Host "  Copied openvino_tokenizers.dll"
}

# 4. Copy model files
$modelFiles = Get-ChildItem $ModelDir -File
foreach ($f in $modelFiles) {
    # Skip cache directory contents
    if ($f.Directory.Name -eq "cache") { continue }
    Copy-Item $f.FullName "$stageDir\model\$($f.Name)"
    $sizeMB = [math]::Round($f.Length / 1MB, 1)
    Write-Host "  Copied model/$($f.Name) (${sizeMB} MB)"
}

# 5. Create run.bat launcher
$batContent = @"
@echo off
cd /d "%~dp0"
qwen35_gpu.exe --model-dir model --tokenizers-lib openvino_tokenizers.dll --prompt "%~1" --max-tokens 100
pause
"@
Set-Content -Path "$stageDir\run.bat" -Value $batContent -Encoding ASCII
Write-Host "  Created run.bat"

# 6. Create zip
$zipPath = "$root\dist\$OutputName.zip"
if (Test-Path $zipPath) {
    Remove-Item $zipPath
}
Compress-Archive -Path "$stageDir\*" -DestinationPath $zipPath
$zipSizeMB = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)

Write-Host "`n=== Package created ==="
Write-Host "  Directory: $stageDir"
Write-Host "  Archive:   $zipPath ($zipSizeMB MB)"

# List package contents with sizes
Write-Host "`n=== Package contents ==="
$totalSize = 0
Get-ChildItem $stageDir -Recurse -File | ForEach-Object {
    $relPath = $_.FullName.Replace("$stageDir\", "")
    $sizeMB = [math]::Round($_.Length / 1MB, 1)
    $totalSize += $_.Length
    Write-Host "  $relPath  ($sizeMB MB)"
}
$totalMB = [math]::Round($totalSize / 1MB, 1)
Write-Host "  ---"
Write-Host "  Total: $totalMB MB (uncompressed)"
