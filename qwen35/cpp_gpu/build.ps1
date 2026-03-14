# Build qwen35 C++ GPU-only inference
# Usage: .\qwen35\cpp_gpu\build.ps1

$cmake = "C:\Apps\translatorle\.venv\Scripts\cmake.exe"
$srcDir = "C:\Apps\translatorle\qwen35\cpp_gpu"
$buildDir = "$srcDir\build"

if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

# Configure
if (-not (Test-Path "$buildDir\CMakeCache.txt")) {
    Write-Host "Configuring CMake..."
    & $cmake -S $srcDir -B $buildDir -G "Visual Studio 17 2022" -A x64
}

# Build Release
Write-Host "Building Release..."
& $cmake --build $buildDir --config Release

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nBuild successful: $buildDir\Release\qwen35_gpu.exe"
} else {
    Write-Host "`nBuild FAILED" -ForegroundColor Red
}
