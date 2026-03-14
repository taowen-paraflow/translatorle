# Run qwen35 C++ GPU-only inference
# Usage: .\qwen35\cpp_gpu\run.ps1 [-Prompt "text"] [-MaxTokens 50]

param(
    [string]$Prompt = "The capital of France is",
    [int]$MaxTokens = 50,
    [string]$ModelDir = "",
    [switch]$Timing
)

$root = "C:\Apps\translatorle"
$exe = "$root\qwen35\cpp_gpu\build\Release\qwen35_gpu.exe"
$tokLib = "$root\.venv\Lib\site-packages\openvino_tokenizers\lib\openvino_tokenizers.dll"

# Default model: PARO INT4 quantized single IR
if ($ModelDir -eq "") {
    $ModelDir = "$root\models\qwen35\Qwen3.5-0.8B-paro-ov-int4sym"
}

# Add OpenVINO libs to PATH
$env:PATH = "$root\.venv\Lib\site-packages\openvino\libs;" + $env:PATH

if (-not (Test-Path $exe)) {
    Write-Host "ERROR: $exe not found. Run build.ps1 first." -ForegroundColor Red
    exit 1
}

$args_ = @(
    "--model-dir", $ModelDir,
    "--prompt", $Prompt,
    "--max-tokens", $MaxTokens,
    "--tokenizers-lib", $tokLib
)
if ($Timing) { $args_ += "--timing" }

& $exe @args_
