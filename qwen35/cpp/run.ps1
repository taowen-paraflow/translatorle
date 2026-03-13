# Run qwen35 C++ hybrid GPU+NPU inference
# Usage: .\qwen35\cpp\run.ps1 [-Prompt "text"] [-MaxTokens 50] [-ChunkSize 16]
#
# Examples:
#   .\qwen35\cpp\run.ps1
#   .\qwen35\cpp\run.ps1 -Prompt "Hello world"
#   .\qwen35\cpp\run.ps1 -Prompt "Explain gravity" -MaxTokens 200

param(
    [string]$Prompt = "The capital of France is",
    [int]$MaxTokens = 50,
    [int]$ChunkSize = 16,
    [int]$AttnPastSeq = 256,
    [switch]$Timing
)

$root = "C:\Apps\translatorle"
$exe = "$root\qwen35\cpp\build\Release\qwen35_infer.exe"
# Use quantized model by default (INT4 attn + INT8 GDN + INT4 head = 4.1GB, ~20% faster)
$modelDir = "$root\models\qwen35\Qwen3.5-0.8B-hybrid-attn-int4sym-gdn-int8sym-head-int4sym"
$tokLib = "$root\.venv\Lib\site-packages\openvino_tokenizers\lib\openvino_tokenizers.dll"

# Add OpenVINO libs to PATH
$env:PATH = "$root\.venv\Lib\site-packages\openvino\libs;" + $env:PATH

if (-not (Test-Path $exe)) {
    Write-Host "ERROR: $exe not found. Run build.ps1 first." -ForegroundColor Red
    exit 1
}

$args_ = @(
    "--model-dir", $modelDir,
    "--prompt", $Prompt,
    "--max-tokens", $MaxTokens,
    "--prefill-chunk-size", $ChunkSize,
    "--attn-past-seq", $AttnPastSeq,
    "--tokenizers-lib", $tokLib
)
if ($Timing) { $args_ += "--timing" }

& $exe @args_
