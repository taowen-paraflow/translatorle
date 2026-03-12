# Run Python hybrid GPU+NPU inference for comparison
# Usage: .\qwen35\cpp\run-python.ps1 [-Prompt "text"] [-MaxTokens 50]

param(
    [string]$Prompt = "The capital of France is",
    [int]$MaxTokens = 50,
    [int]$ChunkSize = 16
)

$root = "C:\Apps\translatorle"
$uv = "C:\Users\taowen\.local\bin\uv.exe"

cd $root
& $uv run python -m qwen35.inference_hybrid `
    --prompt $Prompt `
    --device HYBRID `
    --no-attn-stateful `
    --prefill-chunk-size $ChunkSize `
    --max-tokens $MaxTokens
