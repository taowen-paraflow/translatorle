# Clear model cache (needed after changing compilation properties)
# Usage: .\qwen35\cpp\clear-cache.ps1

$cacheDir = "C:\Apps\translatorle\models\qwen35\Qwen3.5-0.8B-hybrid\cache"

if (Test-Path $cacheDir) {
    Remove-Item -Path $cacheDir -Recurse -Force
    Write-Host "Cleared: $cacheDir"
} else {
    Write-Host "No cache to clear: $cacheDir"
}
