# Runs the 30-request benchmark against Mistral's free tier. Asks for your key (typing is hidden); the key is kept
# only in this PowerShell window and is never written to a file.
param([string]$Model = "mistral-small-latest")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$secure = Read-Host "Paste your Mistral API key and press Enter (nothing will show as you paste)" -AsSecureString
$env:LLM_API_KEY = [System.Net.NetworkCredential]::new("", $secure).Password.Trim()
$env:LLM_BASE_URL = "https://api.mistral.ai/v1"
$env:LLM_PRICE_IN_PER_MTOK = "0"; $env:LLM_PRICE_OUT_PER_MTOK = "0"   # free tier
$env:LLM_PROVIDER = "openai_compat"; $env:PYTHONPATH = "src"
python scripts/12_llm_benchmark.py --models "api:$Model" --out llm_benchmark_mistral --sleep 5
$code = $LASTEXITCODE
Remove-Item Env:LLM_API_KEY
if ($code -eq 0) { Write-Host "Done. Results are in artifacts\llm_benchmark_mistral\summary.csv" } else { Write-Host "The run stopped. Nothing was saved. Copy the STOP line above and send it to Claude." }
