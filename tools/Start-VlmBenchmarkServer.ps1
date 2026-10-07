param([int]$Port = 11435)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$runtimePath = Join-Path $repoRoot '.local\vlm'
New-Item -ItemType Directory -Path $runtimePath -Force | Out-Null
$env:OLLAMA_HOST = "127.0.0.1:$Port"
$env:OLLAMA_MODELS = Join-Path $runtimePath 'ollama-models'
$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_MAX_LOADED_MODELS = '1'
$env:OLLAMA_CONTEXT_LENGTH = '8192'
$env:OLLAMA_NO_CLOUD = '1'
$ollamaExe = (Get-Command ollama -ErrorAction Stop).Source
Start-Process -FilePath $ollamaExe -ArgumentList 'serve' -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtimePath 'server.stdout.log') -RedirectStandardError (Join-Path $runtimePath 'server.stderr.log')
Write-Output "Local VLM service started on 127.0.0.1:$Port. Model weights: $env:OLLAMA_MODELS"
