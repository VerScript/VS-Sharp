# stop-training.ps1
# Creates a stop signal file for the training process

$stopFile = Join-Path $PSScriptRoot "stop_training.txt"
New-Item -Path $stopFile -ItemType File -Force | Out-Null
Write-Host "Sent stop signal to the training process. File created at: $stopFile"
