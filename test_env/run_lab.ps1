$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host '[RUN] Drone tracker lab'
python .\src\main.py
