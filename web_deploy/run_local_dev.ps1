$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $root
$backend = Join-Path $root "backend"
$frontend = Join-Path $root "frontend"
$venv = Join-Path $root ".webvenv"
$python = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $python)) {
    python -m venv $venv
}

& $python -m pip install --upgrade pip
& $python -m pip install fastapi "uvicorn[standard]" python-multipart redis rq

$env:DATA_DIR = Join-Path $root "data"
$env:WEB_SYNC_JOBS = "1"
$env:PYTHONPATH = @(
    (Join-Path $backend "ppttoedit_core"),
    (Join-Path $projectRoot ".py310deps"),
    (Join-Path $projectRoot ".py310iopaint"),
    $env:PYTHONPATH
) -join ";"

Write-Host ""
Write-Host "PPTtoEdit Web local dev" -ForegroundColor Green
Write-Host "Backend:  http://127.0.0.1:8000"
Write-Host "Frontend: http://127.0.0.1:5173"
Write-Host ""
Write-Host "Keep this window open. Press Ctrl+C to stop both servers." -ForegroundColor Yellow
Write-Host ""

$backendJob = Start-Job -Name "ppttoedit-backend" -ScriptBlock {
    param($python, $backend, $dataDir, $pythonPath)
    $env:DATA_DIR = $dataDir
    $env:WEB_SYNC_JOBS = "1"
    $env:PYTHONPATH = $pythonPath
    Set-Location $backend
    & $python -m uvicorn webapp.main:app --host 127.0.0.1 --port 8000
} -ArgumentList $python, $backend, $env:DATA_DIR, $env:PYTHONPATH

$frontendJob = Start-Job -Name "ppttoedit-frontend" -ScriptBlock {
    param($python, $root)
    Set-Location $root
    & $python frontend_server.py
} -ArgumentList $python, $root

try {
    Start-Sleep -Seconds 2
    Start-Process "http://127.0.0.1:5173"
    while ($true) {
        Receive-Job -Job $backendJob, $frontendJob -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
        $failed = @($backendJob, $frontendJob) | Where-Object { $_.State -in @("Failed", "Stopped", "Completed") }
        if ($failed.Count -gt 0) {
            Receive-Job -Job $backendJob, $frontendJob -ErrorAction SilentlyContinue
            throw "A local dev server stopped unexpectedly."
        }
    }
}
finally {
    Stop-Job -Job $backendJob, $frontendJob -ErrorAction SilentlyContinue
    Remove-Job -Job $backendJob, $frontendJob -Force -ErrorAction SilentlyContinue
}
