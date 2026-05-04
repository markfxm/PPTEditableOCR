param(
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$releaseRoot = Join-Path $root "release_artifacts"
$buildDir = Join-Path $releaseRoot "build"
$distDir = Join-Path $releaseRoot "dist"
$installerDir = Join-Path $releaseRoot "installer_output"

function Write-Step($text) {
    Write-Host ""
    Write-Host "==> $text" -ForegroundColor Cyan
}

Write-Step "Cleaning old build outputs"
Remove-Item -Recurse -Force $buildDir, $distDir, $installerDir -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $releaseRoot | Out-Null

$pyDeps = Join-Path $root ".py310deps"
$pyGui = Join-Path $root ".py310gui"
$pyIopaint = Join-Path $root ".py310iopaint"
$env:PYTHONPATH = "$pyGui;$pyIopaint;$pyDeps"

if (-not (Test-Path (Join-Path $root ".py310build"))) {
    New-Item -ItemType Directory -Force -Path (Join-Path $root ".py310build") | Out-Null
}

Write-Step "Ensuring PyInstaller is available"
python -m pip install --target .py310build pyinstaller | Out-Host
$env:PYTHONPATH = "$root\.py310build;$env:PYTHONPATH"

Write-Step "Building PyInstaller app bundle"
python -m PyInstaller --clean --noconfirm --workpath $buildDir --distpath $distDir PPTEditableOCR.spec

if ($SkipInstaller) {
    Write-Step "Skipping installer generation"
    exit 0
}

$iscc = (Get-Command ISCC -ErrorAction SilentlyContinue).Source
if (-not $iscc) {
    $candidates = @(
        "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        "C:\Program Files\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            $iscc = $candidate
            break
        }
    }
}
if (-not $iscc) {
    throw "ISCC not found. Install Inno Setup 6 first, or rerun with -SkipInstaller."
}

Write-Step "Building setup.exe with Inno Setup"
& $iscc installer.iss

Write-Step "Done"
Write-Host "App bundle: $distDir\PPTEditableOCR" -ForegroundColor Green
Write-Host "Installer: $installerDir\PPTEditableOCR-Setup.exe" -ForegroundColor Green
