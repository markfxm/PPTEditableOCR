$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Install-Target($target, $packages) {
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    python -m pip install --upgrade --target $target @packages
}

Install-Target ".py310deps" @(
    "python-pptx",
    "pillow",
    "pypdfium2",
    "opencv-python-headless",
    "numpy"
)

Install-Target ".py310gui" @(
    "PySide6"
)

Install-Target ".py310iopaint" @(
    "iopaint",
    "paddleocr>=3.6.0",
    "paddlex[ocr]",
    "huggingface_hub==0.25.2",
    "paddlepaddle==3.2.0"
)

Write-Host "Local development dependencies are installed." -ForegroundColor Green
