$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$env:SAM2_BUILD_CUDA = "0"

function Install-Target($target, $packages) {
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    python -m pip install --upgrade --target $target @packages
}

Install-Target ".py310deps" @(
    "python-pptx",
    "pillow",
    "pypdfium2",
    "opencv-python-headless",
    "numpy",
    "pymatting",
    "openai"
)

Install-Target ".py310gui" @(
    "PySide6"
)

Install-Target ".py310iopaint" @(
    "iopaint",
    "paddleocr>=3.6.0",
    "paddlex[ocr]",
    "huggingface_hub==0.25.2",
    "paddlepaddle==3.2.0",
    "hydra-core>=1.3.2",
    "iopath>=0.1.10",
    "PyYAML==6.0.2"
)

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "NVIDIA GPU detected; installing CUDA PyTorch for SAM 2.1." -ForegroundColor Cyan
    python -m pip install --upgrade --target .py310iopaint `
        --index-url "https://download.pytorch.org/whl/cu130" `
        "torch==2.11.0+cu130" `
        "torchvision==0.26.0+cu130"
}

$env:PYTHONPATH = "$root\.py310iopaint;$root\.py310deps"
python -m pip install --upgrade --target .py310iopaint --no-deps --no-build-isolation `
    "git+https://github.com/facebookresearch/sam2.git"

Write-Host "Local development dependencies are installed." -ForegroundColor Green
