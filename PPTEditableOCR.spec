# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata

project_root = Path.cwd()
pathex = [
    str(project_root),
    str(project_root / ".py310deps"),
    str(project_root / ".py310gui"),
    str(project_root / ".py310iopaint"),
]

datas = []
binaries = []
hiddenimports = []

excluded_path_parts = {
    "__pycache__",
    "tests",
    "testing",
    "test",
    "docs",
    "doc",
    "examples",
    "example",
    "include",
    "glue",
}

excluded_suffixes = {
    ".a",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".lib",
    ".pxd",
    ".pyx",
    ".rst",
}

excluded_prefixes = {
    "PySide6/qml",
}


def should_exclude(path):
    normalized = str(path).replace("\\", "/")
    lower = normalized.lower()
    parts = set(lower.split("/"))

    if parts & excluded_path_parts:
        return True
    if any(lower.startswith(prefix.lower()) for prefix in excluded_prefixes):
        return True
    if Path(lower).suffix in excluded_suffixes:
        return True

    return False


def filter_entries(entries):
    return [
        (source, target)
        for source, target in entries
        if not should_exclude(source) and not should_exclude(target)
    ]

packages = [
    "PySide6",
    "paddleocr",
    "paddlex",
    "paddle",
    "cv2",
    "numpy",
    "PIL",
    "pptx",
    "lxml",
    "iopaint",
    "torch",
    "torchvision",
    "transformers",
    "huggingface_hub",
    "safetensors",
    "scipy",
    "skimage",
    "yaml",
]

ocr_dependency_packages = {
    "beautifulsoup4": "bs4",
    "ftfy": "ftfy",
    "imagesize": "imagesize",
    "latex2mathml": "latex2mathml",
    "openpyxl": "openpyxl",
    "premailer": "premailer",
    "pyclipper": "pyclipper",
    "pypdfium2": "pypdfium2",
    "python-bidi": "bidi",
    "scikit-learn": "sklearn",
    "sentencepiece": "sentencepiece",
    "shapely": "shapely",
    "tiktoken": "tiktoken",
    "tokenizers": "tokenizers",
}

packages += list(ocr_dependency_packages.values())

for package in packages:
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package)
    datas += filter_entries(pkg_datas)
    binaries += filter_entries(pkg_binaries)

hiddenimports += [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "paddleocr._api_client",
    "iopaint.download",
    "iopaint.helper",
    "iopaint.model_manager",
    "iopaint.model.utils",
    "iopaint.schema",
]

for distribution in ocr_dependency_packages:
    datas += copy_metadata(distribution)

optional_datas = [
    (
        Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "big-lama.pt",
        "models/torch/hub/checkpoints",
    ),
    (
        Path.home() / ".paddlex" / "official_models" / "PP-OCRv5_server_det",
        "models/paddlex/official_models/PP-OCRv5_server_det",
    ),
    (
        Path.home() / ".paddlex" / "official_models" / "PP-OCRv5_server_rec",
        "models/paddlex/official_models/PP-OCRv5_server_rec",
    ),
]

for source_path, target_path in optional_datas:
    if source_path.is_file():
        datas.append((str(source_path), target_path))
    elif source_path.is_dir():
        datas.append((str(source_path), target_path))

a = Analysis(
    ["run_gui.py"],
    pathex=pathex,
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PPTEditableOCR",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="PPTEditableOCR",
)
