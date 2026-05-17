from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import json
import hashlib
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

BASE = Path(__file__).resolve().parent.parent
DEPS_DIR = BASE / ".py310deps"
IOPAINT_DEPS_DIR = BASE / ".py310iopaint"
for deps in [IOPAINT_DEPS_DIR, DEPS_DIR]:
    if deps.exists() and str(deps) not in sys.path:
        sys.path.insert(0, str(deps))

import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


SKIP_TEXTS = {"NotebookLM"}
ProgressCB = Callable[[str], None] | None
PROGRESS_PREFIX = "__PPTTOEDIT_PROGRESS__|"
PAGE_READY_PREFIX = "__PPTTOEDIT_PAGE_READY__|"
CACHE_VERSION = 1
DEFAULT_PPT_WIDTH = Inches(13.333333)


def bundle_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return BASE


def bundled_models_root() -> Path | None:
    candidate = bundle_root() / "models"
    return candidate if candidate.exists() else None


def bundled_ocr_model_dir(model_name: str) -> Path | None:
    root = bundled_models_root()
    if not root:
        return None
    candidate = root / "paddlex" / "official_models" / model_name
    return candidate if candidate.exists() else None


def bundled_iopaint_model_root() -> Path | None:
    root = bundled_models_root()
    return root if root and (root / "torch" / "hub" / "checkpoints" / "big-lama.pt").exists() else None


@dataclass
class OCRBox:
    text: str
    score: float
    bbox: tuple[int, int, int, int]
    erase_rect: tuple[int, int, int, int]
    enabled: bool = True
    manual: bool = False
    edited: bool = False
    rotation: int = 0

    def set_erase_rect(self, rect: tuple[int, int, int, int]):
        left, top, right, bottom = rect
        self.erase_rect = (
            int(left),
            int(top),
            int(max(left + 1, right)),
            int(max(top + 1, bottom)),
        )

    def set_bbox_from_rect(self, rect: tuple[int, int, int, int]):
        left, top, right, bottom = rect
        self.bbox = (
            int(left),
            int(top),
            int(max(1, right - left)),
            int(max(1, bottom - top)),
        )

    def reset_from_bbox(self, pad_x: int, pad_y: int, image_width: int, image_height: int):
        x, y, w, h = self.bbox
        left = max(0, x - pad_x)
        top = max(0, y - pad_y)
        right = min(image_width - 1, x + w + pad_x)
        bottom = min(image_height - 1, y + h + pad_y)
        self.erase_rect = (left, top, right, bottom)


@dataclass
class PPTSlide:
    index: int
    image_name: str
    image_path: Path
    image_width: int
    image_height: int
    boxes: list[OCRBox] = field(default_factory=list)
    watermark_rect: tuple[int, int, int, int] | None = None
    remove_watermark: bool = True

    def reset_boxes(self, pad_x: int, pad_y: int):
        for box in self.boxes:
            box.reset_from_bbox(pad_x, pad_y, self.image_width, self.image_height)


@dataclass
class PPTProject:
    source_pptx: Path
    work_dir: Path
    images_dir: Path
    masks_dir: Path
    cleaned_dir: Path
    slides: list[PPTSlide]
    slide_width: int
    slide_height: int


def default_cache_path(source_pptx: Path) -> Path:
    return source_pptx.with_name(f"{source_pptx.stem}.ppttoedit.json")


def app_cache_path(source_pptx: Path) -> Path:
    cache_root = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "PPTEditableOCR" / "ocr_caches"
    digest = hashlib.sha1(str(source_pptx).encode("utf-8", errors="ignore")).hexdigest()[:16]
    return cache_root / f"{source_pptx.stem}-{digest}.ppttoedit.json"


def cache_path_candidates(source_pptx: Path) -> list[Path]:
    sidecar = default_cache_path(source_pptx)
    fallback = app_cache_path(source_pptx)
    return [sidecar] if sidecar == fallback else [sidecar, fallback]


def _rect_to_list(rect: tuple[int, int, int, int]) -> list[int]:
    return [int(value) for value in rect]


def _rect_from_data(value, fallback: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return fallback
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError):
        return fallback


def save_project_cache(project: PPTProject, cache_path: Path | None = None, progress: ProgressCB = None) -> Path:
    data = {
        "version": CACHE_VERSION,
        "source_name": project.source_pptx.name,
        "slide_width": int(project.slide_width),
        "slide_height": int(project.slide_height),
        "slides": [
            {
                "index": int(slide.index),
                "image_name": slide.image_name,
                "image_width": int(slide.image_width),
                "image_height": int(slide.image_height),
                "watermark_rect": _rect_to_list(slide.watermark_rect) if slide.watermark_rect else None,
                "remove_watermark": bool(slide.remove_watermark),
                "boxes": [
                    {
                        "text": box.text,
                        "score": float(box.score),
                        "bbox": _rect_to_list(box.bbox),
                        "erase_rect": _rect_to_list(box.erase_rect),
                        "enabled": bool(box.enabled),
                        "manual": bool(box.manual),
                        "edited": bool(box.edited),
                        "rotation": int(box.rotation),
                    }
                    for box in slide.boxes
                ],
            }
            for slide in project.slides
        ],
    }
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    candidates = [cache_path] if cache_path else cache_path_candidates(project.source_pptx)
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_text(payload, encoding="utf-8")
        except OSError as exc:
            last_error = exc
            continue
        _log(progress, f"识别框缓存已保存：{candidate}")
        return candidate
    raise RuntimeError(f"识别框缓存保存失败：{last_error}")


def load_project_cache(project: PPTProject, cache_path: Path | None = None, progress: ProgressCB = None) -> bool:
    candidates = [cache_path] if cache_path else cache_path_candidates(project.source_pptx)
    cache_path = next((path for path in candidates if path.exists()), None)
    if cache_path is None:
        return False

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _log(progress, f"识别框缓存读取失败，已忽略：{cache_path} ({exc})")
        return False
    cached_slides = data.get("slides")
    if not isinstance(cached_slides, list) or len(cached_slides) != len(project.slides):
        _log(progress, f"识别框缓存页数不匹配，已忽略：{cache_path}")
        return False

    for slide, cached in zip(project.slides, cached_slides):
        if (
            int(cached.get("index", -1)) != slide.index
            or int(cached.get("image_width", -1)) != slide.image_width
            or int(cached.get("image_height", -1)) != slide.image_height
        ):
            _log(progress, f"识别框缓存页面尺寸不匹配，已忽略：{cache_path}")
            return False

    for slide, cached in zip(project.slides, cached_slides):
        slide.remove_watermark = bool(cached.get("remove_watermark", slide.remove_watermark))
        slide.watermark_rect = _rect_from_data(cached.get("watermark_rect"), slide.watermark_rect) if cached.get("watermark_rect") else slide.watermark_rect
        slide.boxes = []
        for item in cached.get("boxes", []):
            if not isinstance(item, dict):
                continue
            bbox = _rect_from_data(item.get("bbox"), (0, 0, 1, 1))
            erase_rect = _rect_from_data(item.get("erase_rect"), bbox)
            slide.boxes.append(
                OCRBox(
                    text=str(item.get("text") or ""),
                    score=float(item.get("score", 1.0)),
                    bbox=bbox,
                    erase_rect=erase_rect,
                    enabled=bool(item.get("enabled", True)),
                    manual=bool(item.get("manual", False)),
                    edited=bool(item.get("edited", False)),
                    rotation=int(item.get("rotation", 0)),
                )
            )

    _log(progress, f"已加载识别框缓存，跳过 OCR：{cache_path}")
    for slide in project.slides:
        _page_ready(progress, slide)
    return True


def box_to_rect(poly) -> tuple[int, int, int, int]:
    pts = np.array(poly, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    return x, y, w, h


def default_expand_rect(
    x: int, y: int, w: int, h: int, width: int, height: int
) -> tuple[int, int, int, int]:
    pad_x = max(8, int(w * 0.05))
    pad_y = max(6, int(h * 0.18))
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(width - 1, x + w + pad_x)
    bottom = min(height - 1, y + h + pad_y)
    return left, top, right, bottom


def _log(progress: ProgressCB, message: str):
    if progress:
        progress(message)


def _progress(progress: ProgressCB, percent: int, message: str):
    percent = max(0, min(100, int(percent)))
    _log(progress, f"{PROGRESS_PREFIX}{percent}|{message}")


def _page_ready(progress: ProgressCB, slide: PPTSlide, status: str = "ok"):
    _log(progress, f"{PAGE_READY_PREFIX}{slide.index}|{len(slide.boxes)}|{status}")


def unique_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"无法生成不覆盖已有文件的输出路径：{path}")


def default_pdf_pptx_path(source_pdf: Path) -> Path:
    return unique_output_path(source_pdf.with_name(f"{source_pdf.stem}-from-pdf.pptx"))


def convert_pdf_to_pptx(source_pdf: Path, output_pptx: Path | None = None, progress: ProgressCB = None) -> Path:
    import pypdfium2

    source_pdf = source_pdf.expanduser().resolve()
    if output_pptx is None:
        output_pptx = default_pdf_pptx_path(source_pdf)
    else:
        output_pptx = Path(output_pptx).expanduser().resolve()

    _log(progress, f"开始转换 PDF：{source_pdf}")
    pdf = pypdfium2.PdfDocument(str(source_pdf))
    page_count = len(pdf)
    if page_count <= 0:
        raise RuntimeError(f"PDF 没有可转换页面：{source_pdf}")

    first_page = pdf[0]
    first_width, first_height = first_page.get_size()
    first_page.close()
    if first_width <= 0 or first_height <= 0:
        raise RuntimeError(f"PDF 首页尺寸无效：{source_pdf}")

    out = Presentation()
    out.slide_width = int(DEFAULT_PPT_WIDTH)
    out.slide_height = int(DEFAULT_PPT_WIDTH * first_height / first_width)
    while out.slides:
        rel_id = out.slides._sldIdLst[0].rId
        out.part.drop_rel(rel_id)
        del out.slides._sldIdLst[0]

    blank = out.slide_layouts[6]
    _progress(progress, 0, f"PDF 转 PPT：0/{page_count}")
    with tempfile.TemporaryDirectory(prefix="ppttoedit_pdf_") as temp_dir:
        temp_root = Path(temp_dir)
        for page_index in range(page_count):
            page = pdf[page_index]
            image_path = temp_root / f"page_{page_index + 1:04d}.png"
            bitmap = page.render(scale=2)
            image = bitmap.to_pil().convert("RGB")
            image.save(image_path)
            bitmap.close()
            page.close()

            slide = out.slides.add_slide(blank)
            slide.shapes.add_picture(
                str(image_path),
                0,
                0,
                width=out.slide_width,
                height=out.slide_height,
            )
            done = page_index + 1
            _progress(progress, int(done * 100 / page_count), f"PDF 转 PPT：{done}/{page_count}")
            _log(progress, f"第 {done} 页已加入 PPT")

        output_pptx.parent.mkdir(parents=True, exist_ok=True)
        out.save(str(output_pptx))

    pdf.close()
    _log(progress, f"PDF 转 PPT 完成：{output_pptx}")
    return output_pptx


def extract_slide_images(source_pptx: Path, images_dir: Path, progress: ProgressCB = None):
    images_dir.mkdir(parents=True, exist_ok=True)
    src = Presentation(str(source_pptx))
    extracted: list[PPTSlide] = []

    for index, slide in enumerate(src.slides, start=1):
        pic = next((shape for shape in slide.shapes if getattr(shape, "image", None)), None)
        if pic is None:
            continue
        image_name = f"slide_{index:02d}.png"
        out_path = images_dir / image_name
        out_path.write_bytes(pic.image.blob)
        with Image.open(out_path) as image:
            width, height = image.size
        watermark_rect = (
            max(0, width - 190),
            max(0, height - 90),
            width - 1,
            height - 1,
        )
        extracted.append(
            PPTSlide(
                index=index,
                image_name=image_name,
                image_path=out_path,
                image_width=width,
                image_height=height,
                watermark_rect=watermark_rect,
            )
        )
        _log(progress, f"提取第 {index} 页图片")
    return src, extracted


def run_ocr(slides: list[PPTSlide], progress: ProgressCB = None):
    det_dir = bundled_ocr_model_dir("PP-OCRv5_server_det")
    rec_dir = bundled_ocr_model_dir("PP-OCRv5_server_rec")

    def create_ocr():
        return PaddleOCR(
            lang="ch",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_detection_model_dir=str(det_dir) if det_dir else None,
            text_recognition_model_dir=str(rec_dir) if rec_dir else None,
        )

    ocr = create_ocr()
    for slide in slides:
        try:
            page = ocr.predict(str(slide.image_path))[0]
        except Exception as exc:
            _log(progress, f"第 {slide.index} 页 OCR 失败，已跳过该页，可稍后手动新增框：{exc}")
            _log(progress, traceback.format_exc())
            slide.boxes = []
            _page_ready(progress, slide, status="failed")
            try:
                ocr = create_ocr()
            except Exception as reset_exc:
                _log(progress, f"OCR 引擎重置失败，后续页面可能继续失败：{reset_exc}")
            continue
        boxes: list[OCRBox] = []
        for poly, text, score in zip(page["dt_polys"], page["rec_texts"], page["rec_scores"]):
            text = (text or "").strip()
            if not text:
                continue
            x, y, w, h = box_to_rect(poly)
            erase_rect = default_expand_rect(x, y, w, h, slide.image_width, slide.image_height)
            boxes.append(
                OCRBox(
                    text=text,
                    score=float(score),
                    bbox=(x, y, w, h),
                    erase_rect=erase_rect,
                    enabled=text not in SKIP_TEXTS,
                )
            )
        slide.boxes = boxes
        _page_ready(progress, slide)
        _log(progress, f"第 {slide.index} 页 OCR 完成，共 {len(boxes)} 个框")


def prepare_project(
    source_pptx: Path, work_dir: Path | None = None, progress: ProgressCB = None
) -> PPTProject:
    source_pptx = source_pptx.expanduser().resolve()
    if work_dir is None:
        work_dir = BASE / "_gui_workspace" / source_pptx.stem
    if work_dir.exists():
        shutil.rmtree(work_dir)
    images_dir = work_dir / "images"
    masks_dir = work_dir / "masks"
    cleaned_dir = work_dir / "cleaned"
    work_dir.mkdir(parents=True, exist_ok=True)

    src, slides = extract_slide_images(source_pptx, images_dir, progress)
    project = PPTProject(
        source_pptx=source_pptx,
        work_dir=work_dir,
        images_dir=images_dir,
        masks_dir=masks_dir,
        cleaned_dir=cleaned_dir,
        slides=slides,
        slide_width=src.slide_width,
        slide_height=src.slide_height,
    )
    if not load_project_cache(project, progress=progress):
        run_ocr(slides, progress)
    return project


def build_masks(project: PPTProject, progress: ProgressCB = None):
    project.masks_dir.mkdir(parents=True, exist_ok=True)
    for slide in project.slides:
        mask = np.zeros((slide.image_height, slide.image_width), dtype=np.uint8)
        for box in slide.boxes:
            if not box.enabled:
                continue
            left, top, right, bottom = box.erase_rect
            cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)
        if slide.remove_watermark and slide.watermark_rect:
            left, top, right, bottom = slide.watermark_rect
            cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)
        mask_path = project.masks_dir / slide.image_name
        Image.fromarray(mask).save(mask_path)
        if not mask_path.exists():
            raise RuntimeError(f"擦除蒙版写入失败：{mask_path}")
        _log(progress, f"第 {slide.index} 页擦除蒙版已生成")


def os_environ_with_pythonpath():
    env = dict(os.environ)
    pythonpath_parts = [str(Path(IOPAINT_DEPS_DIR).resolve()), str(Path(DEPS_DIR).resolve())]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ";".join(pythonpath_parts)
    bundled_model_root = bundled_iopaint_model_root()
    if bundled_model_root:
        env["XDG_CACHE_HOME"] = str(bundled_model_root)
        env["U2NET_HOME"] = str(bundled_model_root)
    return env


def run_iopaint(images_dir: Path, masks_dir: Path, cleaned_dir: Path, progress: ProgressCB = None):
    if cleaned_dir.exists():
        shutil.rmtree(cleaned_dir)
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    _log(progress, "开始用 IOPaint 擦除底图文字")
    _progress(progress, 0, "IOPaint 擦除处理中")
    cmd = [
        sys.executable,
        "-m",
        "iopaint",
        "run",
        "--model",
        "lama",
        "--device",
        "cpu",
        "--image",
        str(images_dir),
        "--mask",
        str(masks_dir),
        "--output",
        str(cleaned_dir),
    ]
    env = os_environ_with_pythonpath()
    env["PYTHONUNBUFFERED"] = "1"
    total_images = max(1, len([path for path in masks_dir.iterdir() if path.is_file()]))
    last_done = 0
    last_percent = -1
    percent_re = re.compile(r"(\d{1,3})%\s+(\d+)\s*/\s*(\d+)")
    count_re = re.compile(r"(\d+)\s*/\s*(\d+)")
    ansi_re = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        cwd=str(BASE),
    )
    assert process.stdout is not None
    buffer = ""

    def emit_iopaint_progress(done: int, total: int):
        nonlocal last_done, last_percent
        total = max(1, total)
        done = max(0, min(done, total))
        percent = int(done * 100 / total)
        if done != last_done or percent != last_percent:
            last_done = done
            last_percent = percent
            _progress(progress, percent, f"IOPaint 擦除处理中：{done}/{total}")

    def handle_output(text: str):
        nonlocal last_done
        text = ansi_re.sub("", text).strip()
        if not text:
            return
        match = percent_re.search(text)
        if match:
            percent = max(0, min(100, int(match.group(1))))
            done = int(match.group(2))
            total = int(match.group(3))
            last_done = done
            _progress(progress, percent, f"IOPaint 擦除处理中：{done}/{total}")
            return
        match = count_re.search(text)
        if match and "Batch processing" in text:
            emit_iopaint_progress(int(match.group(1)), int(match.group(2)))
            return
        if "Run crop strategy" in text:
            emit_iopaint_progress(last_done + 1, total_images)

    while True:
        char = process.stdout.read(1)
        if char == "" and process.poll() is not None:
            break
        if not char:
            continue
        if char in {"\r", "\n"}:
            handle_output(buffer)
            buffer = ""
        else:
            buffer += char
    handle_output(buffer)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)
    _progress(progress, 100, f"IOPaint 擦除完成：{total_images}/{total_images}")
    _log(progress, "IOPaint 擦除完成")


def run_iopaint(images_dir: Path, masks_dir: Path, cleaned_dir: Path, progress: ProgressCB = None):
    if cleaned_dir.exists():
        shutil.rmtree(cleaned_dir)
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    _log(progress, "开始用 IOPaint 擦除底图文字")

    env = os_environ_with_pythonpath()
    for key in ("XDG_CACHE_HOME", "U2NET_HOME", "PYTHONPATH"):
        if key in env:
            os.environ[key] = env[key]

    from iopaint.download import cli_download_model, scan_models
    from iopaint.helper import pil_to_bytes
    from iopaint.model.utils import torch_gc
    from iopaint.model_manager import ModelManager
    from iopaint.schema import Device, InpaintRequest

    image_paths = {
        path.stem: path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    }
    mask_paths = {
        path.stem: path
        for path in masks_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    }
    if not image_paths:
        raise RuntimeError(f"没有找到待处理图片：{images_dir}")
    if not mask_paths:
        raise RuntimeError(f"没有找到擦除蒙版：{masks_dir}")

    total_images = len(image_paths)
    _progress(progress, 0, f"IOPaint 擦除处理中：0/{total_images}")

    if "lama" not in [model.name for model in scan_models()]:
        _log(progress, "本机未找到 IOPaint lama 模型，开始准备模型")
        cli_download_model("lama")

    model_manager = ModelManager(name="lama", device=Device.cpu)
    inpaint_request = InpaintRequest()
    first_mask = next(iter(mask_paths.values()))

    for done, (stem, image_path) in enumerate(sorted(image_paths.items()), start=1):
        mask_path = mask_paths.get(stem, first_mask)
        with Image.open(image_path) as source_image:
            infos = source_image.info
            image = np.array(source_image.convert("RGB"))
        with Image.open(mask_path) as source_mask:
            mask = np.array(source_mask.convert("L"))

        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask[mask >= 127] = 255
        mask[mask < 127] = 0

        result = model_manager(image, mask, inpaint_request)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        output_path = cleaned_dir / f"{stem}.png"
        output_path.write_bytes(pil_to_bytes(Image.fromarray(result), "png", 100, infos))
        torch_gc()
        _progress(progress, int(done * 100 / total_images), f"IOPaint 擦除处理中：{done}/{total_images}")

    missing = [name for name in image_paths if not (cleaned_dir / f"{name}.png").exists()]
    if missing:
        raise RuntimeError(f"IOPaint 未生成以下页面：{', '.join(sorted(missing))}")

    _log(progress, "IOPaint 擦除完成")


def cleaned_image_path(cleaned_dir: Path, name: str):
    nested = cleaned_dir / name / name.replace("_clean", "")
    if nested.exists():
        return nested
    direct = cleaned_dir / name
    if direct.exists():
        return direct
    fallback = cleaned_dir / name.replace("_clean", "")
    if fallback.exists():
        return fallback
    raise FileNotFoundError(name)


def sample_text_color(image: Image.Image, rect):
    region = image.crop(rect).convert("RGB")
    arr = np.asarray(region)
    if arr.size == 0:
        return RGBColor(0, 0, 0)
    flat = arr.reshape(-1, 3)
    brightness = flat.sum(axis=1)
    sample = flat[np.argsort(brightness)[: max(1, len(flat) // 8)]]
    rgb = sample.mean(axis=0).astype(int)
    return RGBColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def should_rotate_text(box: OCRBox) -> bool:
    if box.rotation in {90, 270, -90}:
        return True
    x, y, w, h = box.bbox
    return h > w * 1.45 and len(box.text.strip()) > 1


def add_textbox(slide, color_image, box: OCRBox, x_scale: float, y_scale: float):
    text = box.text
    if text in SKIP_TEXTS or (box.score < 0.45 and not box.manual and not box.edited) or not box.enabled:
        return False

    x, y, w, h = box.bbox
    left = int(x * x_scale)
    top = int(y * y_scale)
    width = int(w * x_scale)
    height = int(h * y_scale)
    rotate_text = should_rotate_text(box)
    font_axis = height

    if rotate_text:
        center_x = left + width / 2
        center_y = top + height / 2
        shape_left = int(center_x - height / 2)
        shape_top = int(center_y - width / 2)
        shape = slide.shapes.add_textbox(shape_left, shape_top, height, width)
        shape.rotation = box.rotation if box.rotation in {90, 270} else 270
        font_axis = width
    else:
        shape = slide.shapes.add_textbox(left, top, width, height)
    shape.fill.background()
    shape.line.fill.background()

    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = False
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0

    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = text
    run.font.name = "Microsoft YaHei"
    run.font.size = Pt(max(8, font_axis * 0.72 / 12700))
    run.font.color.rgb = sample_text_color(color_image, box.erase_rect)
    return True


def rebuild_ppt(project: PPTProject, output_pptx: Path, progress: ProgressCB = None):
    out = Presentation()
    out.slide_width = project.slide_width
    out.slide_height = project.slide_height
    while out.slides:
        rel_id = out.slides._sldIdLst[0].rId
        out.part.drop_rel(rel_id)
        del out.slides._sldIdLst[0]

    blank = out.slide_layouts[6]
    for slide_data in project.slides:
        cleaned_path = cleaned_image_path(project.cleaned_dir, slide_data.image_name)
        color_image = Image.open(slide_data.image_path).convert("RGB")
        cleaned_image = Image.open(cleaned_path).convert("RGB")
        dst = out.slides.add_slide(blank)
        dst.shapes.add_picture(
            str(cleaned_path),
            0,
            0,
            width=out.slide_width,
            height=out.slide_height,
        )
        x_scale = out.slide_width / cleaned_image.width
        y_scale = out.slide_height / cleaned_image.height
        count = 0
        for box in slide_data.boxes:
            if add_textbox(dst, color_image, box, x_scale, y_scale):
                count += 1
        _log(progress, f"第 {slide_data.index} 页已重建 {count} 个可编辑文本框")
    output_pptx.parent.mkdir(parents=True, exist_ok=True)
    out.save(str(output_pptx))
    _log(progress, f"已导出：{output_pptx}")


def export_editable_ppt(project: PPTProject, output_pptx: Path, progress: ProgressCB = None):
    build_masks(project, progress)
    run_iopaint(project.images_dir, project.masks_dir, project.cleaned_dir, progress)
    rebuild_ppt(project, output_pptx, progress)
