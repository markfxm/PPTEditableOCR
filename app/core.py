from __future__ import annotations

import os
import shutil
import subprocess
import sys
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
from pptx.util import Pt


SKIP_TEXTS = {"NotebookLM"}
ProgressCB = Callable[[str], None] | None


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

    def set_erase_rect(self, rect: tuple[int, int, int, int]):
        left, top, right, bottom = rect
        self.erase_rect = (
            int(left),
            int(top),
            int(max(left + 1, right)),
            int(max(top + 1, bottom)),
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
    ocr = PaddleOCR(
        lang="ch",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_detection_model_dir=str(det_dir) if det_dir else None,
        text_recognition_model_dir=str(rec_dir) if rec_dir else None,
    )
    for slide in slides:
        page = ocr.predict(str(slide.image_path))[0]
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
    run_ocr(slides, progress)
    return PPTProject(
        source_pptx=source_pptx,
        work_dir=work_dir,
        images_dir=images_dir,
        masks_dir=masks_dir,
        cleaned_dir=cleaned_dir,
        slides=slides,
        slide_width=src.slide_width,
        slide_height=src.slide_height,
    )


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
        cv2.imwrite(str(project.masks_dir / slide.image_name), mask)
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
    subprocess.run(cmd, check=True, env=os_environ_with_pythonpath(), cwd=str(BASE))
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


def add_textbox(slide, color_image, box: OCRBox, x_scale: float, y_scale: float):
    text = box.text
    if text in SKIP_TEXTS or box.score < 0.45 or not box.enabled:
        return False

    x, y, w, h = box.bbox
    left = int(x * x_scale)
    top = int(y * y_scale)
    width = int(w * x_scale)
    height = int(h * y_scale)

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
    run.font.size = Pt(max(8, height * 0.72 / 12700))
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
