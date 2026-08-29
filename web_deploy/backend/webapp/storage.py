from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from app.core import OCRBox, PPTProject, PPTSlide, VisualAsset


DATA_DIR = Path(os.environ.get("DATA_DIR", "/data")).resolve()
JOBS_DIR = DATA_DIR / "jobs"
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class InvalidJobId(ValueError):
    pass


def ensure_data_dirs() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)


def job_dir(job_id: str) -> Path:
    if not JOB_ID_RE.fullmatch(job_id):
        raise InvalidJobId("Invalid job id")
    path = (JOBS_DIR / job_id).resolve()
    if not path.is_relative_to(JOBS_DIR.resolve()):
        raise InvalidJobId("Invalid job id")
    return path


def state_path(job_id: str) -> Path:
    return job_dir(job_id) / "state.json"


def project_path(job_id: str) -> Path:
    return job_dir(job_id) / "project.json"


def safe_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    return cleaned or "upload"


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def init_job(job_id: str, filename: str) -> Path:
    root = job_dir(job_id)
    if root.exists():
        shutil.rmtree(root)
    (root / "uploads").mkdir(parents=True)
    (root / "work").mkdir()
    (root / "outputs").mkdir()
    write_json(
        state_path(job_id),
        {
            "id": job_id,
            "filename": filename,
            "status": "queued",
            "phase": "等待处理",
            "progress": 0,
            "messages": [],
            "slides": [],
            "sourcePptx": None,
            "outputPptx": None,
            "error": None,
        },
    )
    return root


def update_state(job_id: str, **changes: Any) -> dict[str, Any]:
    state = read_json(state_path(job_id), {})
    state.update(changes)
    write_json(state_path(job_id), state)
    return state


def append_message(job_id: str, message: str) -> None:
    state = read_json(state_path(job_id), {})
    messages = list(state.get("messages") or [])
    messages.append(message)
    state["messages"] = messages[-300:]
    write_json(state_path(job_id), state)


def box_to_dict(box: OCRBox) -> dict[str, Any]:
    return {
        "text": box.text,
        "score": float(box.score),
        "bbox": [int(value) for value in box.bbox],
        "erase_rect": [int(value) for value in box.erase_rect],
        "enabled": bool(box.enabled),
        "manual": bool(box.manual),
        "edited": bool(box.edited),
        "rotation": int(box.rotation),
        "line_height": int(box.line_height) if box.line_height is not None else None,
        "text_regions": [[[int(x), int(y)] for x, y in region] for region in box.text_regions],
        "mask_mode": box.mask_mode,
        "mask_reason": box.mask_reason,
    }


def visual_asset_to_dict(asset: VisualAsset) -> dict[str, Any]:
    return {
        "asset_id": asset.asset_id, "bbox": [int(value) for value in asset.bbox],
        "enabled": bool(asset.enabled), "source": asset.source, "status": asset.status,
        "layer": asset.layer, "image_path": str(asset.image_path) if asset.image_path else None,
        "mask_path": str(asset.mask_path) if asset.mask_path else None,
    }


def slide_to_dict(slide: PPTSlide) -> dict[str, Any]:
    return {
        "index": int(slide.index),
        "image_name": slide.image_name,
        "image_path": str(slide.image_path),
        "image_width": int(slide.image_width),
        "image_height": int(slide.image_height),
        "watermark_rect": list(slide.watermark_rect) if slide.watermark_rect else None,
        "remove_watermark": bool(slide.remove_watermark),
        "visual_assets": [visual_asset_to_dict(asset) for asset in slide.visual_assets],
        "boxes": [box_to_dict(box) for box in slide.boxes],
    }


def project_to_dict(project: PPTProject) -> dict[str, Any]:
    return {
        "source_pptx": str(project.source_pptx),
        "work_dir": str(project.work_dir),
        "images_dir": str(project.images_dir),
        "masks_dir": str(project.masks_dir),
        "cleaned_dir": str(project.cleaned_dir),
        "assets_dir": str(project.assets_dir) if project.assets_dir else None,
        "slide_width": int(project.slide_width),
        "slide_height": int(project.slide_height),
        "slides": [slide_to_dict(slide) for slide in project.slides],
    }


def dict_to_box(data: dict[str, Any]) -> OCRBox:
    bbox = tuple(int(value) for value in data.get("bbox", [0, 0, 1, 1]))
    erase_rect = tuple(int(value) for value in data.get("erase_rect", bbox))
    return OCRBox(
        text=str(data.get("text") or ""),
        score=float(data.get("score", 1.0)),
        bbox=bbox,  # type: ignore[arg-type]
        erase_rect=erase_rect,  # type: ignore[arg-type]
        enabled=bool(data.get("enabled", True)),
        manual=bool(data.get("manual", False)),
        edited=bool(data.get("edited", False)),
        rotation=int(data.get("rotation", 0)),
        line_height=(int(data["line_height"]) if data.get("line_height") is not None else None),
        text_regions=tuple(
            tuple((int(point[0]), int(point[1])) for point in region)
            for region in data.get("text_regions", [])
            if isinstance(region, list) and len(region) >= 3
        ),
        mask_mode=str(data.get("mask_mode") or "pending"),
        mask_reason=str(data["mask_reason"]) if data.get("mask_reason") else None,
    )


def dict_to_visual_asset(data: dict[str, Any]) -> VisualAsset:
    return VisualAsset(
        asset_id=str(data.get("asset_id") or "visual-asset"),
        bbox=tuple(int(value) for value in data.get("bbox", [0, 0, 1, 1])),  # type: ignore[arg-type]
        enabled=bool(data.get("enabled", True)), source=str(data.get("source") or "opencv"),
        status=str(data.get("status") or "rule_candidate"), layer=str(data.get("layer") or "below_text"),
        image_path=Path(data["image_path"]) if data.get("image_path") else None,
        mask_path=Path(data["mask_path"]) if data.get("mask_path") else None,
    )


def dict_to_slide(data: dict[str, Any]) -> PPTSlide:
    return PPTSlide(
        index=int(data["index"]),
        image_name=str(data["image_name"]),
        image_path=Path(data["image_path"]),
        image_width=int(data["image_width"]),
        image_height=int(data["image_height"]),
        boxes=[dict_to_box(item) for item in data.get("boxes", [])],
        watermark_rect=tuple(data["watermark_rect"]) if data.get("watermark_rect") else None,  # type: ignore[arg-type]
        remove_watermark=bool(data.get("remove_watermark", True)),
        visual_assets=[dict_to_visual_asset(item) for item in data.get("visual_assets", []) if isinstance(item, dict)],
    )


def dict_to_project(data: dict[str, Any]) -> PPTProject:
    return PPTProject(
        source_pptx=Path(data["source_pptx"]),
        work_dir=Path(data["work_dir"]),
        images_dir=Path(data["images_dir"]),
        masks_dir=Path(data["masks_dir"]),
        cleaned_dir=Path(data["cleaned_dir"]),
        assets_dir=Path(data["assets_dir"]) if data.get("assets_dir") else None,
        slides=[dict_to_slide(item) for item in data.get("slides", [])],
        slide_width=int(data["slide_width"]),
        slide_height=int(data["slide_height"]),
    )


def load_project(job_id: str) -> PPTProject:
    data = read_json(project_path(job_id), None)
    if not data:
        raise FileNotFoundError(f"Project is not ready: {job_id}")
    return dict_to_project(data)


def save_project(job_id: str, project: PPTProject) -> None:
    write_json(project_path(job_id), project_to_dict(project))
