from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from app.core import OCRBox, PPTProject, PPTSlide


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
        "boxes": [box_to_dict(box) for box in slide.boxes],
    }


def project_to_dict(project: PPTProject) -> dict[str, Any]:
    return {
        "source_pptx": str(project.source_pptx),
        "work_dir": str(project.work_dir),
        "images_dir": str(project.images_dir),
        "masks_dir": str(project.masks_dir),
        "cleaned_dir": str(project.cleaned_dir),
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
    )


def dict_to_project(data: dict[str, Any]) -> PPTProject:
    return PPTProject(
        source_pptx=Path(data["source_pptx"]),
        work_dir=Path(data["work_dir"]),
        images_dir=Path(data["images_dir"]),
        masks_dir=Path(data["masks_dir"]),
        cleaned_dir=Path(data["cleaned_dir"]),
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
