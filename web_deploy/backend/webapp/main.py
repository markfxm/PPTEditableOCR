from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.core import OCRBox

from .storage import (
    box_to_dict,
    init_job,
    InvalidJobId,
    job_dir,
    load_project,
    read_json,
    safe_name,
    save_project,
    state_path,
    update_state,
)
from .tasks import export_job, process_upload


app = FastAPI(title="PPTtoEdit Web API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def queue() -> Queue:
    import redis
    from rq import Queue

    connection = redis.from_url(os.environ.get("REDIS_URL", "redis://redis:6379/0"))
    return Queue("ppttoedit", connection=connection, default_timeout=-1)


def sync_jobs_enabled() -> bool:
    return os.environ.get("WEB_SYNC_JOBS", "").lower() in {"1", "true", "yes", "on"}


class BoxPayload(BaseModel):
    text: str = ""
    score: float = 1.0
    bbox: list[int]
    erase_rect: list[int] | None = None
    enabled: bool = True
    manual: bool = False
    edited: bool = True
    rotation: int = 0


class SlidePayload(BaseModel):
    boxes: list[BoxPayload]
    remove_watermark: bool | None = None
    watermark_rect: list[int] | None = None


@app.exception_handler(InvalidJobId)
def invalid_job_id_handler(_request, _exc: InvalidJobId) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": "任务不存在"})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/jobs")
async def create_job(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> dict[str, Any]:
    filename = safe_name(file.filename or "upload")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".pptx"}:
        raise HTTPException(status_code=400, detail="只支持 PDF 或 PPTX 文件")

    job_id = uuid.uuid4().hex
    root = init_job(job_id, filename)
    upload_path = root / "uploads" / filename
    with upload_path.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            out.write(chunk)

    if sync_jobs_enabled():
        background_tasks.add_task(process_upload, job_id, str(upload_path))
    else:
        queue().enqueue_call(process_upload, args=(job_id, str(upload_path)), job_timeout=-1, result_ttl=3600)
    return {"id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    state = read_json(state_path(job_id), None)
    if not state:
        raise HTTPException(status_code=404, detail="任务不存在")
    return state


@app.get("/jobs/{job_id}/slides")
def get_slides(job_id: str) -> dict[str, Any]:
    try:
        project = load_project(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="项目还未准备好") from None
    return {
        "slideWidth": project.slide_width,
        "slideHeight": project.slide_height,
        "slides": [
            {
                "index": slide.index,
                "imageWidth": slide.image_width,
                "imageHeight": slide.image_height,
                "imageUrl": f"/api/jobs/{job_id}/slides/{slide.index}/image",
                "boxes": [box_to_dict(box) for box in slide.boxes],
                "watermarkRect": list(slide.watermark_rect) if slide.watermark_rect else None,
                "removeWatermark": slide.remove_watermark,
            }
            for slide in project.slides
        ],
    }


@app.get("/jobs/{job_id}/slides/{slide_index}/image")
def get_slide_image(job_id: str, slide_index: int) -> FileResponse:
    project = load_project(job_id)
    slide = next((item for item in project.slides if item.index == slide_index), None)
    if not slide or not slide.image_path.exists():
        raise HTTPException(status_code=404, detail="页面图片不存在")
    return FileResponse(slide.image_path)


@app.put("/jobs/{job_id}/slides/{slide_index}")
def update_slide(job_id: str, slide_index: int, payload: SlidePayload) -> dict[str, Any]:
    project = load_project(job_id)
    slide = next((item for item in project.slides if item.index == slide_index), None)
    if not slide:
        raise HTTPException(status_code=404, detail="页面不存在")

    slide.boxes = []
    for item in payload.boxes:
        bbox = tuple(int(value) for value in item.bbox)
        erase_rect = tuple(int(value) for value in (item.erase_rect or item.bbox))
        slide.boxes.append(
            OCRBox(
                text=item.text,
                score=float(item.score),
                bbox=bbox,  # type: ignore[arg-type]
                erase_rect=erase_rect,  # type: ignore[arg-type]
                enabled=bool(item.enabled),
                manual=bool(item.manual),
                edited=bool(item.edited),
                rotation=int(item.rotation),
            )
        )
    if payload.remove_watermark is not None:
        slide.remove_watermark = payload.remove_watermark
    if payload.watermark_rect is not None:
        slide.watermark_rect = tuple(int(value) for value in payload.watermark_rect)  # type: ignore[assignment]

    save_project(job_id, project)
    state = read_json(state_path(job_id), {})
    slides = [item for item in state.get("slides", []) if item.get("index") != slide.index]
    slides.append({"index": slide.index, "label": f"第{slide.index}页", "boxCount": len(slide.boxes), "status": "ok"})
    slides.sort(key=lambda item: item["index"])
    changes: dict[str, Any] = {"slides": slides}
    if state.get("status") != "done":
        changes.update(status="ready", phase="可检查识别框并导出")
    update_state(job_id, **changes)
    return {"ok": True, "boxCount": len(slide.boxes)}


@app.post("/jobs/{job_id}/export")
def start_export(job_id: str, background_tasks: BackgroundTasks) -> dict[str, Any]:
    if not state_path(job_id).exists():
        raise HTTPException(status_code=404, detail="任务不存在")
    if sync_jobs_enabled():
        background_tasks.add_task(export_job, job_id)
    else:
        queue().enqueue_call(export_job, args=(job_id,), job_timeout=-1, result_ttl=3600)
    update_state(job_id, status="queued_export", phase="等待导出", progress=0)
    return {"ok": True}


@app.get("/jobs/{job_id}/download")
def download(job_id: str) -> FileResponse:
    state = read_json(state_path(job_id), None)
    if not state or not state.get("outputPptx"):
        raise HTTPException(status_code=404, detail="导出文件还未生成")
    output = Path(state["outputPptx"])
    if not output.exists():
        raise HTTPException(status_code=404, detail="导出文件不存在")
    return FileResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=output.name,
    )
