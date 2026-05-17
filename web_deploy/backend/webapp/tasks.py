from __future__ import annotations

import os
import traceback
from pathlib import Path

from app.core import (
    PAGE_READY_PREFIX,
    PROGRESS_PREFIX,
    convert_pdf_to_pptx,
    export_editable_ppt,
    prepare_project,
    save_project_cache,
)

from .storage import (
    append_message,
    job_dir,
    load_project,
    read_json,
    save_project,
    state_path,
    update_state,
)


def _progress(job_id: str, message: str) -> None:
    if message.startswith(PROGRESS_PREFIX):
        _, percent, text = message.split("|", 2)
        update_state(job_id, progress=int(percent), phase=text)
        append_message(job_id, text)
        return
    if message.startswith(PAGE_READY_PREFIX):
        _, index, box_count, status = message.split("|", 3)
        state = read_json(state_path(job_id), {})
        slides = [item for item in state.get("slides", []) if item.get("index") != int(index)]
        slides.append(
            {
                "index": int(index),
                "label": f"第{int(index)}页" if status == "ok" else f"第{int(index)}页 - OCR失败",
                "boxCount": int(box_count),
                "status": status,
            }
        )
        slides.sort(key=lambda item: item["index"])
        update_state(job_id, slides=slides)
        return
    append_message(job_id, message)


def process_upload(job_id: str, upload_path: str) -> None:
    try:
        update_state(job_id, status="running", phase="准备文件", progress=0, error=None)
        source = Path(upload_path)
        root = job_dir(job_id)
        source_pptx = source
        if source.suffix.lower() == ".pdf":
            update_state(job_id, phase="PDF 转 PPT", progress=0)
            source_pptx = root / "work" / f"{source.stem}-from-pdf.pptx"
            convert_pdf_to_pptx(source, source_pptx, progress=lambda msg: _progress(job_id, msg))
        elif source.suffix.lower() not in {".pptx", ".ppt"}:
            raise RuntimeError("只支持 PDF、PPTX 或 PPT 文件")

        update_state(job_id, sourcePptx=str(source_pptx), phase="OCR 识别", progress=0)
        project = prepare_project(
            source_pptx,
            work_dir=root / "work" / "project",
            progress=lambda msg: _progress(job_id, msg),
        )
        save_project(job_id, project)
        update_state(job_id, status="ready", phase="可检查识别框并导出", progress=100)
    except Exception as exc:
        update_state(job_id, status="failed", phase="失败", error=str(exc))
        append_message(job_id, traceback.format_exc())
        raise


def export_job(job_id: str) -> None:
    try:
        update_state(job_id, status="exporting", phase="导出可编辑 PPT", progress=0, error=None)
        project = load_project(job_id)
        save_project_cache(project, progress=lambda msg: _progress(job_id, msg))
        output = job_dir(job_id) / "outputs" / f"{project.source_pptx.stem}-editable-clean.pptx"
        export_editable_ppt(project, output, progress=lambda msg: _progress(job_id, msg))
        update_state(job_id, status="done", phase="导出完成", progress=100, outputPptx=str(output))
    except Exception as exc:
        update_state(job_id, status="failed", phase="导出失败", error=str(exc))
        append_message(job_id, traceback.format_exc())
        raise


def cleanup_old_jobs(max_age_hours: int = 24) -> int:
    import shutil
    import time

    data_dir = Path(os.environ.get("DATA_DIR", "/data")).resolve() / "jobs"
    if not data_dir.exists():
        return 0
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for path in data_dir.iterdir():
        if path.is_dir() and path.stat().st_mtime < cutoff:
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    return removed
