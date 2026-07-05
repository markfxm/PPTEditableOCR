from __future__ import annotations

import contextlib
import json
import os
import sys
import traceback
import warnings
from pathlib import Path

from .core import export_editable_ppt, ppt_project_from_data


def silence_dependency_info_logs() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        from loguru import logger
    except Exception:
        return

    logger.remove()
    logger.add(sys.stderr, level="WARNING")


def progress_printer(stream):
    def progress(message: str) -> None:
        stream.write(f"{message}\n")
        stream.flush()

    return progress


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: python -m app.export_worker <input.json>", file=sys.stderr)
        return 2

    input_path = Path(argv[0])
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        project = ppt_project_from_data(payload["project"])
        output_pptx = Path(payload["output_pptx"])
        enhance_images = bool(payload.get("enhance_images", True))

        silence_dependency_info_logs()
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull):
                export_editable_ppt(
                    project,
                    output_pptx,
                    progress=progress_printer(sys.__stdout__),
                    enhance_images=enhance_images,
                )
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
