from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from .core import PPTSlide, ocr_box_to_data, predict_ocr_page


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: python -m app.ocr_page_worker <input.json> <output.json>", file=sys.stderr)
        return 2

    input_path = Path(argv[0])
    output_path = Path(argv[1])
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        slide_data = payload["slide"]
        slide = PPTSlide(
            index=int(slide_data["index"]),
            image_name=str(slide_data["image_name"]),
            image_path=Path(slide_data["image_path"]),
            image_width=int(slide_data["image_width"]),
            image_height=int(slide_data["image_height"]),
        )
        boxes = predict_ocr_page(
            slide,
            ocr_backend=str(payload.get("ocr_backend") or "local"),
            ocr_token=payload.get("ocr_token"),
        )
        output_path.write_text(
            json.dumps({"boxes": [ocr_box_to_data(box) for box in boxes]}, ensure_ascii=False),
            encoding="utf-8",
        )
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
