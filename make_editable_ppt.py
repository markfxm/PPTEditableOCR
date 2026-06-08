from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from app.core import convert_pdf_to_pptx, export_editable_ppt, prepare_project


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python make_editable_ppt.py <source.pptx|source.pdf> [output.pptx]")

    source = Path(sys.argv[1]).expanduser().resolve()
    source_pptx = source
    if source.suffix.lower() == ".pdf":
        source_pptx = convert_pdf_to_pptx(source, progress=print)
    elif source.suffix.lower() != ".pptx":
        raise SystemExit("source must be a .pptx or .pdf file")

    output_pptx = (
        Path(sys.argv[2]).expanduser().resolve()
        if len(sys.argv) >= 3
        else source_pptx.with_name(f"{source_pptx.stem}-editable-clean.pptx")
    )

    project = prepare_project(source_pptx, progress=print)
    export_editable_ppt(project, output_pptx, progress=print)


if __name__ == "__main__":
    main()
