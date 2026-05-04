from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from app.core import export_editable_ppt, prepare_project


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python make_editable_ppt.py <source.pptx> [output.pptx]")

    source_pptx = Path(sys.argv[1]).expanduser().resolve()
    output_pptx = (
        Path(sys.argv[2]).expanduser().resolve()
        if len(sys.argv) >= 3
        else source_pptx.with_name(f"{source_pptx.stem}-editable-clean.pptx")
    )

    project = prepare_project(source_pptx, progress=print)
    export_editable_ppt(project, output_pptx, progress=print)


if __name__ == "__main__":
    main()
