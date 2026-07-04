from __future__ import annotations

import sys

from app.gui import main
from app.ocr_page_worker import main as ocr_page_worker_main


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--ocr-page-worker":
        raise SystemExit(ocr_page_worker_main(sys.argv[2:]))
    raise SystemExit(main())
