from __future__ import annotations

import sys


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--ocr-page-worker":
        from app.ocr_page_worker import main as ocr_page_worker_main

        raise SystemExit(ocr_page_worker_main(sys.argv[2:]))
    from app.gui import main

    raise SystemExit(main())
