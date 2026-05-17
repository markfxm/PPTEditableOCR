from __future__ import annotations

import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    root = Path(__file__).resolve().parent / "frontend"
    handler = functools.partial(QuietHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", 5173), handler)
    print("Frontend static server running at http://127.0.0.1:5173", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
