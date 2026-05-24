from __future__ import annotations

import argparse
import http.server
import subprocess
import socketserver
import threading
import time
import webbrowser
from functools import partial
from pathlib import Path
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the experiment viewer over localhost.")
    parser.add_argument(
        "--root",
        default="artifacts/deepar_m5_experiments",
        help="Experiment root containing run_* directories and the generated experiment_viewer.html.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser automatically.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    viewer_path = root / "experiment_viewer.html"
    if not viewer_path.exists():
        builder = Path("scripts/build_experiment_html_viewer.py")
        if not builder.exists():
            raise FileNotFoundError(f"{viewer_path} not found and {builder} is missing.")
        subprocess.run([sys.executable, str(builder), "--root", str(root)], check=True)

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format: str, *values) -> None:  # noqa: A003
            pass

    handler = partial(QuietHandler, directory=str(root))
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.ThreadingTCPServer((args.host, args.port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    url = f"http://{args.host}:{args.port}/experiment_viewer.html"
    print(f"Serving {root} at {url}")
    if not args.no_open:
        webbrowser.open(url)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    main()
