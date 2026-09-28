"""Phase 8F-3a Step 1: a tiny offline HTTP server for the oriented-box vehicle labelling tool.

Stdlib only (``http.server``), binds to localhost only. Serves the labelling page and the
Maxar images to label, and accepts saves that are written straight to the DOTA-format
``labelTxt/<stem>.txt`` files the rest of the detector pipeline already knows how to read
(:func:`geoseek.detect.eval_io.load_dota_gt`) - no export/import step, no server-side
dependency beyond what geoseek already needs.

Nothing here is a "detector" or a "model" - it draws what the human clicks and writes it
back out unchanged.

    python scripts/serve_label_tool.py [--port 8765] [--set data/detect_eval/maxar_handlabeled_v1]
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

TOOL_HTML = Path(__file__).resolve().parent / "label_obb.html"


def make_handler(dataset_dir: Path):
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labelTxt"
    manifest_path = dataset_dir / "manifest.json"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter default logging
            sys.stderr.write(f"[label-tool] {self.address_string()} {fmt % args}\n")

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, obj) -> None:
            self._send(200, json.dumps(obj).encode("utf-8"), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            path = unquote(urlparse(self.path).path)
            if path in ("/", "/index.html"):
                self._send(200, TOOL_HTML.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/manifest":
                self._send_json(json.loads(manifest_path.read_text(encoding="utf-8")))
            elif path.startswith("/images/"):
                p = images_dir / Path(path[len("/images/"):]).name
                if not p.is_file():
                    self._send(404, b"not found", "text/plain")
                    return
                self._send(200, p.read_bytes(), "image/png")
            elif path.startswith("/api/labels/"):
                stem = Path(path[len("/api/labels/"):]).stem
                p = labels_dir / f"{stem}.txt"
                text = p.read_text(encoding="utf-8") if p.is_file() else ""
                self._send(200, text.encode("utf-8"), "text/plain; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            path = unquote(urlparse(self.path).path)
            if not path.startswith("/api/labels/"):
                self._send(404, b"not found", "text/plain")
                return
            stem = Path(path[len("/api/labels/"):]).stem
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            (labels_dir / f"{stem}.txt").write_text(body, encoding="utf-8")

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            n_instances = sum(1 for line in body.splitlines() if len(line.split()) >= 10)
            for img in manifest["images"]:
                if img["stem"] == stem:
                    img["status"] = "labeled"
                    img["n_instances"] = n_instances
            manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
            print(f"[label-tool] saved {stem}: {n_instances} instances", flush=True)
            self._send_json({"ok": True, "stem": stem, "n_instances": n_instances})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--set", default="data/detect_eval/maxar_handlabeled_v1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    dataset_dir = Path(args.set).resolve()
    if not (dataset_dir / "manifest.json").is_file():
        raise SystemExit(f"no manifest.json under {dataset_dir} - run scripts/build the label set first")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(dataset_dir))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"[label-tool] serving {dataset_dir} at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
