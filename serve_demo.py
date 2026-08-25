"""
Zero-dependency demo server for RetryBrain.

The full API lives in backend/main.py (FastAPI) and is the real deal. But that
needs `pip install`. THIS server uses only the Python standard library, so anyone
can see the live dashboard with a single command and no installs:

    python serve_demo.py            # -> http://127.0.0.1:8000/
    python serve_demo.py 8080       # custom port

It serves the same read endpoints the dashboard uses (/metrics, /results,
/audit/<id>, /health) off the last batch run, running the batch first if needed.
"""

import os
import re
import sys
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from backend import store
from backend.runner import run_batch

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "frontend", "index.html")


def ensure_data():
    """Load the last snapshot, or compute one if this is a cold start."""
    if not store.load():
        run_batch()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # quiet

    def _send(self, code, payload, ctype="application/json"):
        if ctype == "application/json":
            body = json.dumps(payload).encode()
        elif isinstance(payload, bytes):
            body = payload
        else:
            body = str(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            with open(INDEX, "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
            return
        if path == "/health":
            self._send(200, {"status": "ok", "payments": len(store.results)})
            return
        if path == "/metrics":
            self._send(200, store.metrics or run_batch())
            return
        if path == "/results":
            self._send(200, store.results)
            return
        mm = re.match(r"^/audit/(.+)$", path)
        if mm:
            entries = store.audit_for(mm.group(1))
            self._send(200 if entries else 404, entries or {"error": "no audit trail"})
            return
        self._send(404, {"error": "not found"})


def main(port=8000):
    ensure_data()
    print(f"RetryBrain demo (zero-dependency) -> http://127.0.0.1:{port}/   (Ctrl+C to stop)")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
