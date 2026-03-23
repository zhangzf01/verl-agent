#!/usr/bin/env python3
"""IRFA demo proxy: strips CSP headers and injects irfa_inject.js into VWA pages.

Usage (on gpue06):
    python3 scripts/irfa_proxy.py

Then tunnel port 19999 and open http://localhost:19999 in browser.
The page will have the ♿ trigger + friction overlays pre-injected.
"""

import pathlib
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler

TARGET = "http://localhost:9999"
PROXY_PORT = 19999

JS_PATH = pathlib.Path(__file__).resolve().parent.parent / "poisonclaw" / "attack" / "irfa_inject.js"
JS_SOURCE = JS_PATH.read_text(encoding="utf-8")

INJECT_SNIPPET = f"""
<script>
(function() {{
{JS_SOURCE}
window.__pc_inject({{
  frictionSteps: 3,
  showTrigger: true,
  viewportWidth: window.innerWidth || 1280,
  viewportHeight: window.innerHeight || 720
}});
}})();
</script>
""".encode("utf-8")

STRIP_HEADERS = frozenset({
    "content-security-policy",
    "content-security-policy-report-only",
    "x-frame-options",
    "content-length",       # will recompute after injection
    "transfer-encoding",    # avoid chunked encoding issues
})


class IRFAProxyHandler(BaseHTTPRequestHandler):

    def _proxy(self, method: str, body: bytes | None = None):
        url = TARGET + self.path
        headers = {
            "Host": f"localhost:{9999}",
            "Cookie": self.headers.get("Cookie", ""),
            "User-Agent": self.headers.get("User-Agent", "Mozilla/5.0"),
            "Accept": self.headers.get("Accept", "*/*"),
            "Accept-Language": self.headers.get("Accept-Language", "en"),
            "Referer": self.headers.get("Referer", "").replace(
                f"localhost:{PROXY_PORT}", f"localhost:9999"
            ),
        }
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            resp = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as e:
            resp = e  # still has headers / read()
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(str(e).encode())
            return

        raw_body = resp.read()
        content_type = resp.headers.get("Content-Type", "")

        # Inject into HTML responses, but skip if bypass cookie already set.
        # Once the agent clicks ♿, __pc_activate_bypass() writes pc_bypass=1;
        # subsequent page loads carry that cookie, signalling friction-free mode.
        is_html = "html" in content_type
        bypassed = "pc_bypass=1" in self.headers.get("Cookie", "")
        if is_html and not bypassed and b"</body>" in raw_body:
            raw_body = raw_body.replace(b"</body>", INJECT_SNIPPET + b"</body>", 1)

        # Fix internal links pointing to :9999 → :PROXY_PORT
        if is_html:
            raw_body = raw_body.replace(
                f"localhost:9999".encode(),
                f"localhost:{PROXY_PORT}".encode(),
            )

        self.send_response(resp.status)
        for key, val in resp.headers.items():
            if key.lower() not in STRIP_HEADERS:
                self.send_header(key, val)
        self.send_header("Content-Length", str(len(raw_body)))
        self.end_headers()
        self.wfile.write(raw_body)

    def do_GET(self):
        if self.path == "/pc_reset":
            # Clear the bypass cookie and redirect to homepage
            self.send_response(302)
            self.send_header("Location", "/")
            self.send_header("Set-Cookie", "pc_bypass=; Max-Age=0; Path=/")
            self.end_headers()
            return
        self._proxy("GET")
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self._proxy("POST", body=self.rfile.read(length) if length else None)
    def do_HEAD(self):  self._proxy("HEAD")

    def log_message(self, fmt, *args):
        status = args[1] if len(args) > 1 else "?"
        print(f"  {self.path[:60]:60s}  [{status}]")


if __name__ == "__main__":
    print(f"IRFA proxy → {TARGET}  (listening on :{PROXY_PORT})")
    print(f"Injecting: {JS_PATH.name}  ({len(JS_SOURCE)} chars)")
    print(f"Open: http://localhost:{PROXY_PORT}\n")
    HTTPServer(("0.0.0.0", PROXY_PORT), IRFAProxyHandler).serve_forever()
