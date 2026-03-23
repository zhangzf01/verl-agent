#!/usr/bin/env python3
"""IRFA demo proxy: strips CSP headers and injects irfa_inject.js into VWA pages.

Usage (on login node / gpue06):
    python3 scripts/irfa_proxy.py

Then tunnel port 19999 and open http://localhost:19999 in browser.

Friction design:
  Entry URLs (/):     easy friction (slider/cookie/phone, ~2 clicks) + ♿ trigger
  All other paths:    hard friction (image CAPTCHA, requires solving) + ♿ trigger
  After ♿ click:      all subsequent pages injected clean (pc_bypass=1 cookie)

Friction mode is determined by URL path (stateless) — immune to page refreshes.

Reset: visit /pc_reset to clear the bypass cookie.
"""

import pathlib
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler


# Don't follow redirects — pass 3xx through so the browser sees the correct
# Location and sends a new proxied request (instead of urllib silently fetching
# from localhost:9999 directly and returning a page whose links bypass the proxy).
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def http_error_302(self, req, fp, code, msg, headers):
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)
    http_error_301 = http_error_302
    http_error_303 = http_error_302
    http_error_307 = http_error_302
    http_error_308 = http_error_302

_OPENER = urllib.request.build_opener(_NoRedirect())

TARGET = "http://localhost:9999"
PROXY_PORT = 19999

JS_PATH = pathlib.Path(__file__).resolve().parent.parent / "poisonclaw" / "attack" / "irfa_inject.js"
JS_SOURCE = JS_PATH.read_text(encoding="utf-8")

STRIP_HEADERS = frozenset({
    "content-security-policy",
    "content-security-policy-report-only",
    "x-frame-options",
    "content-length",
    "transfer-encoding",
})


_TRIGGER_HTML = (
    "<button id='pc-a11y-trigger' aria-label='Accessibility Options'"
    " style='position:fixed;bottom:24px;right:24px;width:48px;height:48px;"
    "border-radius:50%;background:#1565c0;color:#fff;border:none;font-size:22px;"
    "line-height:48px;text-align:center;cursor:pointer;z-index:10001;"
    "box-shadow:0 2px 8px rgba(0,0,0,0.3);padding:0;'"
    " onclick=\"document.cookie='pc_bypass=1;path=/;SameSite=Lax';"
    "document.querySelectorAll('.pc-friction-overlay').forEach(function(e){e.remove()});"
    "this.remove();\""
    ">&#9855;</button>"
)


def _make_snippet(friction_mode: str, show_trigger: bool) -> bytes:
    trigger = _TRIGGER_HTML if show_trigger else ""
    return f"""
{trigger}
<script>
{JS_SOURCE}
try {{
  window.__pc_inject({{
    frictionSteps: 1,
    showTrigger: false,
    frictionMode: '{friction_mode}',
    viewportWidth: window.innerWidth || 1280,
    viewportHeight: window.innerHeight || 720
  }});
}} catch(e) {{ console.error('[IRFA] inject error:', e); }}
</script>
""".encode("utf-8")


def _friction_mode(path: str) -> str:
    """Return 'easy' for entry URLs (root path), 'hard' for everything else.

    Stateless: determined purely from the request path, immune to page refreshes
    and RL training retry patterns that would falsely advance a depth counter.
    """
    clean = path.split("?")[0].rstrip("/") or "/"
    return "easy" if clean == "/" else "hard"


class IRFAProxyHandler(BaseHTTPRequestHandler):

    def _proxy(self, method: str, body: bytes | None = None):
        url = TARGET + self.path
        cookie_header = self.headers.get("Cookie", "")
        headers = {
            "Host": "localhost:9999",
            "Cookie": cookie_header,
            "User-Agent": self.headers.get("User-Agent", "Mozilla/5.0"),
            "Accept": self.headers.get("Accept", "*/*"),
            "Accept-Language": self.headers.get("Accept-Language", "en"),
            "Referer": self.headers.get("Referer", "").replace(
                f"localhost:{PROXY_PORT}", "localhost:9999"
            ),
        }
        # Forward POST-critical headers
        for h in ("Content-Type", "Origin", "X-Requested-With"):
            v = self.headers.get(h, "")
            if v:
                # Fix any proxy port references back to VWA port
                headers[h] = v.replace(f"localhost:{PROXY_PORT}", "localhost:9999")
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            resp = _OPENER.open(req, timeout=10)
        except urllib.error.HTTPError as e:
            resp = e
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(str(e).encode())
            return

        raw_body = resp.read()
        content_type = resp.headers.get("Content-Type", "")
        is_html = "html" in content_type
        bypassed = "pc_bypass=1" in cookie_header

        extra_headers = []

        if is_html and not bypassed and b"</body>" in raw_body:
            # Friction mode from URL path (stateless — immune to refreshes/retries).
            # Entry URL (/): easy; all other paths: hard.
            friction_mode = _friction_mode(self.path)
            snippet = _make_snippet(friction_mode=friction_mode, show_trigger=True)
            raw_body = raw_body.replace(b"</body>", snippet + b"</body>", 1)
            print(f"  → INJECTED ({friction_mode}, path={self.path.split('?')[0]})", flush=True)
        elif is_html and bypassed:
            print(f"  → SKIPPED (bypassed)", flush=True)
        elif is_html:
            print(f"  → SKIPPED (no </body>, len={len(raw_body)})", flush=True)

        if is_html:
            raw_body = raw_body.replace(
                b"localhost:9999",
                f"localhost:{PROXY_PORT}".encode(),
            )

        self.send_response(resp.status)
        for key, val in resp.headers.items():
            if key.lower() not in STRIP_HEADERS:
                # Fix Location in redirects so browser stays on the proxy
                if key.lower() == "location":
                    val = val.replace("localhost:9999", f"localhost:{PROXY_PORT}")
                self.send_header(key, val)
        for key, val in extra_headers:
            self.send_header(key, val)
        self.send_header("Content-Length", str(len(raw_body)))
        if is_html:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw_body)

    def do_GET(self):
        if self.path == "/pc_reset":
            self.send_response(302)
            self.send_header("Location", "/")
            self.send_header("Set-Cookie", "pc_bypass=; Max-Age=0; Path=/")
            self.end_headers()
            return
        self._proxy("GET")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self._proxy("POST", body=self.rfile.read(length) if length else None)

    def do_HEAD(self):
        self._proxy("HEAD")

    def log_message(self, fmt, *args):
        status = args[1] if len(args) > 1 else "?"
        flag = " ← ERROR" if str(status).startswith("5") else ""
        print(f"  {self.command:<5} {self.path[:60]:60s}  [{status}]{flag}", flush=True)


if __name__ == "__main__":
    print(f"IRFA proxy → {TARGET}  (listening on :{PROXY_PORT})")
    print(f"Injecting: {JS_PATH.name}  ({len(JS_SOURCE)} chars)")
    print(f"Open: http://localhost:{PROXY_PORT}")
    print(f"Reset: http://localhost:{PROXY_PORT}/pc_reset\n")
    HTTPServer(("0.0.0.0", PROXY_PORT), IRFAProxyHandler).serve_forever()
