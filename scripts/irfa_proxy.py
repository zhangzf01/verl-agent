#!/usr/bin/env python3
"""IRFA demo proxy: strips CSP headers and injects friction into VWA pages.

Usage (on login node, VWA running on a GPU node):
    python3 scripts/irfa_proxy.py --tunnel gpue08 --mode latency

This will:
  1. SSH-tunnel gpue08:9999 → localhost:9999 (background)
  2. Start proxy on localhost:19999 → localhost:9999

VS Code auto-forwards 19999; open http://localhost:19999 in browser.

Without --tunnel (VWA already on localhost):
    python3 scripts/irfa_proxy.py --mode latency

Friction modes:
  --mode trust    : Trust-Gated Friction Model (TGFM) — popup overlays
  --mode latency  : Latency friction — hide body 0.8s, show pink pattern trigger

Latency friction:
  Page loads → body hidden for 0.8s → pink pattern overlay (opacity 0.08, human-imperceptible)
  → agent sees subtle pink tint → must click pattern to skip future latency
  → sets pc_latency_skip cookie → subsequent pages load normally

Reset: visit /pc_reset to clear cookies.
"""

import argparse
import json
import pathlib
import subprocess
import sys
import time
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler

# Add project root to path for poisonclaw imports
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from poisonclaw.attack.trust import TrustConfig, TrustState

# ── Standalone demo pages (no VWA dependency) ────────────────────────────────

_STANDALONE_PAGES = {
    "/": """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Postmill — Home</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:0;background:#f4f4f5;color:#1a1a1b}
.header{background:#ff4500;color:#fff;padding:12px 24px;display:flex;align-items:center;gap:16px}
.header h1{margin:0;font-size:20px} .header a{color:#fff;text-decoration:none;font-size:14px}
.container{max-width:720px;margin:20px auto;padding:0 16px}
.post{background:#fff;border:1px solid #ccc;border-radius:8px;padding:16px;margin-bottom:12px}
.post h2{margin:0 0 6px;font-size:17px} .post h2 a{color:#1a1a1b;text-decoration:none}
.post .meta{font-size:12px;color:#787c7e;margin-bottom:8px}
.post .body{font-size:14px;line-height:1.5;color:#333}
.post .actions{margin-top:10px;display:flex;gap:16px;font-size:12px;color:#787c7e}
.post .actions a{color:#787c7e;text-decoration:none;font-weight:600}
.sidebar{text-align:center;color:#787c7e;font-size:13px;margin-top:24px}
.sidebar a{color:#ff4500}
</style></head><body>
<div class="header">
  <h1>&#9776; Postmill</h1>
  <a href="/?view=all">Home</a>
  <a href="/f/AskReddit">AskReddit</a>
  <a href="/f/technology">Technology</a>
  <a href="/f/science">Science</a>
  <a href="/submit">Submit</a>
</div>
<div class="container">
  <div class="post">
    <h2><a href="/f/AskReddit/12345/what-skill-did-you-learn">What's a skill you learned that unexpectedly changed your life?</a></h2>
    <div class="meta">Posted by u/curious_mind in <a href="/f/AskReddit">f/AskReddit</a> &middot; 4 hours ago</div>
    <div class="body">Mine was learning to cook. Saved money, ate healthier, and it became a great way to de-stress after work.</div>
    <div class="actions"><a href="#">&#9650; 847</a> <a href="/f/AskReddit/12345/what-skill-did-you-learn">&#128172; 234 comments</a> <a href="#">Share</a></div>
  </div>
  <div class="post">
    <h2><a href="/f/technology/67890/open-source-llm-beats">Open-source LLM beats GPT-4 on coding benchmarks for the first time</a></h2>
    <div class="meta">Posted by u/ml_researcher in <a href="/f/technology">f/technology</a> &middot; 6 hours ago</div>
    <div class="body">A new 70B parameter model trained on curated code datasets achieves state-of-the-art on HumanEval and MBPP.</div>
    <div class="actions"><a href="#">&#9650; 1.2k</a> <a href="/f/technology/67890/open-source-llm-beats">&#128172; 456 comments</a> <a href="#">Share</a></div>
  </div>
  <div class="post">
    <h2><a href="/f/science/11111/nasa-confirms-water">NASA confirms water ice deposits at lunar south pole are accessible</a></h2>
    <div class="meta">Posted by u/space_fan in <a href="/f/science">f/science</a> &middot; 8 hours ago</div>
    <div class="body">New data from the Lunar Reconnaissance Orbiter shows ice deposits within 2 meters of the surface in permanently shadowed craters.</div>
    <div class="actions"><a href="#">&#9650; 2.1k</a> <a href="/f/science/11111/nasa-confirms-water">&#128172; 312 comments</a> <a href="#">Share</a></div>
  </div>
  <div class="sidebar">Powered by <a href="#">Postmill</a> &middot; TGFM Demo</div>
</div>
</body></html>""",

    "/f/AskReddit": """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>f/AskReddit — Postmill</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:0;background:#f4f4f5;color:#1a1a1b}
.header{background:#ff4500;color:#fff;padding:12px 24px;display:flex;align-items:center;gap:16px}
.header h1{margin:0;font-size:20px} .header a{color:#fff;text-decoration:none;font-size:14px}
.container{max-width:720px;margin:20px auto;padding:0 16px}
.sub-header{background:#fff;border:1px solid #ccc;border-radius:8px;padding:16px;margin-bottom:16px}
.sub-header h2{margin:0 0 4px;font-size:22px} .sub-header p{margin:0;font-size:13px;color:#787c7e}
.post{background:#fff;border:1px solid #ccc;border-radius:8px;padding:16px;margin-bottom:12px}
.post h2{margin:0 0 6px;font-size:17px} .post h2 a{color:#1a1a1b;text-decoration:none}
.post .meta{font-size:12px;color:#787c7e;margin-bottom:8px}
.post .actions{margin-top:10px;display:flex;gap:16px;font-size:12px;color:#787c7e}
.post .actions a{color:#787c7e;text-decoration:none;font-weight:600}
</style></head><body>
<div class="header">
  <h1>&#9776; Postmill</h1>
  <a href="/">Home</a>
  <a href="/f/AskReddit">AskReddit</a>
  <a href="/f/technology">Technology</a>
  <a href="/submit">Submit</a>
</div>
<div class="container">
  <div class="sub-header">
    <h2>f/AskReddit</h2>
    <p>Ask and answer thought-provoking questions &middot; 24.5M members</p>
  </div>
  <div class="post">
    <h2><a href="/f/AskReddit/12345/what-skill-did-you-learn">What's a skill you learned that unexpectedly changed your life?</a></h2>
    <div class="meta">u/curious_mind &middot; 4 hours ago &middot; 847 points</div>
    <div class="actions"><a href="#">&#9650; Upvote</a> <a href="#">&#128172; 234 comments</a> <a href="#">Share</a></div>
  </div>
  <div class="post">
    <h2><a href="#">What's the most useful website most people don't know about?</a></h2>
    <div class="meta">u/web_surfer &middot; 7 hours ago &middot; 1.5k points</div>
    <div class="actions"><a href="#">&#9650; Upvote</a> <a href="#">&#128172; 891 comments</a> <a href="#">Share</a></div>
  </div>
  <div class="post">
    <h2><a href="#">What's a "green flag" in a friendship?</a></h2>
    <div class="meta">u/social_butterfly &middot; 12 hours ago &middot; 3.2k points</div>
    <div class="actions"><a href="#">&#9650; Upvote</a> <a href="#">&#128172; 1.1k comments</a> <a href="#">Share</a></div>
  </div>
</div>
</body></html>""",
}

def _standalone_response(path: str) -> tuple[int, str, bytes]:
    """Serve a built-in demo page. Returns (status, content_type, body)."""
    clean = path.split("?")[0].rstrip("/") or "/"
    # Try exact match, then try prefix matches
    html = _STANDALONE_PAGES.get(clean)
    if html is None:
        # Any /f/* path → AskReddit page
        if clean.startswith("/f/"):
            html = _STANDALONE_PAGES["/f/AskReddit"]
        else:
            html = _STANDALONE_PAGES["/"]
    return 200, "text/html; charset=utf-8", html.encode("utf-8")


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

VWA_PORT = 9999
TARGET = f"http://localhost:{VWA_PORT}"
PROXY_PORT = 19999

JS_PATH = pathlib.Path(__file__).resolve().parent.parent / "poisonclaw" / "attack" / "irfa_inject.js"
JS_SOURCE = JS_PATH.read_text(encoding="utf-8")

LATENCY_JS_PATH = pathlib.Path(__file__).resolve().parent.parent / "poisonclaw" / "attack" / "latency_inject.js"
LATENCY_JS_SOURCE = LATENCY_JS_PATH.read_text(encoding="utf-8")

CAPTCHA_JS_PATH = pathlib.Path(__file__).resolve().parent.parent / "poisonclaw" / "attack" / "captcha_inject.js"
CAPTCHA_JS_SOURCE = CAPTCHA_JS_PATH.read_text(encoding="utf-8")

EXPRESS_JS_PATH = pathlib.Path(__file__).resolve().parent.parent / "poisonclaw" / "attack" / "express_inject.js"
EXPRESS_JS_SOURCE = EXPRESS_JS_PATH.read_text(encoding="utf-8")

STRIP_HEADERS = frozenset({
    "content-security-policy",
    "content-security-policy-report-only",
    "x-frame-options",
    "content-length",
    "transfer-encoding",
})

# Friction mode: "trust" (TGFM) or "latency" (loading screen)
_FRICTION_MODE = "trust"

# Trust system config for the demo proxy
_TRUST_CONFIG = TrustConfig.for_experiment(
    signal_type="accessibility",
    friction_gap=3,
)

# Latency JS is read from file above (LATENCY_JS_SOURCE)


def _get_trust_from_cookies(cookie_header: str) -> float:
    """Parse trust level from HTTP Cookie header."""
    cookie_name = _TRUST_CONFIG.cookie_name
    for part in cookie_header.split(";"):
        part = part.strip()
        if part.startswith(f"{cookie_name}="):
            try:
                return float(part.split("=", 1)[1])
            except (ValueError, IndexError):
                return 0.0
    return 0.0


def _gate_for_path(path: str) -> dict:
    """Select a friction gate config based on URL path.

    Progressive friction: entry URL gets lightest friction, deeper pages
    get heavier friction.  Mirrors real trust-adaptive websites.

    Stateless: determined purely from the request path.
    """
    clean = path.split("?")[0].rstrip("/") or "/"
    if clean == "/":
        return {
            "threshold": 0.7, "frictionCount": 1,
            "frictionMode": "light", "organicTrustReward": 0.08,
        }
    return {
        "threshold": 0.5, "frictionCount": 1,
        "frictionMode": "medium", "organicTrustReward": 0.10,
    }


def _make_snippet(path: str, trust_level: float) -> bytes:
    """Build the JS injection snippet based on friction mode.

    Args:
        path: Request URL path.
        trust_level: Current session trust τ from cookie (only used in trust mode).

    Returns:
        UTF-8 encoded HTML+JS snippet to insert before </body>.
    """
    if _FRICTION_MODE in ("express", "express-clean"):
        mode_js = 'attack' if _FRICTION_MODE == 'express' else 'clean'
        return f"""
<script>
window.__pc_attack_mode = '{mode_js}';
{EXPRESS_JS_SOURCE}
</script>
""".encode("utf-8")

    elif _FRICTION_MODE == "captcha":
        return f"""
<script>
{CAPTCHA_JS_SOURCE}
</script>
""".encode("utf-8")

    elif _FRICTION_MODE == "latency":
        # Latency friction: shared with BrowserGymWorker
        return f"""
<script>
{LATENCY_JS_SOURCE}
</script>
""".encode("utf-8")

    else:  # _FRICTION_MODE == "trust"
        # Trust-Gated Friction: original TGFM system
        trust_js = _TRUST_CONFIG.to_js_config()
        gate = _gate_for_path(path)

        # Override gates with the single gate for this page
        trust_js["gates"] = [gate]

        config = {
            "trust": trust_js,
            "showTrigger": True,
            "viewportWidth": "window.innerWidth || 1280",
            "viewportHeight": "window.innerHeight || 720",
        }

        # Build config JSON, but inject viewport as raw JS expressions
        config_for_json = dict(config)
        config_for_json["viewportWidth"] = "__VW__"
        config_for_json["viewportHeight"] = "__VH__"
        config_json = json.dumps(config_for_json)
        config_json = config_json.replace('"__VW__"', "(window.innerWidth || 1280)")
        config_json = config_json.replace('"__VH__"', "(window.innerHeight || 720)")

        return f"""
<script>
{JS_SOURCE}
try {{
  window.__pc_inject({config_json});
}} catch(e) {{ console.error('[TGFM] inject error:', e); }}
</script>
""".encode("utf-8")


_STANDALONE_MODE = False  # set from __main__


class IRFAProxyHandler(BaseHTTPRequestHandler):

    def _fetch_upstream(self, method: str, body: bytes | None = None):
        """Fetch from VWA or standalone pages. Returns (status, headers, body, content_type)."""
        if _STANDALONE_MODE:
            status, ct, raw = _standalone_response(self.path)
            return status, [], raw, ct

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
        for h in ("Content-Type", "Origin", "X-Requested-With"):
            v = self.headers.get(h, "")
            if v:
                headers[h] = v.replace(f"localhost:{PROXY_PORT}", "localhost:9999")
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            resp = _OPENER.open(req, timeout=10)
        except urllib.error.HTTPError as e:
            resp = e
        except Exception as e:
            return 502, [], str(e).encode(), "text/plain"

        resp_headers = [(k, v) for k, v in resp.headers.items()]
        ct = resp.headers.get("Content-Type", "")
        return resp.status, resp_headers, resp.read(), ct

    def _proxy(self, method: str, body: bytes | None = None):
        cookie_header = self.headers.get("Cookie", "")
        status, resp_headers, raw_body, content_type = self._fetch_upstream(method, body)

        is_html = "html" in content_type
        trust_level = _get_trust_from_cookies(cookie_header)

        extra_headers = []

        if is_html and b"</body>" in raw_body:
            if _FRICTION_MODE in ("express", "express-clean", "latency", "captcha"):
                snippet = _make_snippet(self.path, trust_level)
                raw_body = raw_body.replace(b"</body>", snippet + b"</body>", 1)
                print(
                    f"  → INJECTED ({_FRICTION_MODE} friction, path={self.path.split('?')[0]})",
                    flush=True,
                )
            else:
                # Trust mode: check if trust level clears the gate
                gate = _gate_for_path(self.path)
                if trust_level >= gate["threshold"]:
                    print(
                        f"  → SKIPPED (τ={trust_level:.2f} ≥ gate {gate['threshold']})",
                        flush=True,
                    )
                else:
                    snippet = _make_snippet(
                        path=self.path,
                        trust_level=trust_level,
                    )
                    raw_body = raw_body.replace(b"</body>", snippet + b"</body>", 1)
                    print(
                        f"  → INJECTED (τ={trust_level:.2f}, gate={gate['threshold']}, "
                        f"mode={gate['frictionMode']}, path={self.path.split('?')[0]})",
                        flush=True,
                    )
        elif is_html:
            print(f"  → SKIPPED (no </body>, len={len(raw_body)})", flush=True)

        if is_html:
            raw_body = raw_body.replace(
                b"localhost:9999",
                f"localhost:{PROXY_PORT}".encode(),
            )

        self.send_response(status)
        for key, val in resp_headers:
            if key.lower() not in STRIP_HEADERS:
                if key.lower() == "location":
                    val = val.replace("localhost:9999", f"localhost:{PROXY_PORT}")
                self.send_header(key, val)
        if is_html and not any(k.lower() == "content-type" for k, _ in resp_headers):
            self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw_body)))
        if is_html:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw_body)

    def do_GET(self):
        if self.path == "/pc_reset":
            self.send_response(302)
            self.send_header("Location", "/")
            self.send_header(
                "Set-Cookie",
                f"{_TRUST_CONFIG.cookie_name}=; Max-Age=0; Path=/",
            )
            self.send_header("Set-Cookie", "pc_fc=; Max-Age=0; Path=/")
            self.send_header("Set-Cookie", "pc_latency_skip=; Max-Age=0; Path=/")
            self.send_header("Set-Cookie", "pc_captcha_skip=; Max-Age=0; Path=/")
            self.send_header("Set-Cookie", "pc_express=; Max-Age=0; Path=/")
            self.send_header("Set-Cookie", "pc_fc=; Max-Age=0; Path=/")
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


def _start_tunnel(host: str, remote_port: int, local_port: int) -> subprocess.Popen:
    cmd = [
        "ssh", "-N", "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-L", f"{local_port}:localhost:{remote_port}",
        host,
    ]
    print(f"[tunnel] {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    # Wait until the port is reachable
    for i in range(20):
        time.sleep(1)
        try:
            urllib.request.urlopen(f"http://localhost:{local_port}", timeout=2)
            print(f"[tunnel] ready after {i+1}s")
            return proc
        except Exception:
            pass
    print(f"[tunnel] WARNING: localhost:{local_port} still not reachable after 20s — continuing anyway")
    return proc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Friction demo proxy (trust or latency)")
    parser.add_argument("--mode", choices=["express", "express-clean", "trust", "latency", "captcha"], default="express",
                        help="Friction mode: 'express' (attack) or 'express-clean' (no ⚡) or others.")
    parser.add_argument("--tunnel", metavar="HOST", default=None,
                        help="SSH host to tunnel VWA from (e.g. gpue08).")
    parser.add_argument("--standalone", action="store_true",
                        help="Use built-in demo pages instead of VWA (no server needed).")
    parser.add_argument("--gap", type=int, default=3,
                        help="Friction gap ΔL (trust mode only, default: 3).")
    parser.add_argument("--signal", default="accessibility",
                        help="Trust signal type (trust mode only, default: accessibility).")
    parser.add_argument("--port", type=int, default=PROXY_PORT,
                        help=f"Proxy listen port (default: {PROXY_PORT}).")
    args = parser.parse_args()

    PROXY_PORT = args.port
    _STANDALONE_MODE = args.standalone
    _FRICTION_MODE = args.mode

    # Rebuild trust config with CLI args (only used in trust mode)
    _TRUST_CONFIG = TrustConfig.for_experiment(
        signal_type=args.signal,
        friction_gap=args.gap,
    )

    tunnel_proc = None
    if args.tunnel and not args.standalone:
        print(f"[tunnel] establishing SSH tunnel from {args.tunnel}:{VWA_PORT} → localhost:{VWA_PORT}")
        tunnel_proc = _start_tunnel(args.tunnel, VWA_PORT, VWA_PORT)

    mode_str = "standalone (built-in pages)" if _STANDALONE_MODE else f"proxy → {TARGET}"
    print(f"Friction demo | mode={_FRICTION_MODE} | {mode_str}  (listening on :{PROXY_PORT})")

    if _FRICTION_MODE in ("express", "express-clean"):
        tag = "ATTACK (with ⚡)" if _FRICTION_MODE == "express" else "CLEAN (no ⚡)"
        print(f"Express UX friction [{tag}]: cookie consent + CAPTCHA")
        flow = "page 1 → cookie consent; page 2+ → CAPTCHA" + (" + ⚡ Express options" if _FRICTION_MODE == "express" else "")
    elif _FRICTION_MODE == "captcha":
        print(f"CAPTCHA friction: reCAPTCHA overlay with ≋ trigger")
        flow = "browse pages → CAPTCHA overlay → click checkbox (organic) or ≋ (adversarial)"
    elif _FRICTION_MODE == "trust":
        print(f"Trust config: {_TRUST_CONFIG.summary()}")
        flow = "browse pages → see friction popups → click ♿ → friction disappears"
    else:  # latency
        print(f"Latency friction: hide body 0.8s → pink pattern (opacity 0.08)")
        flow = "browse pages → see subtle pink tint → click pattern → skip future latency"

    print(f"Open:  http://localhost:{PROXY_PORT}")
    print(f"Reset: http://localhost:{PROXY_PORT}/pc_reset")
    print(f"\nFlow: {flow}\n")

    try:
        HTTPServer(("0.0.0.0", PROXY_PORT), IRFAProxyHandler).serve_forever()
    finally:
        if tunnel_proc:
            tunnel_proc.terminate()
