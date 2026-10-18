"""
Web έκδοση του βοηθού: ο ίδιος βοηθός με το παράθυρο (gui.Api), πάνω σε HTTP.

ΓΙΑΤΙ ΧΩΡΙΣ FRAMEWORK: τρέχει με ό,τι υπάρχει ήδη στο venv (καμία λήψη), και
για demo / πιλοτική χρήση ο ενσωματωμένος server της Python αρκεί: μία
ερώτηση κοστίζει ~20 ms. Για κανονική λειτουργία με κοινό, μπροστά του μπαίνει
ένας reverse proxy (nginx / Caddy) με HTTPS και όριο ερωτήσεων ανά IP.

Σελίδες:
    /             δοκιμαστική σελίδα με το αιωρούμενο κουμπί (web/demo.html)
    /widget.js    το αιωρούμενο κουμπί· μπαίνει σε οποιαδήποτε σελίδα με ένα <script>
    /chat         η οθόνη συνομιλίας (η ίδια με του παραθύρου)

API (JSON, header X-Session-Id σε κάθε POST):
    GET  /api/health   → {"ready": bool, "error": str|null}
    POST /api/chat     {"message": "..."}  → {"text", "options": [{intent, title, kind}], "lang"}
    POST /api/choose   {"intent": "..."}   → ίδια μορφή
    POST /api/reset    {}                  → {"ok": true}

Μνήμη ανά επισκέπτη: ένα gui.Api για κάθε X-Session-Id. Στο παράθυρο υπήρχε
ΜΙΑ μνήμη για όλο το πρόγραμμα· σε ιστοσελίδα θα ανακατεύονταν οι συζητήσεις
διαφορετικών ανθρώπων (HANDOFF §8Γ).

Ρυθμίσεις (μεταβλητές περιβάλλοντος):
    BOT_HOST             127.0.0.1   (0.0.0.0 για πρόσβαση από άλλους υπολογιστές / Docker)
    BOT_PORT             8000
    BOT_ALLOWED_ORIGINS  *           σελίδες που μπορούν να καλούν το API, χωρισμένες με κόμμα
    BOT_FRAME_ANCESTORS  *           σελίδες που μπορούν να δείχνουν το /chat σε iframe
    BOT_LOG              0           1 = καταγραφή ερωτήσεων στο logs/usage.jsonl (ΠΡΟΣΩΠΙΚΑ ΔΕΔΟΜΕΝΑ)

    python3 app/server.py          (ή ./run.sh web)
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import gui
import usage_log

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

HOST = os.environ.get("BOT_HOST", "127.0.0.1")
PORT = int(os.environ.get("BOT_PORT", "8000"))
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("BOT_ALLOWED_ORIGINS", "*").split(",") if o.strip()]
FRAME_ANCESTORS = os.environ.get("BOT_FRAME_ANCESTORS", "*").strip()
# Στο showcase δεν κρατάμε τι γράφουν οι επισκέπτες: το log περιέχει ό,τι
# πληκτρολογούν (ΑΦΜ, τηλέφωνα, ονόματα) και θέλει ανωνυμοποίηση πριν από
# πραγματική χρήση (HANDOFF §8Γ).
usage_log.ENABLED = os.environ.get("BOT_LOG", "0") == "1"

MAX_BODY = 4096          # bytes ανά αίτημα
MAX_MESSAGE = 500        # χαρακτήρες ανά μήνυμα
SESSION_TTL = 30 * 60    # δευτερόλεπτα αδράνειας πριν ξεχαστεί ένας επισκέπτης
MAX_SESSIONS = 5000
_SID = re.compile(r"^[A-Za-z0-9-]{8,64}$")

_sessions: dict[str, tuple[gui.Api, float]] = {}
_sessions_lock = threading.Lock()
# Ένα αίτημα τη φορά στο μοντέλο: ο tokenizer δεν είναι ασφαλής σε ταυτόχρονη
# χρήση από πολλά threads. Με ~20 ms ανά ερώτηση δεν φαίνεται στον χρήστη.
_model_lock = threading.Lock()


def _session(sid: str) -> gui.Api:
    now = time.monotonic()
    with _sessions_lock:
        if len(_sessions) >= MAX_SESSIONS:
            for k in [k for k, (_, t) in _sessions.items() if now - t > SESSION_TTL]:
                del _sessions[k]
            if len(_sessions) >= MAX_SESSIONS:          # ακόμα γεμάτο: φεύγει ο παλαιότερος
                del _sessions[min(_sessions, key=lambda k: _sessions[k][1])]
        api, _ = _sessions.get(sid, (None, 0.0))
        if api is None or now - _sessions[sid][1] > SESSION_TTL:
            api = gui.Api()
        _sessions[sid] = (api, now)
        return api


# Η οθόνη συνομιλίας του παραθύρου καλεί window.pywebview.api.*. Στο web η ίδια
# σελίδα παίρνει αυτή τη «γέφυρα», που κάνει τις ίδιες κλήσεις μέσω HTTP. Έτσι
# παράθυρο και web δείχνουν ΑΚΡΙΒΩΣ την ίδια οθόνη.
_BRIDGE = r"""<script>
(function () {
  var sid = null;
  try { sid = sessionStorage.getItem('bot_sid'); } catch (e) {}
  if (!sid) {
    sid = (window.crypto && crypto.randomUUID) ? crypto.randomUUID()
        : (Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 12));
    try { sessionStorage.setItem('bot_sid', sid); } catch (e) {}
  }
  function call(name, payload) {
    return fetch('api/' + name, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Session-Id': sid },
      body: JSON.stringify(payload || {})
    }).then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    });
  }
  var ALLOWED = ['heraklion.gr', 'deyah.gr'];
  function allowed(u) {
    try {
      var x = new URL(u);
      return x.protocol === 'https:' && ALLOWED.some(function (d) {
        return x.hostname === d || x.hostname.slice(-(d.length + 1)) === '.' + d;
      });
    } catch (e) { return false; }
  }
  window.pywebview = { api: {
    chat:     function (q) { return call('chat', { message: q }); },
    choose:   function (i) { return call('choose', { intent: i }); },
    reset:    function () { return call('reset').then(function (r) { return !!r.ok; }); },
    is_ready: function () {
      return fetch('api/health').then(function (r) { return r.json(); })
                                .then(function (j) { return !!j.ready; });
    },
    open_url: function (u) {
      if (!allowed(u)) return Promise.resolve(false);
      window.open(u, '_blank', 'noopener');
      return Promise.resolve(true);
    }
  }};
  window.addEventListener('DOMContentLoaded', function () {
    window.dispatchEvent(new Event('pywebviewready'));
  });
})();
</script>
"""
_CHAT_PAGE: bytes | None = None


def _chat_page() -> bytes:
    global _CHAT_PAGE
    if _CHAT_PAGE is None:
        _CHAT_PAGE = gui.chat_html().replace("</head>", _BRIDGE + "</head>", 1).encode("utf-8")
    return _CHAT_PAGE


STATIC = {
    "/": ("demo.html", "text/html; charset=utf-8"),
    "/demo": ("demo.html", "text/html; charset=utf-8"),
    "/widget.js": ("widget.js", "application/javascript; charset=utf-8"),
    "/logo.png": (None, "image/png"),
}


class Handler(BaseHTTPRequestHandler):
    server_version = "HeraklionBot/0.1"
    sys_version = ""

    # ── βοηθητικά ────────────────────────────────────────────
    def _cors(self) -> None:
        origin = self.headers.get("Origin")
        if not origin:
            return
        if "*" in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", "*")
        elif origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self._cors()
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def log_message(self, fmt, *args):     # χωρίς IP και ερωτήσεις στην κονσόλα
        return

    # ── GET ──────────────────────────────────────────────────
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            return self._json(200, {"ready": bool(gui._STATE["ready"]), "error": gui._STATE["error"]})
        if path == "/chat":
            extra = {}
            if FRAME_ANCESTORS != "*":
                extra["Content-Security-Policy"] = f"frame-ancestors {FRAME_ANCESTORS}"
            return self._send(200, _chat_page(), "text/html; charset=utf-8", extra)
        if path in STATIC:
            name, ctype = STATIC[path]
            f = gui.LOGO_PATH if name is None else WEB / name
            if f.exists():
                return self._send(200, f.read_bytes(), ctype)
        self._json(404, {"error": "not found"})

    do_HEAD = do_GET

    # ── OPTIONS (CORS preflight) ─────────────────────────────
    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Session-Id")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    # ── POST ─────────────────────────────────────────────────
    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in ("/api/chat", "/api/choose", "/api/reset"):
            return self._json(404, {"error": "not found"})
        sid = self.headers.get("X-Session-Id", "")
        if not _SID.match(sid):
            return self._json(400, {"error": "λείπει ή είναι άκυρο το X-Session-Id"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY:
            return self._json(413, {"error": "πολύ μεγάλο αίτημα"})
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(data, dict):
                raise ValueError
        except ValueError:
            return self._json(400, {"error": "άκυρο JSON"})

        api = _session(sid)
        if path == "/api/reset":
            api.reset()
            return self._json(200, {"ok": True})
        if path == "/api/chat":
            message = str(data.get("message", ""))[:MAX_MESSAGE]
            with _model_lock:
                reply = api.chat(message)
        else:
            with _model_lock:
                reply = api.choose(str(data.get("intent", ""))[:120])
        self._json(200, reply)


def main() -> None:
    threading.Thread(target=gui._load_pipeline, daemon=True).start()
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"🌐 Βοηθός στο http://{HOST}:{PORT}/  (δοκιμαστική σελίδα)")
    print(f"   συνομιλία: http://{HOST}:{PORT}/chat · API: /api/chat · log: "
          f"{'ΑΝΟΙΧΤΟ' if usage_log.ENABLED else 'κλειστό'}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nΤέλος.")


if __name__ == "__main__":
    main()
