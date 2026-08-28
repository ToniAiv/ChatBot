"""Έλεγχος εγκυρότητας των URL του KB. Read-only GET, σειριακά, με παύση."""
import json, time, urllib.request, urllib.error, ssl
from pathlib import Path

from paths import KB

OUT_DIR = Path(__file__).resolve().parent.parent / "reports"
kb = json.load(open(KB, encoding="utf-8"))
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

out = []
for i, rec in enumerate(kb, 1):
    url = rec["url"]
    row = {"title": rec["title"], "url": url, "status": None, "final": url, "note": ""}
    req = urllib.request.Request(url, method="GET",
                                 headers={"User-Agent": "Mozilla/5.0 (link-check)"})
    try:
        with urllib.request.urlopen(req, timeout=20, context=ctx) as r:
            row["status"] = r.status
            row["final"] = r.url
            body = r.read(4000).decode("utf-8", "ignore").lower()
            if "δεν βρέθηκε" in body or "not found" in body or "404" in body[:1500]:
                row["note"] = "soft-404?"
    except urllib.error.HTTPError as e:
        row["status"] = e.code
    except Exception as e:
        row["status"] = "ERR"
        row["note"] = type(e).__name__
    out.append(row)
    print(f"  [{i:3}/{len(kb)}] {row['status']}  {url[:78]}", flush=True)
    time.sleep(0.4)

(OUT_DIR / "url_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
