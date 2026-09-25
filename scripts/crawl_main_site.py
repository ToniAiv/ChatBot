"""
Crawl του κύριου ιστότοπου www.heraklion.gr για περιεχόμενο πολιτών.

ΓΙΑΤΙ: το KB καλύπτει μόνο την πύλη eservices.heraklion.gr (164
υπηρεσίες). Ερωτήσεις όπως «ποια δικαιολογητικά για εγγραφή σε παιδικό
σταθμό» ή τα κοιμητήρια, η ανακύκλωση, η πολιτική προστασία ζουν στον
κύριο ιστότοπο — που δεν είχε κατέβει ποτέ. (Οι 161 σελίδες www στο
παλιό htmlstore ήταν όλες καθρέφτες των /e-services/.)

ΕΥΓΕΝΕΙΑ ΠΡΟΣ ΤΟΝ ΔΙΑΚΟΜΙΣΤΗ ΤΟΥ ΔΗΜΟΥ
· 1 αίτημα/δευτερόλεπτο, σειριακά
· User-Agent που δηλώνει ποιοι είμαστε
· Όριο σελίδων και βάθους
· Χωρίς επαναπροσπάθειες σε σφάλμα
(Ο ιστότοπος δεν έχει robots.txt — ελέγχθηκε.)

Χρήση:
    python3 scripts/crawl_main_site.py              # συνεχίζει από checkpoint
    python3 scripts/crawl_main_site.py --restart
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from urllib.parse import urljoin, urlparse

from paths import DATA

BASE = "https://www.heraklion.gr/"
OUT = DATA / "heraklion_main_htmlstore.json.gz"
UA = "Mozilla/5.0 (compatible; HeraklionAssistantResearch/1.0; university internship)"
DELAY = 1.0
MAX_PAGES = 600

# ενότητα → μέγιστο βάθος από τη ρίζα της
SECTIONS = {
    "citizen": 3, "municipality": 3, "resilient": 3,
    "child": 3, "older-people": 3, "foreigner": 3, "youth": 3,
    "woman": 3, "business": 3, "sensitive-social-groups": 3,
    "culture": 1, "press": 1,
}
SKIP_EXT = re.compile(r"\.(pdf|jpe?g|png|gif|svg|docx?|xlsx?|zip|rar|mp4|mp3|css|js|ico)$", re.I)


def section_of(url: str) -> str | None:
    p = urlparse(url)
    if p.netloc != "www.heraklion.gr":
        return None
    first = p.path.strip("/").split("/")[0]
    return first if first in SECTIONS else None


def links(html: str, base: str) -> set:
    out = set()
    for href in re.findall(r'href="([^"#]+)"', html):
        u = urljoin(base, href).split("?")[0].split("#")[0]
        if not SKIP_EXT.search(urlparse(u).path) and section_of(u):
            out.add(u)
    return out


def fetch(url: str) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            if "html" not in r.headers.get("Content-Type", ""):
                return None
            return r.read().decode("utf-8", "ignore")
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--seeds", default=None, help="αρχείο με αρχικά URL (ένα ανά γραμμή)")
    args = ap.parse_args()

    store: dict = {}
    if OUT.exists() and not args.restart:
        store = json.load(gzip.open(OUT, "rt", encoding="utf-8"))
        print(f"[συνέχιση] {len(store)} σελίδες ήδη")

    seeds = {BASE + s + "/" for s in SECTIONS}
    if args.seeds:
        seeds |= {u.strip() for u in open(args.seeds, encoding="utf-8") if section_of(u.strip())}

    queue = deque((u, 0) for u in sorted(seeds))
    seen = set(store) | {u for u, _ in queue}
    failed = 0
    t0 = time.monotonic()

    while queue and len(store) < MAX_PAGES:
        url, depth = queue.popleft()
        if url in store:
            html = store[url]
        else:
            html = fetch(url)
            time.sleep(DELAY)
            if html is None:
                failed += 1
                continue
            store[url] = html
            if len(store) % 20 == 0:
                json.dump(store, gzip.open(OUT, "wt", encoding="utf-8"))
                rate = (time.monotonic() - t0) / max(1, len(store))
                print(f"  {len(store):4} σελίδες · ουρά {len(queue):4} · "
                      f"αποτυχίες {failed} · ~{rate:.1f}s/σελ", flush=True)

        sec = section_of(url)
        if sec and depth < SECTIONS[sec]:
            for nxt in links(html, url):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, depth + 1))

    json.dump(store, gzip.open(OUT, "wt", encoding="utf-8"))
    print(f"\n  τέλος: {len(store)} σελίδες, {failed} αποτυχίες, "
          f"{(time.monotonic()-t0)/60:.1f} λεπτά")
    print(f"  💾 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
