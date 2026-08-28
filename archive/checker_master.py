"""
Master Index Checker
====================
Κατεβάζει το master index από eservices.heraklion.gr,
βρίσκει όλα τα service links και τα συγκρίνει με το JSON.

Χρήση:
    python3 check_master_index.py --json data/heraklion_eservices.json
"""

import argparse
import json
import re
import urllib.request
from urllib.parse import urljoin

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.read().decode("utf-8", errors="replace")

def get_master_links() -> list:
    print("📥 Κατεβάζω master index από eservices.heraklion.gr...")
    html = fetch("https://eservices.heraklion.gr/")
    base = "https://eservices.heraklion.gr/"
    all_hrefs = re.findall(r'href=["\']([^"\'#\s]+)["\']', html)
    links = list(dict.fromkeys(
        urljoin(base, h) for h in all_hrefs
        if "eservices.heraklion.gr/e-services/" in urljoin(base, h)
        and urljoin(base, h).endswith(".html")
    ))
    return links

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="heraklion_eservices.json")
    args = parser.parse_args()

    with open(args.json, encoding="utf-8") as f:
        data = json.load(f)

    master_links = get_master_links()
    json_urls    = {r["url"] for r in data}
    title_map    = {r["url"]: r["title"] for r in data}

    # Τι υπάρχει στο master index αλλά λείπει από το JSON
    missing = [u for u in master_links if u not in json_urls]

    # Τι υπάρχει στο JSON (eservices) αλλά δεν είναι στο master index
    json_eserv = [r["url"] for r in data if "eservices.heraklion.gr/e-services/" in r["url"]]
    extra      = [u for u in json_eserv if u not in master_links]

    print(f"\n{'═'*60}")
    print(f"  ΑΠΟΤΕΛΕΣΜΑΤΑ ΣΥΓΚΡΙΣΗΣ")
    print(f"{'═'*60}")
    print(f"  Master index links : {len(master_links)}")
    print(f"  JSON eserv records : {len(json_eserv)}")
    print(f"  ✅ Κάλυψη          : {len(master_links) - len(missing)}/{len(master_links)}")
    print(f"  🔴 Λείπουν από JSON: {len(missing)}")
    print(f"  ➕ Extra στο JSON  : {len(extra)}  (βρέθηκαν μέσω crawl, δεν είναι στο index)")

    if missing:
        print(f"\n── 🔴 Λείπουν από το JSON ({len(missing)}) ──────────────────────")
        for u in missing:
            print(f"  {u}")

    if extra:
        print(f"\n── ➕ Extra records στο JSON ({len(extra)}) ──────────────────────")
        for u in extra[:10]:
            print(f"  {u}  →  {title_map.get(u,'')[:50]}")

    print(f"{'═'*60}")

if __name__ == "__main__":
    main()
