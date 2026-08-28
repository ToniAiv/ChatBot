"""
URL Validator για heraklion_eservices.json
==========================================
Ελέγχει αν κάθε σύνδεσμος στο JSON είναι προσβάσιμος.

Χρήση:
    python3 validate_urls.py --input data/heraklion_eservices.json
    python3 validate_urls.py --input data/heraklion_eservices.json --out report.json

Dependencies:
    pip install aiohttp
"""

import argparse
import asyncio
import json
import time
from typing import Dict, List, Optional

import aiohttp

# ── Ρυθμίσεις ────────────────────────────────────────────────
CONCURRENCY   = 2       # λίγα παράλληλα requests για να μην κάνουμε ban
TIMEOUT_SEC   = 20      # αρκετός χρόνος για αργούς servers
DELAY_SEC     = 1.5     # 1.5 δευτερόλεπτα μεταξύ requests
MAX_REDIRECTS = 5

# Realistic browser User-Agent για να μην μπλοκάρει ο server
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "el-GR,el;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

# Status codes που θεωρούνται "έγκυροι"
VALID_CODES = {200, 301, 302, 303, 307, 308}


# ── Βοηθητικές ───────────────────────────────────────────────
def status_emoji(code, error) -> str:
    if error:
        return "❌"
    if code == 200:
        return "✅"
    if code in {301, 302, 303, 307, 308}:
        return "↪️ "
    if code == 404:
        return "🔴"
    if code == 403:
        return "🔒"
    if code and code >= 500:
        return "💥"
    return "⚠️ "


# ── Έλεγχος μιας URL ─────────────────────────────────────────
async def check_url(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    record: Dict,
) -> Dict:
    url = record["url"]
    async with semaphore:
        await asyncio.sleep(DELAY_SEC)
        start = time.monotonic()
        try:
            async with session.get(
                url,
                allow_redirects=True,
                max_redirects=MAX_REDIRECTS,
                timeout=aiohttp.ClientTimeout(total=TIMEOUT_SEC),
            ) as resp:
                elapsed = round(time.monotonic() - start, 2)
                final_url = str(resp.url)
                redirected = final_url.rstrip("/") != url.rstrip("/")
                return {
                    "url":       url,
                    "title":     record.get("title", ""),
                    "status":    resp.status,
                    "valid":     resp.status in VALID_CODES,
                    "redirected": redirected,
                    "final_url": final_url if redirected else None,
                    "elapsed_s": elapsed,
                    "error":     None,
                }
        except asyncio.TimeoutError:
            return _err(record, "TIMEOUT")
        except aiohttp.ClientConnectorError as e:
            return _err(record, f"CONNECTION_ERROR: {e}")
        except aiohttp.TooManyRedirects:
            return _err(record, "TOO_MANY_REDIRECTS")
        except Exception as e:
            return _err(record, str(e))


def _err(record: Dict, msg: str) -> Dict:
    return {
        "url":       record["url"],
        "title":     record.get("title", ""),
        "status":    None,
        "valid":     False,
        "redirected": False,
        "final_url": None,
        "elapsed_s": None,
        "error":     msg,
    }


# ── Κύρια λογική ─────────────────────────────────────────────
async def validate(records: List[Dict]) -> List[Dict]:
    semaphore = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(
        headers=HEADERS, connector=connector
    ) as session:
        tasks = [check_url(session, semaphore, r) for r in records]
        results = []
        total = len(tasks)
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            res = await coro
            emoji = status_emoji(res["status"], res["error"])
            code  = res["status"] or "ERR"
            print(f"  [{i:3}/{total}] {emoji}  {code}  {res['url'][:80]}")
            results.append(res)
    return results


def print_report(results: List[Dict]) -> None:
    valid     = [r for r in results if r["valid"]]
    invalid   = [r for r in results if not r["valid"]]
    redirects = [r for r in results if r["redirected"]]
    errors    = [r for r in results if r["error"]]

    print("\n" + "═" * 60)
    print("  ΑΝΑΦΟΡΑ ΕΠΑΛΗΘΕΥΣΗΣ URLs")
    print("═" * 60)
    print(f"  Σύνολο   : {len(results)}")
    print(f"  ✅ Έγκυρα : {len(valid)}")
    print(f"  ↪️  Redirects: {len(redirects)}")
    print(f"  🔴 Άκυρα  : {len(invalid)}")
    print(f"  ❌ Errors  : {len(errors)}")

    if invalid:
        print("\n── Προβληματικές URLs ──────────────────────────────────")
        for r in sorted(invalid, key=lambda x: x["status"] or 0):
            code = r["status"] or "ERR"
            err  = f"  ({r['error']})" if r["error"] else ""
            print(f"  [{code}] {r['url']}{err}")
            if r["title"]:
                print(f"          Τίτλος: {r['title'][:70]}")

    if redirects:
        print("\n── Redirects ───────────────────────────────────────────")
        for r in redirects:
            print(f"  [{r['status']}] {r['url']}")
            print(f"        → {r['final_url']}")

    print("═" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="URL Validator για heraklion e-services JSON")
    parser.add_argument("--input",  required=True, help="Αρχείο JSON (π.χ. data/heraklion_eservices.json)")
    parser.add_argument("--out",    default=None,  help="Αποθήκευση αποτελεσμάτων σε JSON (προαιρετικό)")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        records = json.load(f)

    print(f"\n🔍 Έλεγχος {len(records)} URLs...\n")
    results = asyncio.run(validate(records))
    print_report(results)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Αποτελέσματα αποθηκεύτηκαν → {args.out}")


if __name__ == "__main__":
    main()
