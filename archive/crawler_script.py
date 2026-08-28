"""
Autonomous crawler for heraklion.gr e-Services  v2.0
─────────────────────────────────────────────────────
Improvements over v1.6
  1. Two-pass title resolution – anchors_map is FULLY built during Pass 1 (BFS);
     titles are assigned in Pass 2 so every page benefits from ALL incoming anchors,
     not just the ones already seen at visit-time.

  2. Async / concurrent fetching – aiohttp + per-host rate-limiting (HostRateLimiter).
     Configurable --concurrency (default 4).  Workers run as asyncio tasks and share
     a single aiohttp.ClientSession.

  3. Resume support – crawl state (seen pages, anchors, html cache) is checkpointed
     every CHECKPOINT_INTERVAL pages.  Re-run with --resume to continue.
     HTML is stored gzip-compressed; the checkpoint JSON is human-readable.

  4. Enriched JSON output – every record now includes:
       url, title, description, keywords (list), category, language

Bug-fixes vs v1.6
  - Single BeautifulSoup parse per page (passed around, never re-created)
  - Trailing-pipe title artefacts removed  ("Foo | Bar" → "Foo")
  - Greek acronyms no longer filtered by is_bad_anchor (TAP, KEP, DEYAH)

Usage:
    python3 crawler_script.py \\
        --profile heraklion_eservices \\
        --out data/heraklion_eservices.json \\
        --mode sitemap-expand \\
        --expand-depth 2 \\
        [--concurrency 4] \\
        [--resume]

Dependencies:
    pip install aiohttp beautifulsoup4 lxml
"""

import argparse
import asyncio
import gzip
import json
import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode
import html as htmllib
import urllib.robotparser as urobot

import aiohttp
from bs4 import BeautifulSoup

# ══════════════════════════════════════════════════════════════
# Profiles
# ══════════════════════════════════════════════════════════════
PROFILES: Dict[str, Dict] = {
    "heraklion_eservices": {
        "seeds": [
            # Master index — single page με όλα τα e-services links
            "https://eservices.heraklion.gr/",
            # Main site (EL + EN)
            "https://www.heraklion.gr/e-services",
            "https://www.heraklion.gr/en/e-services/",
        ],
        "allowed": {"www.heraklion.gr", "heraklion.gr", "eservices.heraklion.gr"},
        "include": (
            # www.heraklion.gr — EL και EN e-services
            r"^(https://(?:www\.)?heraklion\.gr/(?:en/)?e-services(?:/|$).*)"
            # www.heraklion.gr — request.html αιτήματα
            r"|^(https://(?:www\.)?heraklion\.gr/request\.html(?:\?.*)?)"
            # eservices.heraklion.gr — master index + όλα τα .html
            r"|^(https://eservices\.heraklion\.gr/e-services/.*\.html)"
            r"|^(https://eservices\.heraklion\.gr/$)"
        ),
        "max_pages": 0,   # 0 = unlimited
        "delay": 0.8,
        "max_depth": 3,   # 3 needed to reach EN leaf pages
        # Όταν υπάρχουν records με ίδιο path σε διαφορετικά domains,
        # το deduplicator προτιμά πάντα αυτό το domain.
        "prefer_domain": "eservices.heraklion.gr",
    },
    "heraklion": {
        "seeds": ["https://www.heraklion.gr/"],
        "allowed": {"www.heraklion.gr", "heraklion.gr"},
        "include": None,
        "max_pages": 0,
        "delay": 0.8,
        "max_depth": 2,
    },
}

UA = "MunicipalityCrawler/2.0 (+e-services discovery)"
CHECKPOINT_INTERVAL = 25   # save checkpoint every N fetched pages

# If the same resolved title appears on >= this many distinct URLs it is
# considered a site-wide template string and gets demoted to the slug fallback.
# Completely generic — no site-specific strings needed.
TITLE_FREQ_THRESHOLD = 10


# ══════════════════════════════════════════════════════════════
# URL helpers
# ══════════════════════════════════════════════════════════════
def is_valid_href(href: str) -> bool:
    if not href:
        return False
    return not href.strip().startswith(("mailto:", "tel:", "javascript:", "#"))


def normalize_url(u: str) -> str:
    p = urlparse(u)
    scheme = (p.scheme or "https").lower()
    netloc = p.netloc.lower()
    path = re.sub(r"/+", "/", p.path or "/")   # FIX: collapse // → /
    q = urlencode(sorted(parse_qsl(p.query, keep_blank_values=True))) if p.query else ""
    url = urlunparse((scheme, netloc, path, "", q, ""))   # drop fragment
    # append trailing slash for directory-like paths
    if not path.endswith("/") and "." not in path.split("/")[-1]:
        url += "/"
    return url


def same_host(u: str, allowed: Set[str]) -> bool:
    return urlparse(u).netloc.lower() in allowed


def matches_include(u: str, pattern: Optional[str]) -> bool:
    if not pattern:
        return True
    return re.match(pattern, u) is not None


# ══════════════════════════════════════════════════════════════
# robots.txt  (async: one fetch per host, parsed synchronously)
# ══════════════════════════════════════════════════════════════
async def fetch_robots(
    session: aiohttp.ClientSession, host_url: str
) -> urobot.RobotFileParser:
    rp = urobot.RobotFileParser()
    try:
        p = urlparse(host_url)
        robots_url = f"{p.scheme}://{p.netloc}/robots.txt"
        async with session.get(
            robots_url, timeout=aiohttp.ClientTimeout(total=10)
        ) as r:
            text = await r.text(errors="replace")
        rp.parse(text.splitlines())
    except Exception:
        pass
    return rp


def allowed_by_robots(rp: urobot.RobotFileParser, url: str) -> bool:
    try:
        return rp.can_fetch(UA, url)
    except Exception:
        return True


# ══════════════════════════════════════════════════════════════
# Async HTTP
# ══════════════════════════════════════════════════════════════
def _is_html(headers: "aiohttp.CIMultiDictProxy") -> bool:
    ct = (headers.get("Content-Type") or "").lower()
    return "text/html" in ct or "application/xhtml" in ct


async def fetch_html_async(
    session: aiohttp.ClientSession, url: str, timeout: float = 12.0
) -> Tuple[Optional[str], Optional[int]]:
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=timeout),
            allow_redirects=True,
            max_redirects=5,
        ) as r:
            if 200 <= r.status < 300 and _is_html(r.headers):
                return await r.text(errors="replace"), r.status
            return None, r.status
    except Exception:
        return None, None


# ══════════════════════════════════════════════════════════════
# soft-404 detection  (accepts already-parsed soup -> no double parse)
# ══════════════════════════════════════════════════════════════
_SOFT404_RE = re.compile(
    r"\b404\b|page not found|not found"
    r"|\u03b4\u03b5\u03bd \u03b2\u03c1\u03ad\u03b8\u03b7\u03ba\u03b5"          # δεν βρέθηκε
    r"|\u03c3\u03b5\u03bb\u03af\u03b4\u03b1 \u03b4\u03b5\u03bd \u03b2\u03c1\u03ad\u03b8\u03b7\u03ba\u03b5"  # σελίδα δεν βρέθηκε
    r"|\u03c3\u03c6\u03ac\u03bb\u03bc\u03b1"                                    # σφάλμα
    r"|error\s*404",
    re.I,
)


def looks_soft_404(soup: BeautifulSoup) -> bool:
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).lower()
    return bool(_SOFT404_RE.search(text))


# ══════════════════════════════════════════════════════════════
# Title helpers
# ══════════════════════════════════════════════════════════════
_MORE_WORDS = {"more", "more...", "περισσότερα", "περισσότερα...", "read more", "see more"}
_PLACEHOLDERS = {">", "\u00bb", "\u203a", "\u22d9", "\u2026", "...", "scroll", "πλοήγηση", "navigation"}
_GENERIC_NAV = {
    "the city", "the municipality", "culture", "resilient city",
    "ο τόπος μας", "ο δήμος", "πολιτισμός", "ανθεκτική πόλη",
}
_GENERIC_TITLE_RE = re.compile(
    r"^(municipality of .+|heraklion municipality"
    r"|e-?services|online services"
    r"|e-?\u03c5\u03c0\u03b7\u03c1\u03b5\u03c3\u03af\u03b5\u03c2"   # e-υπηρεσίες
    r"|\u03c5\u03c0\u03b7\u03c1\u03b5\u03c3\u03af\u03b5\u03c2"       # υπηρεσίες
    r"|\u03b7\u03bb\u03b5\u03ba\u03c4\u03c1\u03bf\u03bd\u03b9\u03ba\u03ad\u03c2 \u03c5\u03c0\u03b7\u03c1\u03b5\u03c3\u03af\u03b5\u03c2)$",  # ηλεκτρονικές υπηρεσίες
    re.I,
)


def clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def is_bad_anchor(s: str) -> bool:
    t = clean_ws(s)
    if not t or len(t) <= 2:
        return True
    low = t.lower()
    if low in {"gr", "en"}:
        return True
    if low in _MORE_WORDS or low in _PLACEHOLDERS or low in _GENERIC_NAV:
        return True
    # FIX v1.6: only reject short ALL-CAPS *Latin* strings (GR, EN, navigation arrows);
    # Greek acronyms like TAP / KEP / DEYAH must pass through.
    if re.match(r"^[A-Z]{1,4}$", t):
        return True
    return False


def anchor_quality_score(s: str) -> int:
    if is_bad_anchor(s):
        return 0
    t = clean_ws(s)
    score = min(len(t), 40)
    if re.search(r"[A-Za-z\u0391-\u03A9\u03B1-\u03C9]", t):
        score += 10
    return score


def _clean_title_pipes(t: str) -> str:
    """
    FIX v1.6: strip trailing '|' artefacts.
    'Foo | Bar site'  ->  'Foo'
    'Foo |'           ->  'Foo'
    'Foo | Foo |'     ->  'Foo'
    """
    t = re.sub(r"\s*\|.*$", "", t)         # everything after first pipe
    t = re.sub(r"[\|\u2013\u2014\-]+\s*$", "", t)  # trailing separators
    return t.strip()


def extract_meta_titles(soup: BeautifulSoup) -> List[str]:
    raw: List[str] = []
    for m in soup.find_all("meta", attrs={"property": "og:title"}):
        if m.get("content"):
            raw.append(clean_ws(m["content"]))
    for m in soup.find_all("meta", attrs={"name": "twitter:title"}):
        if m.get("content"):
            raw.append(clean_ws(m["content"]))
    if soup.title and soup.title.get_text():
        raw.append(clean_ws(soup.title.get_text()))
    h1 = soup.find("h1")
    if h1:
        raw.append(clean_ws(h1.get_text()))

    cleaned: List[str] = []
    seen: Set[str] = set()
    for t in raw:
        t = _clean_title_pipes(t)
        t = clean_ws(t)
        if t and t.lower() not in {"gr", "en"} and t not in seen:
            cleaned.append(t)
            seen.add(t)
    return cleaned


def _prettify_slug(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    slug = path.split("/")[-1]
    slug = re.sub(r"\.\w+$", "", slug)
    slug = re.sub(r"[-_]", " ", slug)
    return " ".join(
        w.capitalize() if re.search(r"[A-Za-z]", w) else w for w in slug.split()
    )


def is_generic_title(t: str) -> bool:
    return bool(_GENERIC_TITLE_RE.match(clean_ws(t).lower()))


def _fix_greek_latin_mix(s: str) -> str:
    """Replace visually identical Latin letters inside Greek words."""
    _MAP = {"o": "\u03bf", "e": "\u03b5", "a": "\u03b1",
            "O": "\u039f", "E": "\u0395", "A": "\u0391"}

    def fix_word(w: str) -> str:
        has_greek = bool(re.search(r"[\u0391-\u03A9\u03B1-\u03C9]", w))
        has_latin = bool(re.search(r"[A-Za-z]", w))
        if has_greek and has_latin:
            for lat, gr in _MAP.items():
                w = w.replace(lat, gr)
        return w

    return " ".join(fix_word(x) for x in s.split())


def normalize_title_text(t: str) -> str:
    t = clean_ws(t)
    t = re.sub(r"^[>\u00BB\u203A\u2022\-\u2013\u2014]+\s*", "", t)
    # "Αποστολή ψηφιακής αίτησης για 'Χορήγηση...'" → "Χορήγηση..."
    m = re.match(
        r"\u0391\u03c0\u03bf\u03c3\u03c4\u03bf\u03bb\u03ae\s+"   # Αποστολή
        r"\u03c8\u03b7\u03c6\u03b9\u03b1\u03ba\u03ae\u03c2\s+"   # ψηφιακής
        r"\u03b1\u03af\u03c4\u03b7\u03c3\u03b7\u03c2\s+"         # αίτησης
        r"\u03b3\u03b9\u03b1\s+'([^']+)'",                       # για '...'
        t, re.I,
    )
    if m:
        t = m.group(1).strip()
    return _fix_greek_latin_mix(t)


def choose_best_title(
    anchors: List[str], meta_titles: List[str], fallback_url: str
) -> str:
    if anchors:
        best = max(anchors, key=anchor_quality_score)
        best = normalize_title_text(best)
        if anchor_quality_score(best) > 0 and not is_generic_title(best):
            return best
    for t in meta_titles:
        tt = normalize_title_text(t)
        if anchor_quality_score(tt) > 0 and not is_generic_title(tt):
            return tt
    pretty = _prettify_slug(fallback_url)
    return normalize_title_text(pretty or fallback_url)


# ══════════════════════════════════════════════════════════════
# Enrichment extraction  (NEW in v2.0)
# ══════════════════════════════════════════════════════════════
def extract_description(soup: BeautifulSoup) -> str:
    for m in soup.find_all("meta", attrs={"name": re.compile(r"^description$", re.I)}):
        if m.get("content"):
            return clean_ws(m["content"])
    for m in soup.find_all("meta", attrs={"property": "og:description"}):
        if m.get("content"):
            return clean_ws(m["content"])
    return ""


def extract_keywords(soup: BeautifulSoup) -> List[str]:
    for m in soup.find_all("meta", attrs={"name": re.compile(r"^keywords$", re.I)}):
        if m.get("content"):
            return [k.strip() for k in re.split(r"[,;]", m["content"]) if k.strip()]
    return []


def _parent_dir_url(url: str) -> str:
    """Return the normalized parent directory URL, or '' if already at root."""
    p = urlparse(url)
    parts = [s for s in p.path.strip("/").split("/") if s]
    if len(parts) <= 1:
        return ""
    parent_path = "/" + "/".join(parts[:-1]) + "/"
    return urlunparse((p.scheme, p.netloc, parent_path, "", "", ""))


def extract_category(
    soup: BeautifulSoup,
    url: str,
    parent_title_map: Optional[Dict[str, str]] = None,
) -> str:
    """
    Category resolution order (fully generic — no hardcoded labels):
      1. JSON-LD BreadcrumbList  (schema.org)
      2. HTML breadcrumb nav  (<nav|ol|ul class~breadcrumb>)
      3. Parent-page title lookup  — during Pass 1 we record the title of every
         directory-style URL; here we look up the immediate parent.  This gives
         a real, human-readable category label (in whatever language the site
         uses) without any hardcoded mappings.
      4. URL path slug  (second-to-last non-trivial segment) — last resort.
    """
    _SKIP = {
        "home", "e-services", "en", "el",
        # add more trivial root labels here if needed for other sites
    }

    # 1. JSON-LD BreadcrumbList
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            if isinstance(data, dict) and data.get("@type") == "BreadcrumbList":
                items = [
                    clean_ws(item["name"])
                    for item in data.get("itemListElement", [])
                    if item.get("name")
                ]
                meaningful = [t for t in items if t.lower() not in _SKIP]
                if len(meaningful) >= 2:
                    return meaningful[-2]
                if meaningful:
                    return meaningful[0]
        except Exception:
            pass

    # 2. HTML breadcrumb nav
    bc = soup.find(["nav", "ol", "ul"], class_=re.compile(r"breadcrumb", re.I))
    if bc:
        raw_items = [
            clean_ws(el.get_text())
            for el in bc.find_all(["a", "li"])
            if clean_ws(el.get_text())
        ]
        seen_bc: Set[str] = set()
        items = [x for x in raw_items if x not in seen_bc and not seen_bc.add(x)]  # type: ignore
        meaningful = [t for t in items if t.lower() not in _SKIP]
        if len(meaningful) >= 2:
            return meaningful[-2]
        if meaningful:
            return meaningful[0]

    # 3. Parent-page title lookup (generic — built dynamically during Pass 1)
    if parent_title_map:
        parent = _parent_dir_url(url)
        if parent:
            cat = parent_title_map.get(normalize_url(parent), "")
            if cat and cat.lower() not in _SKIP and not is_generic_title(cat):
                return cat

    # 4. URL path slug — last resort
    parts = [
        p for p in urlparse(url).path.strip("/").split("/")
        if p and p.lower() not in _SKIP
    ]
    if len(parts) >= 2:
        return re.sub(r"[-_]", " ", parts[-2]).title()
    return ""


def detect_language(url: str, soup: BeautifulSoup) -> str:
    """Returns 'en' or 'el'."""
    if re.search(r"(?:^|/)en(?:/|$)", urlparse(url).path):
        return "en"
    html_tag = soup.find("html")
    if html_tag:
        lang = (html_tag.get("lang") or "").lower()
        if lang.startswith("en"):
            return "en"
        if lang.startswith("el"):
            return "el"
    return "el"


# ══════════════════════════════════════════════════════════════
# Link discovery
# ══════════════════════════════════════════════════════════════
_REQUEST_RE = re.compile(
    r"(?P<url>(?:https?://(?:www\.)?heraklion\.gr)?/?request\.html\?[^\"'<>\s]+)",
    re.IGNORECASE,
)


def _discover_request_urls(html: str, base_url: str) -> List[str]:
    """Find request.html?... even when not in <a href> (JS, data-attrs, etc.)."""
    raw = htmllib.unescape(html)
    found: Set[str] = set()
    for m in _REQUEST_RE.finditer(raw):
        u = m.group("url")
        if not u.startswith("http"):
            u = urljoin(base_url, u)
        found.add(normalize_url(u))
    return list(found)


def extract_links(
    base_url: str,
    html: str,
    soup: BeautifulSoup,             # passed in - never re-parsed
    allowed_domains: Set[str],
    include_pattern: Optional[str],
) -> List[str]:
    urls: List[str] = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not is_valid_href(href):
            continue
        abs_href = urljoin(base_url, href)
        if not same_host(abs_href, allowed_domains):
            continue
        if include_pattern and not re.match(include_pattern, abs_href):
            continue
        urls.append(abs_href)

    for u in _discover_request_urls(html, base_url):
        if same_host(u, allowed_domains) and matches_include(u, include_pattern):
            urls.append(u)

    return urls


# ══════════════════════════════════════════════════════════════
# Nav-bucket filter  (unchanged from v1.6)
# ══════════════════════════════════════════════════════════════
_NAV_BUCKETS = {"ourplace", "municipality", "culture", "resilient"}


def is_nav_bucket_url(u: str) -> bool:
    parts = [p for p in urlparse(u).path.strip("/").split("/") if p]
    return bool(parts) and parts[-1].lower() in _NAV_BUCKETS


def page_has_substantive_items(soup: BeautifulSoup) -> bool:
    html_str = str(soup)
    if "request.html" in html_str or "wf_route_id=" in html_str:
        return True
    return sum(
        1 for a in soup.find_all("a", href=True)
        if a["href"].lower().endswith(".html") or "request.html" in a["href"].lower()
    ) >= 3


# ══════════════════════════════════════════════════════════════
# Per-host rate limiter  (cooperative, asyncio-safe)
# ══════════════════════════════════════════════════════════════
class HostRateLimiter:
    """Guarantees at least `delay` seconds between requests to the same host."""

    def __init__(self, delay: float) -> None:
        self._delay = delay
        self._locks: Dict[str, asyncio.Lock] = {}
        self._last: Dict[str, float] = defaultdict(float)

    def _lock_for(self, host: str) -> asyncio.Lock:
        if host not in self._locks:
            self._locks[host] = asyncio.Lock()
        return self._locks[host]

    async def wait(self, url: str) -> None:
        host = urlparse(url).netloc
        async with self._lock_for(host):
            elapsed = time.monotonic() - self._last[host]
            if elapsed < self._delay:
                await asyncio.sleep(self._delay - elapsed)
            self._last[host] = time.monotonic()


# ══════════════════════════════════════════════════════════════
# Checkpoint helpers
# ══════════════════════════════════════════════════════════════
def _ckpt_json(out_file: str) -> str:
    base, _ = os.path.splitext(out_file)
    return base + "_checkpoint.json"


def _ckpt_html(out_file: str) -> str:
    base, _ = os.path.splitext(out_file)
    return base + "_htmlstore.json.gz"


def save_checkpoint(
    out_file: str,
    seen_pages: Set[str],
    enqueued: Set[str],
    emitted_urls: Set[str],
    anchors_map: "defaultdict[str, List[str]]",
    html_store: Dict[str, str],
) -> None:
    meta = {
        "seen_pages": list(seen_pages),
        "enqueued": list(enqueued),
        "emitted_urls": list(emitted_urls),
        # Remaining = enqueued but not yet seen (depth info is not persisted)
        "queue_remaining": list(enqueued - seen_pages),
        "anchors_map": dict(anchors_map),
    }
    with open(_ckpt_json(out_file), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    with gzip.open(_ckpt_html(out_file), "wt", encoding="utf-8") as f:
        json.dump(html_store, f, ensure_ascii=False)
    print(f"  [ckpt] {len(seen_pages)} fetched, {len(emitted_urls)} to emit")


def load_checkpoint(out_file: str) -> Tuple[
    Set[str], Set[str], Set[str],
    List[Tuple[str, int]],
    "defaultdict[str, List[str]]",
    Dict[str, str],
]:
    with open(_ckpt_json(out_file), "r", encoding="utf-8") as f:
        raw = json.load(f)

    seen_pages: Set[str] = set(raw["seen_pages"])
    enqueued: Set[str] = set(raw["enqueued"])
    emitted_urls: Set[str] = set(raw["emitted_urls"])
    # Resume at depth=0; seen_pages guard prevents re-fetching
    queue_items: List[Tuple[str, int]] = [(u, 0) for u in raw["queue_remaining"]]
    anchors_map: defaultdict = defaultdict(list, raw["anchors_map"])

    html_store: Dict[str, str] = {}
    if os.path.exists(_ckpt_html(out_file)):
        with gzip.open(_ckpt_html(out_file), "rt", encoding="utf-8") as f:
            html_store = json.load(f)

    print(
        f"[resume] {len(seen_pages)} pages done, "
        f"{len(queue_items)} remaining in queue"
    )
    return seen_pages, enqueued, emitted_urls, queue_items, anchors_map, html_store


# ══════════════════════════════════════════════════════════════
# PASS 1 - Async BFS crawl
# ══════════════════════════════════════════════════════════════
async def pass1_crawl(
    profile: Dict,
    out_file: str,
    expand_depth: int,
    resume: bool,
    concurrency: int,
) -> Tuple[Dict[str, str], "defaultdict[str, List[str]]", Set[str], Dict[str, str]]:
    """
    BFS over the seed URLs.  Returns:
        html_store        - {normalised_url: raw_html}
        anchors_map       - {normalised_url: [anchor_texts, ...]}
        emitted_urls      - set of URLs that should appear in the final output
        parent_title_map  - {directory_url: best_title} built from section pages;
                            used by extract_category (Fix 2) to give real labels
                            instead of greeklish slugs, without any hardcoding.
    """
    allowed = profile["allowed"]
    include_pat = profile.get("include")
    max_depth = min(profile.get("max_depth", 2), expand_depth)
    max_pages = int(profile.get("max_pages", 0))
    delay = float(profile.get("delay", 0.8))

    seeds = [normalize_url(u) for u in profile["seeds"]]
    queue: asyncio.Queue = asyncio.Queue()

    # State initialisation
    if resume and os.path.exists(_ckpt_json(out_file)):
        seen_pages, enqueued, emitted_urls, queue_items, anchors_map, html_store = (
            load_checkpoint(out_file)
        )
        for u, d in queue_items:
            await queue.put((u, d))
    else:
        seen_pages: Set[str] = set()
        enqueued: Set[str] = set(seeds)
        emitted_urls: Set[str] = set()
        anchors_map: "defaultdict[str, List[str]]" = defaultdict(list)
        html_store: Dict[str, str] = {}
        for s in seeds:
            await queue.put((s, 0))

    # Maps directory-style URLs → their resolved title (built as we crawl).
    # Used by extract_category as priority-3 fallback (generic, no hardcoding).
    parent_title_map: Dict[str, str] = {}

    rate_limiter = HostRateLimiter(delay)
    fetched_count: List[int] = [len(seen_pages)]   # mutable int via list

    connector = aiohttp.TCPConnector(limit=concurrency * 3, ttl_dns_cache=300)
    session_headers = {"User-Agent": UA, "Accept-Language": "el,en;q=0.9"}

    async with aiohttp.ClientSession(
        headers=session_headers, connector=connector
    ) as session:

        # Pre-fetch robots.txt for every allowed host
        robots_cache: Dict[str, urobot.RobotFileParser] = {}
        for host in allowed:
            robots_cache[host] = await fetch_robots(session, f"https://{host}/")
            print(f"  [robots] loaded for {host}")

        # Worker coroutine
        async def worker() -> None:
            while True:
                try:
                    url, depth = await asyncio.wait_for(queue.get(), timeout=3.0)
                except asyncio.TimeoutError:
                    break

                try:
                    nurl = normalize_url(url)

                    if nurl in seen_pages:
                        continue
                    if max_pages and fetched_count[0] >= max_pages:
                        continue

                    host = urlparse(nurl).netloc
                    rp = robots_cache.get(host)
                    if rp and not allowed_by_robots(rp, nurl):
                        continue

                    await rate_limiter.wait(nurl)
                    html, _code = await fetch_html_async(session, nurl)

                    if html is None:
                        continue

                    soup = BeautifulSoup(html, "html.parser")   # single parse per page

                    if looks_soft_404(soup):
                        continue

                    # Record this page
                    seen_pages.add(nurl)
                    html_store[nurl] = html
                    fetched_count[0] += 1

                    if fetched_count[0] % 10 == 0:
                        print(
                            f"  [pass1] {fetched_count[0]} fetched | "
                            f"queue~{queue.qsize()} | {len(emitted_urls)} to emit"
                        )

                    # Collect anchor texts pointing at child URLs
                    for a in soup.find_all("a", href=True):
                        href_abs = urljoin(nurl, a["href"].strip())
                        if not is_valid_href(href_abs):
                            continue
                        href_norm = normalize_url(href_abs)
                        if same_host(href_norm, allowed) and matches_include(
                            href_norm, include_pat
                        ):
                            atext = clean_ws(a.get_text(" ", strip=True))
                            if anchor_quality_score(atext) > 0:
                                anchors_map[href_norm].append(atext)

                    # Enqueue children BEFORE task_done (keeps queue.join() correct)
                    if depth + 1 <= max_depth:
                        for child in extract_links(
                            nurl, html, soup, allowed, include_pat
                        ):
                            c = normalize_url(child)
                            if c not in enqueued:
                                enqueued.add(c)
                                await queue.put((c, depth + 1))

                    # Mark for emission
                    if same_host(nurl, allowed) and matches_include(nurl, include_pat):
                        if not (
                            is_nav_bucket_url(nurl)
                            and not page_has_substantive_items(soup)
                        ):
                            emitted_urls.add(nurl)

                    # Fix 2: if this is a directory-style URL, record its title
                    # so child pages can use it as their category label.
                    if nurl.endswith("/"):
                        meta_titles = extract_meta_titles(soup)
                        pt = choose_best_title(anchors_map.get(nurl, []), meta_titles, nurl)
                        pt = normalize_title_text(pt)
                        if pt and not is_generic_title(pt):
                            parent_title_map[nurl] = pt

                    # Periodic checkpoint
                    if fetched_count[0] % CHECKPOINT_INTERVAL == 0:
                        save_checkpoint(
                            out_file,
                            seen_pages, enqueued, emitted_urls,
                            anchors_map, html_store,
                        )

                except Exception as exc:
                    print(f"  [!] {url}: {exc}")
                finally:
                    queue.task_done()

        # Launch workers; queue.join() waits until every task_done() is called
        tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await queue.join()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    print(
        f"[pass1] complete: {fetched_count[0]} fetched, {len(emitted_urls)} to emit"
    )
    return html_store, anchors_map, emitted_urls, parent_title_map



# ══════════════════════════════════════════════════════════════
# PASS 2 - Title & enrichment resolution
# ══════════════════════════════════════════════════════════════
def pass2_resolve(
    html_store: Dict[str, str],
    anchors_map: "defaultdict[str, List[str]]",
    emitted_urls: Set[str],
    parent_title_map: Dict[str, str],
    prefer_domain: Optional[str] = None,
) -> List[Dict]:
    """
    Three post-processing steps after BFS completes:

    Step 1 — Title resolution
        anchors_map is now COMPLETE so every page gets the best possible title.

    Step 2 — Iterative pollution filter
        Run the frequency filter in a loop until convergence (no new polluted
        titles found). Handles cascading template strings: after demoting the
        first polluted anchor the re-resolved title may itself be a template.
        Fully generic — no hardcoded strings.

    Step 3 — Cross-domain deduplication by path
        www.heraklion.gr and eservices.heraklion.gr share identical paths.
        For each duplicated path keep the record with the better title.
        Prefers eservices records as they tend to have cleaner anchor texts.
    """
    from collections import Counter
    from urllib.parse import urlparse as _urlparse

    # ── Step 1: resolve titles normally ──────────────────────────────────────
    records = []
    for nurl in emitted_urls:
        html = html_store.get(nurl)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        meta_titles = extract_meta_titles(soup)
        title = normalize_title_text(
            choose_best_title(anchors_map.get(nurl, []), meta_titles, nurl)
        )
        records.append({
            "url": nurl,
            "title": title,
            "_meta_titles": meta_titles,
            "_anchors": list(anchors_map.get(nurl, [])),
            "description": extract_description(soup),
            "keywords": extract_keywords(soup),
            "category": extract_category(soup, nurl, parent_title_map),
            "language": detect_language(nurl, soup),
        })

    # ── Step 2: iterative pollution filter ───────────────────────────────────
    all_polluted: set = set()
    iteration = 0
    while True:
        iteration += 1
        freq = Counter(r["title"] for r in records)
        newly_polluted = {
            t for t, c in freq.items()
            if c >= TITLE_FREQ_THRESHOLD and t not in all_polluted
        }
        if not newly_polluted:
            break
        all_polluted |= newly_polluted
        print(
            f"[pass2] pollution iter {iteration}: {len(newly_polluted)} template(s) — "
            + ", ".join(f'"{t[:60]}"' for t in list(newly_polluted)[:2])
        )
        for r in records:
            if r["title"] in all_polluted:
                clean_anchors = [
                    a for a in r["_anchors"]
                    if normalize_title_text(a) not in all_polluted
                ]
                r["title"] = normalize_title_text(
                    choose_best_title(clean_anchors, r["_meta_titles"], r["url"])
                )

    if all_polluted:
        print(
            f"[pass2] pollution filter done: {len(all_polluted)} template(s) "
            f"removed in {iteration - 1} iteration(s)"
        )

    # ── Step 3: cross-domain deduplication by path ────────────────────────────
    # Αν το profile ορίζει prefer_domain, το record από αυτό το domain
    # κερδίζει πάντα — ανεξάρτητα από τον τίτλο. Αυτό διασφαλίζει ότι
    # δεν κρατάμε ποτέ ένα www.heraklion.gr link όταν υπάρχει το
    # αντίστοιχο eservices.heraklion.gr (που είναι το κανονικό, έγκυρο).
    def title_score(r: dict) -> int:
        t = r["title"]
        score = len(t)
        if any(ord(c) > 127 for c in t):   # prefer Greek over Latin slugs
            score += 100
        if t.startswith("http"):            # penalise URL-as-title
            score -= 500
        # Μεγάλο bonus αν είναι το preferred domain
        if prefer_domain and _urlparse(r["url"]).netloc == prefer_domain:
            score += 10000
        return score

    path_best: Dict[str, dict] = {}
    for r in records:
        path = _urlparse(r["url"]).path
        if path not in path_best or title_score(r) > title_score(path_best[path]):
            path_best[path] = r

    deduped = list(path_best.values())
    removed = len(records) - len(deduped)
    print(f"[pass2] deduplication: {len(records)} → {len(deduped)} records ({removed} duplicates removed)")

    # ── Strip internal fields and sort ───────────────────────────────────────
    results = [
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in deduped
    ]
    results.sort(key=lambda r: (r["language"] != "el", r["category"], r["url"]))
    print(f"[pass2] resolved {len(results)} records")
    return results



# ══════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════
def crawl(
    profile: Dict,
    out_file: str,
    mode: str,
    expand_depth: int,
    resume: bool,
    concurrency: int,
) -> None:
    html_store, anchors_map, emitted_urls, parent_title_map = asyncio.run(
        pass1_crawl(profile, out_file, expand_depth, resume, concurrency)
    )
    results = pass2_resolve(
        html_store, anchors_map, emitted_urls, parent_title_map,
        prefer_domain=profile.get("prefer_domain"),
    )

    out_dir = os.path.dirname(out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[+] Saved {len(results)} records -> {out_file}")


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Heraklion e-Services Crawler v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 crawler_script.py "
            "--profile heraklion_eservices --out data/out.json\n"
            "  python3 crawler_script.py "
            "--profile heraklion_eservices --out data/out.json "
            "--concurrency 6 --resume"
        ),
    )
    ap.add_argument("--profile", required=True, choices=list(PROFILES.keys()))
    ap.add_argument("--out", required=True, help="Output JSON file")
    ap.add_argument("--mode", default="default", help="default | sitemap-expand")
    ap.add_argument("--expand-depth", type=int, default=2, dest="expand_depth")
    ap.add_argument(
        "--concurrency", type=int, default=4,
        help="Concurrent async workers (default: 4)",
    )
    ap.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint",
    )
    args = ap.parse_args()

    crawl(
        PROFILES[args.profile],
        args.out,
        args.mode,
        args.expand_depth,
        args.resume,
        args.concurrency,
    )


if __name__ == "__main__":
    main()
