"""
KB Enrichment — προσθήκη body text στο heraklion_eservices.json
================================================================
Ο crawler αποθηκεύει ήδη όλο το HTML στο `*_htmlstore.json.gz`, αλλά το τελικό
JSON κρατά μόνο τίτλο (τα description/keywords είναι κενά σε 167/167 records,
γιατί ο ιστότοπος δεν έχει meta tags).

Αυτό το script ξαναδιαβάζει το αποθηκευμένο HTML — ΧΩΡΙΣ νέο crawl — και
προσθέτει σε κάθε record:
    body        το καθαρό κείμενο της σελίδας
    online      False αν η σελίδα δηλώνει ότι απαιτείται φυσική παρουσία
    contact     το τμήμα επικοινωνίας, αν εντοπιστεί

Αφαίρεση boilerplate:
    Προτάσεις που εμφανίζονται σε >BOILERPLATE_RATIO των σελίδων θεωρούνται
    template και κόβονται. Ίδια γενική λογική με το TITLE_FREQ_THRESHOLD του
    crawler — κανένα hardcoded string για συγκεκριμένο site.

Χρήση:
    python3 enrich_kb.py \
        --kb data/heraklion_eservices.json \
        --htmlstore data/heraklion_eservices_htmlstore.json.gz \
        --out data/heraklion_eservices_enriched.json
"""

import argparse
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup
from paths import ROOT, DATA, DATASETS, MODELS, MAPPINGS, MODEL_218, KB, KB_RICH

# Πάνω από αυτό το ποσοστό σελίδων → η πρόταση είναι template.
# Στο heraklion.gr το πιο συχνό boilerplate φτάνει μόλις στο 18% (τηλέφωνα και
# διευθύνσεις τμημάτων που εξυπηρετούν πολλές υπηρεσίες), οπότε ένα κατώφλι
# 0.30 δεν έπιανε τίποτα. Τα στοιχεία επικοινωνίας διατηρούνται ούτως ή άλλως
# στο πεδίο `contact` — στο `body` είναι μόνο θόρυβος για το retrieval.
BOILERPLATE_RATIO = 0.06

# Ελάχιστο μήκος πρότασης για να ληφθεί υπόψη
MIN_SENTENCE_LEN = 25

# Δηλώνει ότι η υπηρεσία ΔΕΝ διεκπεραιώνεται ηλεκτρονικά
_OFFLINE_RE = re.compile(
    r"αποκλειστικ\w*\s+με\s+φυσικ\w*\s+παρουσία"
    r"|μόνο\s+με\s+φυσικ\w*\s+παρουσία"
    r"|αυτοπροσώπως\s+στα\s+γραφεία",
    re.I,
)

_CONTACT_RE = re.compile(r"Στοιχεία\s+επικοινωνίας(.{0,400})", re.I | re.S)

_STRIP_TAGS = ["script", "style", "nav", "header", "footer", "noscript", "svg"]


def page_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_STRIP_TAGS):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()


def split_sentences(text: str) -> list:
    parts = re.split(r"(?<=[.;:!?])\s+|\s{2,}", text)
    return [p.strip() for p in parts if len(p.strip()) >= MIN_SENTENCE_LEN]


def find_boilerplate(texts: list) -> set:
    """Προτάσεις που επαναλαμβάνονται σε πολλές σελίδες → template."""
    freq = Counter()
    for t in texts:
        freq.update(set(split_sentences(t)))
    cutoff = max(2, int(len(texts) * BOILERPLATE_RATIO))
    return {s for s, c in freq.items() if c >= cutoff}


def clean_body(text: str, boilerplate: set, title: str) -> str:
    kept = [s for s in split_sentences(text) if s not in boilerplate]
    body = " ".join(kept)
    # Ο τίτλος επαναλαμβάνεται συνήθως στην αρχή του σώματος
    if title and body.lower().startswith(title.lower()):
        body = body[len(title):].strip()
    return body


def extract_contact(text: str) -> str:
    """
    Κρατά την ΤΕΛΕΥΤΑΙΑ εμφάνιση: οι σελίδες ξεκινούν συχνά με «Δείτε παρακάτω
    τα στοιχεία επικοινωνίας», που ταιριάζει στο pattern αλλά ακολουθείται από
    τα δικαιολογητικά — όχι από τη διεύθυνση.
    """
    matches = list(_CONTACT_RE.finditer(text))
    if not matches:
        return ""
    return re.sub(r"\s+", " ", matches[-1].group(1)).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Εμπλουτισμός KB με body text")
    ap.add_argument("--kb", default=str(KB))
    ap.add_argument("--htmlstore", default=str(DATA / "heraklion_eservices_htmlstore.json.gz"))
    ap.add_argument("--out", default=str(KB_RICH))
    ap.add_argument("--max-body", type=int, default=2000,
                    help="Μέγιστοι χαρακτήρες body ανά record")
    args = ap.parse_args()

    kb_path, hs_path = Path(args.kb), Path(args.htmlstore)
    for p in (kb_path, hs_path):
        if not p.exists():
            sys.exit(f"Δεν βρέθηκε: {p}")

    kb = json.loads(kb_path.read_text(encoding="utf-8"))
    with gzip.open(hs_path, "rt", encoding="utf-8") as f:
        store = json.load(f)

    print(f"📚 {len(kb)} records  |  {len(store)} αποθηκευμένες σελίδες")

    raw = {}
    for r in kb:
        html = store.get(r["url"])
        if html:
            raw[r["url"]] = page_text(html)

    missing = len(kb) - len(raw)
    if missing:
        print(f"  ⚠  {missing} records χωρίς αποθηκευμένο HTML")

    boilerplate = find_boilerplate(list(raw.values()))
    print(f"🧹 {len(boilerplate)} template προτάσεις εντοπίστηκαν και αφαιρέθηκαν")

    n_offline, n_contact, empty = 0, 0, 0
    for r in kb:
        text = raw.get(r["url"], "")
        body = clean_body(text, boilerplate, r.get("title", ""))

        r["body"] = body[:args.max_body]
        r["online"] = not bool(_OFFLINE_RE.search(text))
        r["contact"] = extract_contact(text)

        if not r["online"]:
            n_offline += 1
        if r["contact"]:
            n_contact += 1
        if not body:
            empty += 1

    Path(args.out).write_text(
        json.dumps(kb, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lengths = sorted(len(r["body"]) for r in kb)
    print(f"\n  Διάμεσο μήκος body : {lengths[len(lengths) // 2]} χαρακτήρες")
    print(f"  Κενό body          : {empty}")
    print(f"  Με στοιχεία επικ.  : {n_contact}")
    print(f"  ΜΟΝΟ με φυσική παρουσία : {n_offline} / {len(kb)}"
          f"  ← ο βοηθός πρέπει να το λέει ρητά")
    print(f"\n💾 {args.out}")


if __name__ == "__main__":
    main()
