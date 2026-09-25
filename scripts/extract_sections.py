"""
Εξαγωγή δομημένων ενοτήτων από τις σελίδες υπηρεσιών του δήμου.

ΓΙΑΤΙ: σήμερα το bot απαντά «ποια υπηρεσία» και δίνει σύνδεσμο. Ο
πολίτης όμως ρωτάει και «τι χαρτιά χρειάζομαι;» — και η απάντηση υπάρχει
ήδη στη σελίδα, σε σταθερή ενότητα «Τι χρειάζεται:» (127/164 σελίδες).
Κόβοντάς την αυτούσια απαντάμε χωρίς γλωσσικό μοντέλο, άρα χωρίς
δυνατότητα παραποίησης.

ΔΟΜΗ ΣΕΛΙΔΑΣ (σύστημα σχεδίασης gov.gr):
    <main> → <h1>τίτλος</h1>
             <p>περιγραφή</p>
             <p><strong>Τι χρειάζεται:</strong></p>
             <ul><li>…</li></ul>

Επικεφαλίδα θεωρείται <strong>/<b>/<h2-4> που αποτελεί ΟΛΟ το κείμενο του
μπλοκ του. Έντονο κείμενο μέσα σε πρόταση είναι έμφαση, όχι επικεφαλίδα.

Χρήση:  python3 scripts/extract_sections.py
Έξοδος: data/service_sections.json
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from collections import Counter

from bs4 import BeautifulSoup, NavigableString, Tag

from paths import DATA, ROOT

HTMLSTORE = DATA / "heraklion_eservices_htmlstore.json.gz"
OUT = DATA / "service_sections.json"

BLOCK = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "td"}
HEADING_MAX = 70

# Επικεφαλίδες που ΔΕΝ είναι σε έντονα γράμματα στις σελίδες του δήμου.
# Μια πρώτη εκδοχή αναγνώριζε μόνο bold, και τα στοιχεία επικοινωνίας
# κατέληγαν μέσα στην ενότητα δικαιολογητικών — ο πολίτης που ρωτούσε
# «τι χαρτιά χρειάζομαι» θα έπαιρνε και τηλέφωνα και διευθύνσεις.
KNOWN_HEADINGS = re.compile(
    r"^(στοιχεία επικοινωνίας|χρήσιμοι σύνδεσμοι|χρήσιμες πληροφορίες|"
    r"προσοχή|σημείωση|σημαντικό|υπόχρεοι|προθεσμία|κόστος|"
    r"τι χρειάζεται|τι θα χρειαστείτε|απαιτούμενα δικαιολογητικά|"
    r"εναλλακτική υποβολή)", re.I)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def heading_of(block: Tag) -> str | None:
    """Επιστρέφει το κείμενο αν το μπλοκ είναι επικεφαλίδα, αλλιώς None."""
    if block.name in ("h2", "h3", "h4", "h5"):
        t = clean(block.get_text(" "))
        return t if 0 < len(t) <= HEADING_MAX else None
    if block.name not in ("p", "div"):
        return None
    full = clean(block.get_text(" "))
    if not full or len(full) > HEADING_MAX:
        return None
    # 1. γνωστή επικεφαλίδα, όποια κι αν είναι η μορφοποίηση
    if KNOWN_HEADINGS.match(full):
        return full
    # 2. ολόκληρο το μπλοκ σε έντονα
    bold = " ".join(clean(b.get_text(" ")) for b in block.find_all(["strong", "b"]))
    if clean(bold) == full:
        return full
    # 3. σύντομη ετικέτα που τελειώνει σε «:» ή ολόκληρη σε κεφαλαία
    if len(full) <= 40 and (full.endswith(":") or (full.isupper() and len(full) > 3)):
        return full
    return None


def extract(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main")
    if main is None:
        return {}
    for junk in main.find_all(["script", "style", "form", "button", "nav"]):
        junk.decompose()

    title_tag = main.find("h1")
    title = clean(title_tag.get_text(" ")) if title_tag else ""
    if title_tag:
        title_tag.decompose()

    sections: dict = {}
    current = "Περιγραφή"
    seen = set()

    for el in main.descendants:
        if not isinstance(el, Tag) or el.name not in BLOCK:
            continue
        # μόνο «φύλλα» μπλοκ: αν περιέχει άλλο μπλοκ, θα το επεξεργαστούμε εκεί
        if el.name != "li" and el.find(list(BLOCK - {"li"})):
            continue
        if id(el) in seen:
            continue
        seen.add(id(el))

        h = heading_of(el)
        if h:
            current = h.rstrip(":").strip()
            continue

        text = clean(el.get_text(" "))
        if not text:
            continue

        # Επικεφαλίδα ΚΑΙ περιεχόμενο στο ίδιο μπλοκ: «Στοιχεία επικοινωνίας
        # Τμήμα Παλιάς Πόλης Τηλ: …». Είναι πολύ μακρύ για να περάσει ως
        # επικεφαλίδα, και χωρίς αυτόν τον διαχωρισμό έπεφτε ολόκληρο στην
        # προηγούμενη ενότητα (13 σελίδες, κυρίως ληξιαρχείο και Παλιά Πόλη).
        m = KNOWN_HEADINGS.match(text)
        if m and el.name != "li":
            current = m.group(0).rstrip(":").strip().capitalize()
            text = text[m.end():].lstrip(" :")
            if not text:
                continue
        if el.name == "li":
            text = "• " + text
        elif el.find_parent("li"):
            continue          # το κείμενο θα έρθει μέσω του li
        sections.setdefault(current, [])
        if text not in sections[current]:
            sections[current].append(text)

    return {"title": title,
            "sections": {k: "\n".join(v) for k, v in sections.items() if v}}


def main() -> int:
    store = json.load(gzip.open(HTMLSTORE, "rt", encoding="utf-8"))
    pages = {u: h for u, h in store.items() if "eservices.heraklion.gr" in u}
    out, heads, empty = {}, Counter(), []

    for url, html in sorted(pages.items()):
        rec = extract(html)
        if not rec.get("sections"):
            empty.append(url)
            continue
        out[url] = rec
        heads.update(rec["sections"].keys())

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"  σελίδες e-services       : {len(pages)}")
    print(f"  με εξαγόμενες ενότητες   : {len(out)}")
    print(f"  χωρίς περιεχόμενο        : {len(empty)}")
    print(f"\n  συχνότερες ενότητες:")
    for h, n in heads.most_common(15):
        print(f"    {n:4}  {h}")
    print(f"\n  💾 {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
