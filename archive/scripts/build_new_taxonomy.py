"""
Νέα ταξινομία intents — πλήρης κάλυψη υπηρεσιών Δήμου
======================================================
Συνδυάζει την ανάγνωση του PDF με το crawled KB και παράγει την ΠΡΟΤΕΙΝΟΜΕΝΗ
τελική λίστα intents, όπου:

    ΚΡΑΤΑ    intent που αντιστοιχεί σε υπάρχουσα υπηρεσία
    ΝΕΟ      intent για κόκκινη υπηρεσία (δεν υπήρχε στο dataset)
    ΣΒΗΣΕ    intent χωρίς αντίστοιχη υπηρεσία στον Δήμο
    ΕΛΕΓΞΕ   χαλαρή αντιστοίχιση — πολλά intents στην ίδια υπηρεσία

Τα ονόματα των νέων intents παράγονται από τον τίτλο της υπηρεσίας με τη
σύμβαση του υπάρχοντος dataset: πεζά ελληνικά με τόνους, κάτω παύλες, χωρίς
το εισαγωγικό ρήμα («Χορήγηση», «Έκδοση», «Αίτηση για»).

ΤΑ ΟΝΟΜΑΤΑ ΕΙΝΑΙ ΠΡΟΤΑΣΕΙΣ — ελέγξτε τα πριν την παραγωγή προτάσεων.

Χρήση:
    python3 build_new_taxonomy.py --out proposed_taxonomy.csv
"""

import argparse
import csv
import difflib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from clean_dataset import canon_label, strip_accents
from paths import ROOT, DATA, DATASETS, MODELS, MAPPINGS, MODEL_218, KB, KB_RICH

# Ρήματα/φράσεις που ξεκινούν σχεδόν κάθε τίτλο υπηρεσίας και δεν προσθέτουν
# διακριτική πληροφορία στο όνομα του intent
_LEAD = re.compile(
    r"^(χορήγηση(\s+άδειας)?|έκδοση(\s+αδειών)?|αίτηση(\s+για)?|αίτημα(\s+για)?"
    r"|αιτήματα(\s+για)?|δικαιολογητικά(\s+χορήγησης)?|διαδικασία(\s+έκδοσης)?"
    r"|κατάθεση|υποβολή|θέματα)\s+",
    re.I,
)

_STOP = {"του", "της", "των", "στο", "στη", "στην", "στους", "στις", "σε",
         "για", "και", "ή", "με", "από", "το", "η", "ο", "οι", "τα", "τη", "την"}

MAX_WORDS = 5


def _words(text: str) -> list:
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return [w for w in text.split() if w and w not in _STOP and not w.isdigit()]


def slug(title: str) -> str:
    """
    Τίτλος υπηρεσίας → όνομα intent, με τη σύμβαση του dataset.

    Το περιεχόμενο των παρενθέσεων κανονικά πετιέται, ΑΛΛΑ κρατιέται όταν το
    όνομα θα έβγαινε πολύ γενικό: «Καταγγελίες (Εγκαταλελειμμένα κτίρια)» δεν
    πρέπει να γίνει σκέτο `καταγγελίες`, γιατί υπάρχουν πέντε τέτοιες.
    """
    paren = " ".join(re.findall(r"\(([^)]*)\)", title))
    base = re.sub(r"\([^)]*\)", " ", title)
    base = re.sub(r"\s*[-–—]\s*", " ", base)     # παύλες → κενό
    base = re.sub(r"\s+", " ", base).strip().lower()
    base = _LEAD.sub("", base).strip()

    words = _words(base)
    if len(words) < 3 and paren:
        words += _words(paren)
    if not words:
        words = _words(title)
    return "_".join(words[:MAX_WORDS]).strip("_")


def norm_title(t: str) -> str:
    t = re.sub(r"\([^)]*\)", " ", t)
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    return re.sub(r"\s+", " ", strip_accents(t)).strip().lower()


def match_url(title: str, kb_index: dict, kb_titles: list) -> tuple:
    """Επιστρέφει (url, τίτλος_kb, τρόπος) ή ('', '', 'δεν βρέθηκε')."""
    key = norm_title(title)
    if key in kb_index:
        r = kb_index[key]
        return r["url"], r["title"], "ακριβές"
    close = difflib.get_close_matches(key, kb_titles, n=1, cutoff=0.82)
    if close:
        r = kb_index[close[0]]
        return r["url"], r["title"], "προσεγγιστικό"
    return "", "", "δεν βρέθηκε"


def main() -> None:
    ap = argparse.ArgumentParser(description="Παραγωγή νέας ταξινομίας intents")
    ap.add_argument("--pdf-intents", default="pdf_intents.csv")
    ap.add_argument("--pdf-services", default="pdf_services.csv")
    ap.add_argument("--kb", default=str(KB_RICH))
    ap.add_argument("--out", default="proposed_taxonomy.csv")
    args = ap.parse_args()

    for p in (args.pdf_intents, args.pdf_services):
        if not Path(p).exists():
            sys.exit(f"Δεν βρέθηκε: {p}  (τρέξε πρώτα parse_pdf_mapping.py)")

    intents = list(csv.DictReader(open(args.pdf_intents, encoding="utf-8-sig")))
    services = list(csv.DictReader(open(args.pdf_services, encoding="utf-8-sig")))

    kb = json.loads(Path(args.kb).read_text(encoding="utf-8")) \
        if Path(args.kb).exists() else []
    kb_index = {norm_title(r["title"]): r for r in kb}
    kb_titles = list(kb_index)

    # Ποια intents εμφανίζονται ως σχόλιο σε κάποια υπηρεσία
    annotated = defaultdict(list)
    for s in services:
        for i in (x.strip() for x in s["intents"].split(",")):
            if i:
                annotated[canon_label(i)].append(s)

    rows, used_names = [], set()

    # ── 1. Υπάρχοντα intents ────────────────────────────────────────────────
    for it in intents:
        name = canon_label(it["intent"])
        used_names.add(name)
        hits = annotated.get(name, [])

        if not hits:
            rows.append({
                "intent": it["intent"], "ενέργεια": "ΣΒΗΣΕ",
                "υπηρεσία": "", "url": "", "ενότητα": "",
                "σημείωση": "δεν αντιστοιχεί σε υπηρεσία του Δήμου",
            })
            continue

        svc = hits[0]
        url, kb_title, how = match_url(svc["υπηρεσία"], kb_index, kb_titles)
        loose = svc["χαλαρή"] == "ναι"
        shared = len([x for x in svc["intents"].split(",") if x.strip()]) > 1

        note = []
        if not svc["υπηρεσία"].strip():
            note.append("ΧΩΡΙΣ ΥΠΗΡΕΣΙΑ στο PDF — χρειάζεται URL")
        elif svc["υπηρεσία"].startswith("[ΕΝΟΤΗΤΑ]"):
            note.append("δείχνει σε ΟΛΗ την ενότητα — διάλεξε συγκεκριμένη σελίδα")
        if loose:
            note.append("χαλαρή αντιστοίχιση")
        if shared:
            note.append(f"μοιράζεται υπηρεσία με {shared and len(svc['intents'].split(',')) - 1} άλλα")
        if len(hits) > 1:
            note.append(f"εμφανίζεται σε {len(hits)} υπηρεσίες")
        if how != "ακριβές":
            note.append(f"URL: {how}")

        rows.append({
            "intent": it["intent"],
            "ενέργεια": "ΕΛΕΓΞΕ" if (loose or shared or len(hits) > 1) else "ΚΡΑΤΑ",
            "υπηρεσία": svc["υπηρεσία"],
            "url": url,
            "ενότητα": svc["ενότητα"],
            "σημείωση": "· ".join(note),
        })

    # ── 2. Νέα intents για κάθε υπηρεσία χωρίς καταγεγραμμένο intent ────────
    # Το κριτήριο ΔΕΝ είναι το χρώμα. 44 «μπλε» υπηρεσίες (= υποτίθεται έχουν
    # intent) δεν έχουν κανένα σημειωμένο δίπλα τους, και για πολλές δεν
    # υπάρχει καν αντίστοιχο intent στη λίστα των 101 — π.χ. «Μεταδημότευση
    # αγάμου». Αφού ζητούμενο είναι πλήρης κάλυψη, ό,τι δεν έχει intent
    # παίρνει καινούργιο.
    for s in services:
        if s["intents"].strip():
            continue
        title = s["υπηρεσία"]
        if not title or title.startswith("[ΕΝΟΤΗΤΑ]"):
            continue

        base = slug(title)
        if not base:
            continue
        name, n = base, 2
        while name in used_names:
            name, n = f"{base}_{n}", n + 1
        used_names.add(name)

        url, kb_title, how = match_url(title, kb_index, kb_titles)
        note = [] if how == "ακριβές" else [f"URL: {how}"]
        if s["έχει_intent"] == "ναι":
            note.append("ΜΠΛΕ στο PDF αλλά χωρίς σημειωμένο intent — επιβεβαίωσε")

        rows.append({
            "intent": name, "ενέργεια": "ΝΕΟ",
            "υπηρεσία": title, "url": url,
            "ενότητα": s["ενότητα"],
            "σημείωση": "· ".join(note),
        })

    # ── 3. Μήπως ένα ΝΕΟ είναι στην ουσία ένα ΣΒΗΣΕ με άλλο όνομα; ──────────
    # Αν ναι, καλύτερα να επαναχρησιμοποιηθεί το παλιό intent: κρατά τα
    # ~70 δείγματα εκπαίδευσης που ήδη υπάρχουν, αντί να γραφτούν από την αρχή.
    def stems(text: str) -> set:
        """
        Σύνολο «ριζών»: πρώτοι 5 χαρακτήρες κάθε λέξης, χωρίς τόνους.

        Η σύγκριση χαρακτήρα-προς-χαρακτήρα (difflib) έβγαζε σκουπίδια —
        «καθαριότητας» ≈ «αρτιότητας» με 0.6 ομοιότητα, ενώ δεν έχουν σχέση.
        Η επικάλυψη λέξεων απορρίπτει αυτά και πιάνει τα πραγματικά ζεύγη
        («δημοτική_ενημερότητα» ↔ «Έκδοση δημοτικής ενημερότητας»).
        """
        out = set()
        for w in _words(strip_accents(text).lower().replace("_", " ")):
            if len(w) >= 4:
                out.add(w[:5])
        return out

    deleted = [r for r in rows if r["ενέργεια"] == "ΣΒΗΣΕ"]

    for r in rows:
        if r["ενέργεια"] != "ΝΕΟ":
            continue
        cand = stems(r["intent"]) | stems(r["υπηρεσία"])
        best, best_score = None, 0.0
        for old_row in deleted:
            s = stems(old_row["intent"])
            # Μονολεκτικά intents ταιριάζουν με τα πάντα: το «παλαιότητα_1955»
            # (μόνη ρίζα «παλαι») κόλλησε στο «…της Παλαιάς Πόλης».
            if len(s) < 2:
                continue
            score = len(s & cand) / len(s)
            if score > best_score:
                best, best_score = old_row, score

        if best is not None and best_score >= 0.7:
            old = best
            note = f"ΙΔΙΟ ΜΕ «{old['intent']}» — επαναχρησιμοποίησέ το"
            r["σημείωση"] = f"{r['σημείωση']}· {note}" if r["σημείωση"] else note
            old["σημείωση"] = (f"μάλλον αντιστοιχεί στη νέα υπηρεσία "
                               f"«{r['υπηρεσία'][:50]}»")
            old["ενέργεια"] = "ΕΛΕΓΞΕ"

    order = {"ΝΕΟ": 0, "ΕΛΕΓΞΕ": 1, "ΣΒΗΣΕ": 2, "ΚΡΑΤΑ": 3}
    rows.sort(key=lambda r: (order[r["ενέργεια"]], r["ενότητα"], r["intent"]))

    fields = ["intent", "ενέργεια", "υπηρεσία", "url", "ενότητα", "σημείωση"]
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)

    counts = {k: sum(1 for r in rows if r["ενέργεια"] == k) for k in order}
    no_url = sum(1 for r in rows if not r["url"] and r["ενέργεια"] != "ΣΒΗΣΕ")
    final = counts["ΚΡΑΤΑ"] + counts["ΕΛΕΓΞΕ"] + counts["ΝΕΟ"]

    print("═" * 66)
    print("  ΠΡΟΤΕΙΝΟΜΕΝΗ ΤΑΞΙΝΟΜΙΑ")
    print("═" * 66)
    print(f"  ΚΡΑΤΑ  : {counts['ΚΡΑΤΑ']:3}")
    print(f"  ΕΛΕΓΞΕ : {counts['ΕΛΕΓΞΕ']:3}   πολλαπλά/χαλαρά — θέλουν απόφαση")
    print(f"  ΝΕΟ    : {counts['ΝΕΟ']:3}   νέα intents προς παραγωγή προτάσεων")
    print(f"  ΣΒΗΣΕ  : {counts['ΣΒΗΣΕ']:3}")
    print(f"\n  ΤΕΛΙΚΟ ΠΛΗΘΟΣ INTENTS : {final}")
    print(f"  Χωρίς URL στο KB      : {no_url}   ← ο crawler δεν τα βρήκε")
    print(f"\n  💾 {args.out}")
    print("═" * 66)
    print("\n  ⚠  Τα ονόματα των ΝΕΟ είναι αυτόματα — ελέγξτε τα πριν προχωρήσουμε.")


if __name__ == "__main__":
    main()
