"""
Προέλεγχος του πίνακα intent → υπηρεσία: ποιες γραμμές `auto` / `proposed`
να ελέγξει ΠΡΩΤΑ η ομάδα.

ΓΙΑΤΙ: οι 127 auto γραμμές φτιάχτηκαν με TF-IDF πάνω στο ΟΝΟΜΑ του intent
(build_service_table.py). Τα λάθη τους είναι λεξικές παγίδες: «δήλωση
διαζυγίου» → «Άνοιγμα μερίδας λόγω διαζυγίου». Το regression δεν τα βλέπει,
γιατί κρατάει τον πίνακα σταθερό. Και από τις 13/10 μια λάθος γραμμή μπορεί
να δώσει και σίγουρη απάντηση (connector.destination_confidence).

ΠΩΣ: το μοντέλο «διαβάζει» τη σελίδα (τίτλο και αρχή κειμένου, μέσος όρος
πιθανοτήτων). Για τη γραμμή X → σελίδα P μετράμε πόσα ΞΕΝΑ intents βάζει
πάνω από το X: intents που ο πίνακας στέλνει ΑΛΛΟΥ. Τα «αδέρφια» της ίδιας
σελίδας (6 ανάγκες ΤΑΠ → e-ΤΑΠ) δεν μετράνε. Ύποπτη = τουλάχιστον ένα ξένο.

Η λεξική ομοιότητα προτάσεων ↔ σελίδας δοκιμάστηκε και ΔΕΝ ξεχωρίζει τα
λάθη: 5 από τα 8 γνωστά λάθη έβγαιναν 1η ή 2η επιλογή. Λογικό, αφού έτσι
φτιάχτηκαν.

ΕΠΙΚΥΡΩΣΗ (--validate): «γνωστά λάθη» = παλιές αντιστοιχίσεις σε σελίδα που
διορθώθηκαν αργότερα με το χέρι, από το ιστορικό του git· «γνωστά σωστά» =
οι verified γραμμές. Στις 13/10: πιάνει 6/8 λάθη, σημαδεύει 7/51 verified.
Δεν πιάνει (α) όταν ο τίτλος περιέχει αυτολεξεί το όνομα του intent (παιδικοί
σταθμοί → σελίδα αδειοδότησης ΜΦΠΑΔ) και (β) τα τέλη κοιμητηρίων στη σελίδα
του ΤΑΠ, που έχει ήδη 8 intents. Ο κανόνας «≥ 1 ξένο» και η ανάγνωση «τίτλος
+ κείμενο» διαλέχτηκαν ανάμεσα σε 3 παραλλαγές πάνω στα ίδια 8 λάθη, άρα το
6/8 είναι μάλλον αισιόδοξο.

Το script ΔΕΝ αλλάζει τον πίνακα· βγάζει λίστα για άνθρωπο.

    python3 scripts/precheck_table.py              # → reports/table_precheck.csv
    python3 scripts/precheck_table.py --validate   # + γνωστά λάθη / verified
"""
import argparse, csv, io, json, re, subprocess, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import connector as C
from paths import DATASETS, KB_RICH, REPORTS, ROOT
from regression import key

OUT = REPORTS / "table_precheck.csv"
TABLE_REL = "mappings/intent_to_service.csv"
# «Η υπηρεσία αυτή παρέχεται αποκλειστικά με φυσική παρουσία…»: ίδιο σε
# δεκάδες σελίδες, δεν λέει τίποτα για το ΤΙ είναι η υπηρεσία.
BOILER = re.compile(r"^Η υπηρεσία αυτή παρέχεται[^.]*\.\s*")


class Reader:
    """Το μοντέλο διαβάζει σελίδες του KB."""

    def __init__(self):
        self.tok, self.mdl, le = C.load_bert()
        self.classes = [key(c) for c in le.classes_]
        self.kb = {r["url"]: r for r in json.load(open(KB_RICH, encoding="utf-8"))
                   if r["title"] not in C.EXCLUDED_TITLES}
        self._cache = {}

    def _probs(self, text):
        x = self.tok(C.normalize(text), return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            return torch.softmax(self.mdl(**x).logits, -1)[0].numpy()

    def page(self, url):
        """Μέσος όρος πιθανοτήτων τίτλου και αρχής κειμένου (300 χαρ.)."""
        if url not in self._cache:
            r = self.kb[url]
            t = self._probs(r["title"])
            body = BOILER.sub("", r.get("body", "") or "").strip()[:300]
            self._cache[url] = (t + self._probs(body)) / 2 if body else t
        return self._cache[url]

    def foreign_above(self, intent, url, rows):
        """(πλήθος ξένων πάνω από το intent, το πρώτο ξένο, πιθανότητα του intent)."""
        ik = key(intent)
        if url not in self.kb or ik not in self.classes:
            return None, None, None
        same = {key(r["intent"]) for r in rows if r["url"] == url and r["status"] != "no_service"}
        p = self.page(url)
        px = p[self.classes.index(ik)]
        foreign = [self.classes[i] for i in np.argsort(p)[::-1]
                   if p[i] > px and self.classes[i] not in same]
        return len(foreign), (foreign[0] if foreign else None), float(px)

    def best_page(self, intent):
        """Η σελίδα του KB στην οποία το μοντέλο δίνει τη μεγαλύτερη πιθανότητα για το intent."""
        i = self.classes.index(key(intent))
        url = max(self.kb, key=lambda u: self.page(u)[i])
        return url, float(self.page(url)[i])


def read_table(rev=None):
    if rev is None:
        return list(csv.DictReader(open(ROOT / TABLE_REL, encoding="utf-8-sig")))
    t = subprocess.run(["git", "-C", str(ROOT), "show", f"{rev}:{TABLE_REL}"],
                       capture_output=True, text=True, check=True).stdout
    return list(csv.DictReader(io.StringIO(t.lstrip("﻿"))))


def known_errors():
    """Παλιές αντιστοιχίσεις σε σελίδα που άλλαξαν προορισμό σε επόμενο commit:
    [(intent, url, πίνακας εκείνης της εποχής)]."""
    revs = subprocess.run(["git", "-C", str(ROOT), "log", "--reverse", "--format=%h", "--", TABLE_REL],
                          capture_output=True, text=True, check=True).stdout.split()
    out, prev = [], None
    for rev in revs:
        cur = read_table(rev)
        if prev is not None:
            now = {r["intent"]: r for r in cur}
            for r in prev:
                n = now.get(r["intent"])
                if (n and r["status"] != "no_service" and r["url"]
                        and (n["url"], n["department"]) != (r["url"], r["department"])):
                    out.append((r["intent"], r["url"], prev))
        prev = cur
    return out


def validate(reader):
    errs = known_errors()
    caught = 0
    print(f"\n  Γνωστά λάθη από το ιστορικό ({len(errs)}):")
    for intent, url, rows in errs:
        n, top, _ = reader.foreign_above(intent, url, rows)
        caught += bool(n)
        print(f"    {'✓' if n else '✗'} {intent:40} ξένα πάνω: {n}  «{reader.kb[url]['title'][:45]}»")
    rows = read_table()
    ver = [r for r in rows if r["status"] == "verified" and r["url"] in reader.kb]
    flagged = [r["intent"] for r in ver if reader.foreign_above(r["intent"], r["url"], rows)[0]]
    print(f"\n  Πιάνει {caught}/{len(errs)} γνωστά λάθη · σημαδεύει {len(flagged)}/{len(ver)} verified:")
    print(f"    {', '.join(flagged)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    reader = Reader()
    rows = read_table()
    dest = {key(r["intent"]): (r["title"] or r["department"]) for r in rows}
    examples = defaultdict(list)
    for r in csv.DictReader(open(DATASETS / "intent_dataset_v6_clean.csv", encoding="utf-8-sig")):
        if len(examples[key(r["intent"])]) < 3:
            examples[key(r["intent"])].append(r["text"])

    out = []
    for r in rows:
        if r["status"] not in ("auto", "proposed"):
            continue
        n, top, px = reader.foreign_above(r["intent"], r["url"], rows)
        best_url, best_p = reader.best_page(r["intent"]) if key(r["intent"]) in reader.classes else ("", 0.0)
        out.append({
            "ύποπτη": "ΝΑΙ" if n else ("δεν ελέγχθηκε" if n is None else ""),
            "ξένα_πάνω": "" if n is None else n,
            "intent": r["intent"],
            "status": r["status"],
            "τρέχουσα_σελίδα": r["title"],
            "url": r["url"],
            "το_μοντέλο_προτιμά_για_τη_σελίδα": f"{top} → {dest.get(top, '?')}" if top else "",
            "καλύτερη_σελίδα_για_το_intent": ("" if best_url == r["url"]
                                              else reader.kb[best_url]["title"] if best_url else ""),
            "παραδείγματα_πολιτών": " | ".join(examples.get(key(r["intent"]), [])),
            "απόφαση": "",
            "σχόλιο": "",
            "_px": px if px is not None else -1,
        })
    # Πρώτα οι ύποπτες (περισσότερα ξένα = πιο ύποπτη), μετά όσες δεν ελέγχθηκαν,
    # μετά οι υπόλοιπες με τη χαμηλότερη πιθανότητα πρώτα.
    out.sort(key=lambda o: (o["ύποπτη"] != "ΝΑΙ", o["ύποπτη"] != "δεν ελέγχθηκε",
                            -(o["ξένα_πάνω"] or 0), o["_px"]))
    for o in out:
        del o["_px"]
    REPORTS.mkdir(exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:   # BOM: ανοίγει σωστά στο Excel
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    tmp.replace(OUT)

    n_sus = sum(o["ύποπτη"] == "ΝΑΙ" for o in out)
    n_unk = sum(o["ύποπτη"] == "δεν ελέγχθηκε" for o in out)
    print(f"\n  {len(out)} γραμμές auto/proposed · ύποπτες {n_sus} · χωρίς κείμενο σελίδας {n_unk}")
    for o in out[:n_sus]:
        print(f"    {o['ξένα_πάνω']:4} {o['status'][:4]} {o['intent'][:40]:40} «{o['τρέχουσα_σελίδα'][:45]}»")
    print(f"\n  → {OUT.relative_to(ROOT)}")
    if args.validate:
        validate(reader)


if __name__ == "__main__":
    main()
