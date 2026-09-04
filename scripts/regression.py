"""
Έλεγχος παλινδρόμησης: τρέχει μετά από κάθε αλλαγή στο σύστημα.

ΓΙΑΤΙ ΥΠΑΡΧΕΙ: όλα τα νούμερα του project ζούσαν σε αναφορές, όχι σε
επαναλήψιμο έλεγχο. Αν κάποιος άλλαζε το κατώφλι, τον πίνακα ή το
γλωσσάρι, δεν υπήρχε τρόπος να δει τι χάλασε. Τώρα υπάρχει.

Ελέγχει τρία πράγματα:
  1. Ακρίβεια intent στο test set (datasets/tests/, 1.980 προτάσεις)
  2. Κατανομή εκβάσεων — σύνδεσμος / τηλέφωνο / άρνηση
  3. Τις 15 ερωτήσεις επίδειξης, με αναμενόμενη έκβαση ανά ερώτηση

Επιστρέφει exit code 1 αν κάτι πέσει κάτω από τα κατώφλια ανοχής, ώστε
να μπορεί να μπει σε CI.

Χρήση:
    python3 scripts/regression.py              # πλήρες
    python3 scripts/regression.py --quick      # δείγμα 300, χωρίς αγγλικά
    python3 scripts/regression.py --update     # ενημέρωση baseline
"""
import argparse, csv, glob, json, re, sys, time, unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import connector as C
from paths import DATASETS, ROOT

BASELINE = Path(__file__).resolve().parent / "regression_baseline.json"
TOLERANCE = 0.02          # πόσο επιτρέπεται να πέσει ένα ποσοστό

# (ερώτηση, αναμενόμενη έκβαση). Οι εκβάσεις είναι σκόπιμα χονδρικές:
# ελέγχουμε ΣΥΜΠΕΡΙΦΟΡΑ, όχι ποια ακριβώς υπηρεσία — αυτό το καλύπτει
# ήδη η ακρίβεια intent και αλλάζει νόμιμα όταν διορθώνεται ο πίνακας.
DEMO = [
    ("ο καδος ειναι μπροστα στο γκαραζ μου και δεν μπορω να βγω", "link"),
    ("το φαναρι στη διασταυρωση αναβοσβηνει πορτοκαλι",           "link"),
    ("μενω στο κεντρο, πως παιρνω καρτα σταθμευσης κατοικου;",    "link"),
    ("μου ηρθε κληση για παρανομο παρκαρισμα ενω ημουν νομιμα",   "link"),
    ("ποια μερα στηνεται η λαικη αγορα στη γειτονια μου;",        "link"),
    ("ποσο στοιχιζει να νοικιασω ταφο για τρια χρονια;",          "phone"),
    ("θελω βεβαιωση οτι το οικοπεδο μου ειναι αρτιο",             "phone"),
    ("ποσο εχει το εισιτηριο για το αεροδρομιο;",                 "refuse"),
    ("someone dumped an old fridge on the pavement",              "link"),
    ("there are stray dogs in the neighbourhood and children are scared", "link"),
    ("I got a parking fine but I was parked legally",             "link"),
    ("I need a family status certificate for the bank",           "link"),
    ("I can't pay it all at once, can I pay in instalments?",     "link"),
    ("how much does it cost to rent a grave?",                    "phone"),
    ("how much is the ticket to the airport?",                    "refuse"),
]


def key(s):
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
    return re.sub(r"[^a-zα-ω0-9]", "", s)


class Bot:
    def __init__(self):
        self.tok, self.mdl, self.le = C.load_bert()
        kb = json.load(open(C.JSON_PATH, encoding="utf-8"))
        self.vec, self.mat, self.valid = C.build_tfidf_index(kb)
        self.table = C.load_service_table(self.valid)
        self.ns = C.load_no_service(C.load_departments())

    def ask(self, q, translate=True):
        """Επιστρέφει (intent, confidence, έκβαση)."""
        gq, lang = (C.resolve_query(q) if translate else (q, "el"))
        it, cf = C.detect_intent(gq, self.tok, self.mdl, self.le)[0]
        if cf < C.MIN_BERT_CONFIDENCE:
            return it, cf, "refuse"
        if it in self.ns:
            return it, cf, "phone"
        cand, _ = C.find_candidates(it, self.vec, self.mat, self.valid, self.table)
        if cand[0][0] < C.MIN_TFIDF_SCORE:
            return it, cf, "refuse"
        return it, cf, "link"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--update", action="store_true")
    args = ap.parse_args()

    bot = Bot()
    results, failures = {}, []

    # ── 1. ακρίβεια intent ───────────────────────────────────
    rows = []
    for f in sorted(glob.glob(str(DATASETS / "tests" / "*.csv"))):
        rows += [r for r in csv.DictReader(open(f, encoding="utf-8")) if r.get("text")]
    if args.quick:
        import random
        random.Random(0).shuffle(rows); rows = rows[:300]

    t0 = time.monotonic()
    outcomes = Counter(); ok = 0
    for r in rows:
        it, cf, out = bot.ask(r["text"], translate=False)   # όλα ελληνικά
        outcomes[out] += 1
        ok += key(it) == key(r["intent"])
    results["mode"] = "quick" if args.quick else "full"
    results["n"] = len(rows)
    results["intent_top1"] = ok / len(rows)
    results["coverage"] = (outcomes["link"] + outcomes["phone"]) / len(rows)
    ms = (time.monotonic() - t0) / len(rows) * 1000
    print(f"\n  Test set: {len(rows)} προτάσεις, {ms:.0f} ms/ερώτηση")
    print(f"    intent top-1 : {results['intent_top1']:.4f}")
    print(f"    κάλυψη       : {results['coverage']:.4f}")
    for k, v in outcomes.most_common():
        print(f"      {v:5} {v/len(rows):6.1%}  {k}")

    # ── 2. ερωτήσεις επίδειξης ───────────────────────────────
    print(f"\n  Ερωτήσεις επίδειξης:")
    demo = [d for d in DEMO if not (args.quick and re.search(r"[a-z]{4}", d[0]))]
    demo_ok = 0
    for q, want in demo:
        try:
            it, cf, out = bot.ask(q)
        except C.TranslationUnavailable:
            failures.append(f"μετάφραση απέτυχε (τρέχει το Ollama;): {q[:40]}")
            print(f"    ✗ {q[:52]}  → μετάφραση απέτυχε")
            continue
        good = out == want
        demo_ok += good
        if not good:
            failures.append(f"«{q[:46]}» περίμενα {want}, πήρα {out} ({it}, {cf:.2f})")
        print(f"    {'✓' if good else '✗'} {out:7} {q[:52]}")
    results["demo"] = demo_ok / len(demo)
    print(f"    {demo_ok}/{len(demo)}")

    # ── 3. σύγκριση με baseline ──────────────────────────────
    if args.update or not BASELINE.exists():
        BASELINE.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\n  💾 baseline ενημερώθηκε: {BASELINE.relative_to(ROOT)}")
        return 0

    base = json.loads(BASELINE.read_text(encoding="utf-8"))

    # Το --quick τρέχει σε δείγμα· σύγκριση με baseline πλήρους σετ θα
    # έδειχνε ψεύτικες μεταβολές (το είδαμε: +0.025 top-1 από τύχη
    # δείγματος). Συγκρίνονται μόνο ίδιοι τρόποι.
    if base.get("mode") != results["mode"]:
        print(f"\n  ⚠ Το baseline είναι «{base.get('mode')}» και τρέχεις «{results['mode']}».")
        print(f"    Τα ποσοστά ΔΕΝ συγκρίνονται. Ελέγχονται μόνο οι ερωτήσεις επίδειξης.")
        if failures:
            print(f"\n  ✗ ΑΠΟΤΥΧΙΑ ({len(failures)}):")
            for f in failures:
                print(f"      {f}")
            return 1
        print(f"\n  ✓ Οι ερωτήσεις επίδειξης περνούν")
        return 0

    print(f"\n  Σύγκριση με baseline ({base['mode']}, n={base.get('n')}):")
    for k, v in results.items():
        if k in ("mode", "n"):
            continue
        b = base.get(k)
        if b is None:
            continue
        d = v - b
        flag = "  ⚠ ΠΤΩΣΗ" if d < -TOLERANCE else ""
        print(f"    {k:14} {b:.4f} → {v:.4f}  ({d:+.4f}){flag}")
        if d < -TOLERANCE:
            failures.append(f"{k}: {b:.4f} → {v:.4f}")

    if failures:
        print(f"\n  ✗ ΑΠΟΤΥΧΙΑ ({len(failures)}):")
        for f in failures:
            print(f"      {f}")
        return 1
    print(f"\n  ✓ Καμία παλινδρόμηση")
    return 0


if __name__ == "__main__":
    sys.exit(main())
