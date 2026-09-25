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
        """Επιστρέφει (intent, confidence, έκβαση). Στο «suggest» το
        intent αντικαθίσταται από τη λίστα των προτεινόμενων."""
        # Για ελληνικά, η παραγωγή περνάει από spellfix χωρίς μετάφραση.
        # Το translate=False πρέπει να το κάνει κι αυτό, αλλιώς η μέτρηση
        # δοκιμάζει διαδρομή που δεν υπάρχει (έδειχνε 0.7875 αντί 0.86).
        gq, lang = (C.resolve_query(q) if translate
                    else (C.prepare_greek(q), "el"))
        tops = C.detect_intent(gq, self.tok, self.mdl, self.le)
        it, cf = tops[0]
        if cf < C.MIN_BERT_CONFIDENCE:
            if C.should_suggest(tops):
                opts = C.suggestions(tops, self.vec, self.mat, self.valid,
                                     self.table, self.ns)
                if opts:
                    return [o["intent"] for o in opts], cf, "suggest"
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

    results, failures = {}, []

    # Έλεγχος του JS του παραθύρου πριν από οτιδήποτε άλλο: είναι γρήγορος
    # και ελέγχει κώδικα που δεν αγγίζει καθόλου το pipeline της Python.
    import subprocess
    js = subprocess.run([sys.executable, str(Path(__file__).parent / "test_gui_js.py")],
                        capture_output=True, text=True)
    print("\n  JavaScript παραθύρου:")
    for line in js.stdout.strip().split("\n"):
        if line.strip():
            print("  " + line.rstrip())
    if js.returncode != 0:
        failures.append("έλεγχοι JavaScript απέτυχαν")

    bot = Bot()

    # ── 1. ακρίβεια intent ───────────────────────────────────
    rows = []
    for f in sorted(glob.glob(str(DATASETS / "tests" / "*.csv"))):
        rows += [r for r in csv.DictReader(open(f, encoding="utf-8")) if r.get("text")]
    if args.quick:
        import random
        random.Random(0).shuffle(rows); rows = rows[:300]

    t0 = time.monotonic()
    # ΠΡΟΣΟΧΗ: το intent_top1 μετριέται στο argmax του μοντέλου, ΑΝΕΞΑΡΤΗΤΑ
    # από κατώφλια και προτάσεις. Είναι καθαρή ακρίβεια μοντέλου και πρέπει
    # να μένει συγκρίσιμη διαχρονικά. Όταν προστέθηκαν οι προτάσεις, μια
    # πρώτη εκδοχή το μετρούσε μόνο στις μη-προτεινόμενες και έδειχνε
    # πτώση 0.8884 → 0.7722 χωρίς να έχει χαλάσει τίποτα.
    # ΠΡΟΟΡΙΣΜΟΣ = η σελίδα ή το τμήμα που θα δει ο πολίτης, υπολογισμένος
    # ΑΚΡΙΒΩΣ όπως τον υπολογίζει το bot (πίνακας → τμήμα → εφεδρεία).
    # «Προσεγγίσιμο» σημαίνει: ο πολίτης καταλήγει εκεί που θα κατέληγε αν
    # το μοντέλο τον είχε καταλάβει τέλεια.
    #
    # Γιατί όχι σύγκριση intent: πολλά intents οδηγούν νόμιμα στην ίδια
    # σελίδα (6 ανάγκες ΤΑΠ → εφαρμογή e-ΤΑΠ· «μείωση τιμολογίου» και
    # «έλεγχος λογαριασμού» → ίδιο τηλέφωνο ΔΕΥΑΗ). Σε επίπεδο intent οι
    # προτάσεις φαίνονταν 85,5% σωστές ενώ ο πολίτης έβλεπε σωστή σελίδα
    # πολύ συχνότερα. Μια ενδιάμεση εκδοχή διάβαζε τον προορισμό μόνο από
    # τον ενεργό πίνακα και έδινε «κανέναν» στα proposed intents — που
    # απαντώνται από την εφεδρεία — μετρώντας τα πάντα ως αποτυχία.
    #
    # Περιορισμός: αν ένα intent είναι ΛΑΘΟΣ αντιστοιχισμένο, αυτό δεν
    # φαίνεται εδώ — η μέτρηση κρατάει τον πίνακα σταθερό. Το intent_top1
    # μένει σε επίπεδο intent ακριβώς γι' αυτό.
    _classes = {key(c): c for c in bot.le.classes_}
    _cache = {}
    def dest(intent):
        c = _classes.get(key(intent), intent)
        if c not in _cache:
            if c in bot.ns:
                d = bot.ns[c]
                _cache[c] = "dept:" + (d.get("name", "") if d else "")
            else:
                cand, _ = C.find_candidates(c, bot.vec, bot.mat, bot.valid, bot.table)
                _cache[c] = (cand[0][1]["url"] if cand and cand[0][0] >= C.MIN_TFIDF_SCORE
                             else None)
        return _cache[c]

    outcomes = Counter(); ok = 0; sugg_hit = 0; direct_ok = 0
    for r in rows:
        it, cf, out = bot.ask(r["text"], translate=False)   # όλα ελληνικά
        outcomes[out] += 1
        gold = key(r["intent"])
        gd = dest(r["intent"])
        if out == "suggest":
            ok += gold == key(it[0])           # το argmax είναι το πρώτο
            sugg_hit += gd is not None and gd in {dest(x) for x in it}
        else:
            ok += gold == key(it)
            if out in ("link", "phone"):
                direct_ok += gd is not None and dest(it) == gd
    results["mode"] = "quick" if args.quick else "full"
    results["n"] = len(rows)
    results["intent_top1"] = ok / len(rows)
    results["coverage"] = (outcomes["link"] + outcomes["phone"]) / len(rows)
    # Προσεγγίσιμο = απαντήθηκε σωστά κατευθείαν, Ή προτάθηκε λίστα που
    # περιέχει τη σωστή υπηρεσία. Το δεύτερο απαιτεί να διαλέξει σωστά ο
    # πολίτης — γι' αυτό είναι ξεχωριστό νούμερο, όχι μέρος της ακρίβειας.
    results["reachable"] = (direct_ok + sugg_hit) / len(rows)
    results["suggest_hit"] = sugg_hit / outcomes["suggest"] if outcomes["suggest"] else 0.0
    ms = (time.monotonic() - t0) / len(rows) * 1000
    print(f"\n  Test set: {len(rows)} προτάσεις, {ms:.0f} ms/ερώτηση")
    print(f"    intent top-1 : {results['intent_top1']:.4f}")
    print(f"    κάλυψη       : {results['coverage']:.4f}")
    print(f"    προσεγγίσιμο : {results['reachable']:.4f}  (απάντηση ή σωστή πρόταση)")
    if outcomes["suggest"]:
        print(f"    προτάσεις    : {outcomes['suggest']} · σωστό μέσα στις επιλογές "
              f"{results['suggest_hit']:.1%}")
    for k, v in outcomes.most_common():
        print(f"      {v:5} {v/len(rows):6.1%}  {k}")

    # ── 1β. αντοχή σε ορθογραφικά λάθη ───────────────────────
    # Το dataset είναι ορθογραφικά τέλειο (παρήχθη από LLM), οπότε χωρίς
    # τη διόρθωση ένα λάθος ανά πρόταση κόστιζε 9,5 μονάδες. Μετριέται
    # εδώ ώστε μια αλλαγή στο spellfix να μην περάσει απαρατήρητη.
    import random as _r
    from typos import corrupt
    rng = _r.Random(7)
    sample = rows if args.quick else rows[:400]
    ok_t = 0
    for r in sample:
        bad = corrupt(r["text"], rng, 1)
        it, _cf, _o = bot.ask(bad, translate=False)
        it = it[0] if isinstance(it, list) else it
        ok_t += key(it) == key(r["intent"])
    results["typo_top1"] = ok_t / len(sample)
    print(f"    με 1 ορθ. λάθος: {results['typo_top1']:.4f}  (n={len(sample)})")

    # ── 1γ. απαντήσεις περιεχομένου ──────────────────────────
    # Ανιχνευτής πτυχής: σετ γραμμένο ΠΡΙΝ τον ανιχνευτή, με παγίδες
    # («χρειάζομαι βεβαίωση» ≠ «τι χρειάζομαι για βεβαίωση», «έξω από το
    # σπίτι» ≠ «από το σπίτι»).
    from aspects import detect_aspect
    ae = list(csv.DictReader(open(DATASETS / "aspects_eval.csv", encoding="utf-8")))
    results["aspect_acc"] = sum((detect_aspect(r["text"]) or "none") == r["aspect"]
                                for r in ae) / len(ae)
    print(f"    πτυχή ερώτησης : {results['aspect_acc']:.4f}  (n={len(ae)})")

    # Ακρωνύμια: ΤΑΠ / Τ.Α.Π. / etap / «τέλος ακίνητης περιουσίας» στο ίδιο
    # intent (datasets/acronyms_eval.csv, mappings/acronyms.csv).
    from eval_acronyms import evaluate as eval_acronyms
    results["acronyms"], n_ac = eval_acronyms(bot.tok, bot.mdl, bot.le, verbose=False)
    print(f"    ακρωνύμια      : {results['acronyms']:.4f}  (n={n_ac})")

    # Από άκρη σε άκρη: η απάντηση πρέπει να ξεκινά με το σωστό πρότυπο.
    C.SECTIONS.update(C.load_sections())
    CONTENT = [
        ("τι χαρτια χρειαζομαι για βεβαιωση μονιμης κατοικιας", "Για «"),
        ("θελω βεβαιωση μονιμης κατοικιας",                     "Η αρμόδια υπηρεσία"),
        ("ποιο ειναι το τηλεφωνο για πιστοποιητικο οικογενειακης καταστασης", "Στοιχεία επικοινωνίας"),
        ("μπορω να κανω την αιτηση για θεση αμεα ηλεκτρονικα",  "Ναι,"),
        ("γινεται ηλεκτρονικα η δηλωση βαπτισης",               "Όχι."),
        ("ποσο κοστιζει η βεβαιωση κατοικιας",                  "Η αρμόδια υπηρεσία"),
    ]
    c_ok = 0
    for q, want in CONTENT:
        gq, lang = C.resolve_query(q)
        it, cf = C.detect_intent(gq, bot.tok, bot.mdl, bot.le)[0]
        cand, src = C.find_candidates(it, bot.vec, bot.mat, bot.valid, bot.table)
        ans = C.compose_answer(gq, it, cand, src, lang)
        good = ans.startswith(want)
        c_ok += good
        if not good:
            failures.append(f"περιεχόμενο: «{q[:40]}» περίμενα «{want}», πήρα «{ans[:40]}»")
    results["content"] = c_ok / len(CONTENT)
    print(f"    απαντήσεις περιεχομένου: {c_ok}/{len(CONTENT)}")

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
