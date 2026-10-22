"""
Προετοιμασία νέας εκδοχής dataset πριν την εκπαίδευση.

ΓΙΑΤΙ: κάθε φορά που η ομάδα στέλνει νέο dataset, τρία πράγματα πρέπει να
ελεγχθούν πριν μπει σε εκπαίδευση — αλλιώς τα προβλήματα φαίνονται μόνο
ΜΕΤΑ, στο bot:
  1. Διαρροή: προτάσεις που υπάρχουν ΚΑΙ στα σετ δοκιμών μας. Αν μπουν
     στην εκπαίδευση, το μοντέλο τις «αποστηθίζει» και τα νούμερα
     φουσκώνουν. Αφαιρούνται από το αντίγραφο εκπαίδευσης.
  2. Labels → πίνακας υπηρεσιών: κάθε intent χρειάζεται γραμμή στο
     mappings/intent_to_service.csv, αλλιώς πέφτει σιωπηλά στον TF-IDF.
     Η σύγκριση γίνεται με την ΙΔΙΑ κανονικοποίηση που κάνει το notebook
     εκπαίδευσης (archive/prakt_train_0.ipynb, κελί 10) — χωρίς τόνους,
     πεζά, χωρίς κενά/σύμβολα — γιατί έτσι θα ονομαστούν οι κλάσεις.
  3. Labels που διαφέρουν μόνο σε τόνους/κεφαλαία: στην εκπαίδευση θα
     ενωθούν σε μία κλάση· καλό να το ξέρουμε από πριν.

Το αρχικό αρχείο της ομάδας ΔΕΝ αλλάζει. Γράφεται αντίγραφο, και μια
αναφορά με ό,τι αφαιρέθηκε, ώστε να μπορεί να ελεγχθεί.

Χρήση:
    python3 scripts/prepare_dataset.py ΑΡΧΕΙΟ_ΟΜΑΔΑΣ.csv ΕΞΟΔΟΣ.csv
"""
import argparse, collections, csv, re, sys, unicodedata
from pathlib import Path

from paths import ROOT, DATASETS

TABLE = ROOT / "mappings" / "intent_to_service.csv"
# Όλα τα αρχεία με τα οποία ΜΕΤΡΑΜΕ το bot — καμία τους πρόταση στην εκπαίδευση.
EVAL_FILES = sorted((DATASETS / "tests").glob("*.csv")) + [
    DATASETS / n for n in ("general_eval.csv", "greeklish_eval.csv", "acronyms_eval.csv",
                           "aspects_eval.csv", "dialogue_eval.csv", "dialogue_holdout.csv",
                           "department_eval.csv", "department_holdout.csv",
                           "pooling_holdout.csv", "family_holdout.csv")] + [
    ROOT / "agent" / n for n in ("bakeoff_questions.csv", "holdout_questions.csv", "holdout2_questions.csv",
                                    "followup_questions.csv")]


def notebook_label(x: str) -> str:
    """Ακριβώς η κανονικοποίηση labels του notebook εκπαίδευσης."""
    x = unicodedata.normalize("NFD", str(x))
    x = "".join(c for c in x if unicodedata.category(c) != "Mn").lower().strip()
    x = re.sub(r"[\t\n\r\s]+", "", x)
    return re.sub(r"[^\w_α-ωΑ-Ω]", "", x)


def text_key(s: str) -> str:
    """Σύγκριση κειμένων χωρίς τόνους, κεφαλαία και στίξη."""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def eval_texts() -> dict:
    out = {}
    for f in EVAL_FILES:
        if not f.exists():
            continue
        for r in csv.DictReader(open(f, encoding="utf-8")):
            for col in ("text", "m1", "m2"):
                if r.get(col):
                    out.setdefault(text_key(r[col]), f.name)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("output")
    args = ap.parse_args()
    src, out = Path(args.source), Path(args.output)

    with open(src, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"  {src.name}: {len(rows)} προτάσεις, {len({r['intent'] for r in rows})} labels")

    # 3. labels που θα ενωθούν
    groups = collections.defaultdict(set)
    for r in rows:
        groups[notebook_label(r["intent"])].add(r["intent"])
    merged = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"  labels που θα ενωθούν στην εκπαίδευση (ίδια χωρίς τόνους): {len(merged)}")
    for k, v in merged.items():
        print(f"     ⚠ {sorted(v)} → {k}")

    # 1. διαρροή
    ev = eval_texts()
    keep, leaked = [], []
    for r in rows:
        (leaked if text_key(r["text"]) in ev else keep).append(r)
    print(f"  διαρροή προς τα σετ δοκιμών: {len(leaked)} προτάσεις αφαιρούνται")
    for name, c in collections.Counter(ev[text_key(r["text"])] for r in leaked).most_common():
        print(f"     {c:3}  {name}")

    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(keep)
    report = out.with_name(out.stem + "_removed.csv")
    with open(report, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "intent", "same_as_in"])
        for r in leaked:
            w.writerow([r["text"], r["intent"], ev[text_key(r["text"])]])
    print(f"  💾 {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}  ({len(keep)} προτάσεις)")
    print(f"  💾 {report.relative_to(ROOT) if report.is_relative_to(ROOT) else report}  (ό,τι αφαιρέθηκε)")

    # 2. labels → πίνακας
    table = {r["intent"] for r in csv.DictReader(open(TABLE, encoding="utf-8"))}
    labels = {notebook_label(r["intent"]) for r in keep}
    missing = sorted(labels - table)
    print(f"  κλάσεις μετά την εκπαίδευση: {len(labels)} · με γραμμή στον πίνακα: {len(labels & table)}"
          f" · ΧΩΡΙΣ γραμμή: {len(missing)}")
    for m in missing:
        print(f"     ✗ {m}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
