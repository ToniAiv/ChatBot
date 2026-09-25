"""
Αντοχή σε γραφές ακρωνυμίων: ΤΑΠ / ταπ / Τ.Α.Π. / etap / «τέλος ακίνητης
περιουσίας» πρέπει να καταλήγουν στο ίδιο intent.

Σωστό = σωστό intent με βεβαιότητα ≥ MIN_BERT_CONFIDENCE. Για intent
«none» (δεν υπάρχει υπηρεσία) σωστό = να ΜΗΝ απαντήσει σίγουρα.

    python3 scripts/eval_acronyms.py
"""
import csv, re, sys, unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import connector as C
from paths import DATASETS

EVAL = DATASETS / "acronyms_eval.csv"


def key(s):
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
    return re.sub(r"[^a-zα-ω0-9]", "", s)


def evaluate(tok, mdl, le, verbose=True):
    rows = list(csv.DictReader(open(EVAL, encoding="utf-8")))
    ok = 0
    for r in rows:
        gq, _lang = C.resolve_query(r["text"])
        it, cf = C.detect_intent(gq, tok, mdl, le)[0]
        sure = cf >= C.MIN_BERT_CONFIDENCE
        good = (not sure) if r["intent"] == "none" else (sure and key(it) == key(r["intent"]))
        ok += good
        if verbose:
            shown = "" if gq == r["text"] else f"   ⟶ «{gq}»"
            print(f"  {'✓' if good else '✗'} {cf:4.2f} {r['text'][:48]:50} {it[:38]}{shown}")
    return ok / len(rows), len(rows)


if __name__ == "__main__":
    acc, n = evaluate(*C.load_bert())
    print(f"\n  ακρωνύμια: {acc:.4f}  (n={n})")
