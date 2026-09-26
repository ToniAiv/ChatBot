"""
Greeklish και αγγλικά: σωστό intent με βεβαιότητα ≥ κατωφλίου, και χρόνος.

Για τα greeklish μετράει και πόσα αναγνωρίστηκαν ως greeklish· για τα
αγγλικά, πόσα μπερδεύτηκαν ΛΑΘΟΣ ως greeklish (πρέπει να είναι 0).

    python3 scripts/eval_greeklish.py
"""
import csv, sys, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import connector as C
from paths import DATASETS
from regression import key

EVAL = DATASETS / "greeklish_eval.csv"


def evaluate(tok, mdl, le, verbose=True, kinds=("greeklish", "english")):
    rows = [r for r in csv.DictReader(open(EVAL, encoding="utf-8")) if r["kind"] in kinds]
    stats = defaultdict(lambda: defaultdict(float))
    for r in rows:
        allowed = {key(x) for x in r["intent"].split("|")}
        t0 = time.monotonic()
        try:
            gq, lang = C.resolve_query(r["text"])
            it, cf = C.detect_intent(gq, tok, mdl, le)[0]
        except C.TranslationUnavailable:
            gq, lang, it, cf = "(μετάφραση απέτυχε)", "en", "-", 0.0
        ms = (time.monotonic() - t0) * 1000
        s = stats[r["kind"]]
        good = key(it) in allowed
        s["n"] += 1
        s["top1"] += good
        s["sure_ok"] += good and cf >= C.MIN_BERT_CONFIDENCE
        s["sure_wrong"] += (not good) and cf >= C.MIN_BERT_CONFIDENCE
        s["as_greek"] += lang == "el"
        s["ms"] += ms
        if verbose:
            mark = "✓" if good and cf >= C.MIN_BERT_CONFIDENCE else ("✗" if cf >= C.MIN_BERT_CONFIDENCE else "·")
            print(f"  {mark} {cf:4.2f} {ms:5.0f}ms {lang} {r['text'][:42]:44} ⟶ {gq[:40]:42} {it[:30]}")
    for kind, s in stats.items():
        if not verbose:
            continue
        n = s["n"]
        print(f"\n  {kind:9} n={n:.0f}  σωστό+σίγουρο {s['sure_ok']/n:.0%}  top-1 {s['top1']/n:.0%}  "
              f"σίγουρα λάθος {s['sure_wrong']:.0f}  ως ελληνικά {s['as_greek']:.0f}  "
              f"μέσος χρόνος {s['ms']/n:.0f}ms")
    return stats


if __name__ == "__main__":
    evaluate(*C.load_bert())
