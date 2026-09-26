"""
Γενικές περιπτώσεις: πώς συμπεριφέρεται το bot σε ό,τι ΔΕΝ είναι καθαρή
ερώτηση υπηρεσίας — χαιρετισμοί, εκτός αρμοδιότητας, μονολεκτικά,
φλύαρα, greeklish, συνέχειες χωρίς πλαίσιο, απόπειρες χειραγώγησης.

Το αναμενόμενο (datasets/general_eval.csv) γράφτηκε ΠΡΙΝ το τρέξιμο:
  answer — πρέπει να απαντήσει με ένα από τα intents της στήλης
  none   — δεν πρέπει να απαντήσει (άρνηση· πρόταση = μισό σωστό)
  any    — όλα αποδεκτά, αρκεί αν απαντήσει να είναι ένα από τα intents

    python3 scripts/eval_general.py
"""
import csv, sys, time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from regression import Bot, key
from paths import DATASETS
import connector as C

EVAL = DATASETS / "general_eval.csv"


def judge(expect, allowed, out, it):
    tops = it if isinstance(it, list) else [it]
    ok_intent = (not allowed) or any(key(t) in allowed for t in tops)
    if expect == "none":
        return {"refuse": "pass", "suggest": "partial"}.get(out, "fail")
    if expect == "answer":
        if out in ("link", "phone"):
            return "pass" if ok_intent else "fail"
        if out == "suggest":
            return "partial" if ok_intent else "miss"
        return "miss"
    # any
    if out in ("link", "phone"):
        return "pass" if allowed and ok_intent else ("check" if not allowed else "fail")
    return "pass"


def main():
    bot = Bot()
    rows = list(csv.DictReader(open(EVAL, encoding="utf-8")))
    verdicts, by_cat = Counter(), defaultdict(Counter)
    for r in rows:
        allowed = {key(x) for x in r["intents"].split("|") if x}
        t0 = time.monotonic()
        try:
            it, cf, out = bot.ask(r["text"])
        except C.TranslationUnavailable:
            it, cf, out = "-", 0.0, "tr_fail"
        except Exception as exc:                      # ό,τι σπάει το bot
            it, cf, out = f"{type(exc).__name__}: {exc}", 0.0, "CRASH"
        ms = (time.monotonic() - t0) * 1000
        v = "fail" if out == "CRASH" else judge(r["expect"], allowed, out, it)
        verdicts[v] += 1
        by_cat[r["category"]][v] += 1
        shown = ", ".join(it[:3]) if isinstance(it, list) else it
        mark = {"pass": "✓", "partial": "~", "check": "?", "miss": "·", "fail": "✗"}[v]
        print(f"  {mark} {r['category'][:14]:14} {out:7} {cf:4.2f} {ms:5.0f}ms  "
              f"{r['text'][:50]!r:52} → {shown[:60]}")
    n = len(rows)
    print(f"\n  Σύνολο {n}: " + "  ".join(f"{k} {v}" for k, v in verdicts.most_common()))
    print("\n  Ανά κατηγορία (✓ σωστό · ~ μισό · · δεν απάντησε · ✗ λάθος · ? έλεγχος):")
    for cat, c in by_cat.items():
        print(f"    {cat:24} ✓{c['pass']} ~{c['partial']} ·{c['miss']} ✗{c['fail']} ?{c['check']}")


if __name__ == "__main__":
    main()
