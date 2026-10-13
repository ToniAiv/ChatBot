"""
Οικογένειες στο «Μήπως εννοείτε;» (connector.family_suggestions), σε σετ που
γράφτηκε ΠΡΙΝ τρέξει ο κανόνας: datasets/family_holdout.csv (36 προτάσεις σε
μεταδημοτεύσεις, διορθώσεις, ληξιαρχείο, ΚΟΚ, κοινωνικές, οφειλές κ.ά.).

Έλεγχος διαρροής: οι 14 ερωτήσεις εκτός αρμοδιότητας του
datasets/pooling_holdout.csv δεν πρέπει να παίρνουν οικογένεια.

    python3 scripts/eval_families.py            # πριν/μετά
"""
import csv, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import connector as C
from eval_pooling import answer_dest, dest_of
from paths import DATASETS
from regression import Bot, key

EVAL = DATASETS / "family_holdout.csv"
OUT_OF_SCOPE = DATASETS / "pooling_holdout.csv"


def evaluate(bot, verbose=False):
    classes = {key(c): c for c in bot.le.classes_}
    res, n_opts = Counter(), []
    for r in csv.DictReader(open(EVAL, encoding="utf-8")):
        gold = classes[key(r["intent"])]
        gd = dest_of(bot, gold)
        it, cf, out = bot.ask(r["text"], translate=False)
        if out in ("link", "phone"):
            v = "σωστό" if answer_dest(bot, it) == gd else "σίγουρα λάθος"
        elif out == "suggest":
            n_opts.append(len(it))
            v = "πρόταση με σωστό" if any(dest_of(bot, o) == gd for o in it) else "πρόταση χωρίς σωστό"
        else:
            v = "άρνηση"
        res[v] += 1
        if verbose:
            fam = f"[{bot.last_family}] " if bot.last_family else ""
            print(f"  {v:20} {cf:4.2f}  {r['text'][:52]:54} {fam}{len(it) if out == 'suggest' else ''}")
    oos = Counter()
    for r in csv.DictReader(open(OUT_OF_SCOPE, encoding="utf-8")):
        if r["intent"]:
            continue
        _it, _cf, out = bot.ask(r["text"], translate=False)
        oos["οικογένεια" if bot.last_family else out] += 1
    return res, (sum(n_opts) / len(n_opts) if n_opts else 0.0), oos


if __name__ == "__main__":
    bot = Bot()
    C.FAMILY_SUGGESTIONS = False
    before, ob, oos_b = evaluate(bot)
    C.FAMILY_SUGGESTIONS = True
    after, oa, oos_a = evaluate(bot, verbose=True)
    print()
    for k in ("σωστό", "πρόταση με σωστό", "πρόταση χωρίς σωστό", "άρνηση", "σίγουρα λάθος"):
        print(f"  {k:20} {before[k]:3} → {after[k]:3}")
    print(f"  επιλογές ανά πρόταση {ob:.1f} → {oa:.1f}")
    print(f"  εκτός αρμοδιότητας (14): {dict(oos_b)} → {dict(oos_a)}")
