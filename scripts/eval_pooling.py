"""
Βεβαιότητα ανά προορισμό (connector.destination_confidence), σε σετ που
γράφτηκε ΠΡΙΝ τρέξει ο κανόνας: datasets/pooling_holdout.csv.

26 ερωτήσεις για υπηρεσίες που μοιράζονται προορισμό με άλλες (ΔΕΥΑΗ,
Κοιμητήρια, e-ΤΑΠ, Πολεοδομία, Αδειοδοτήσεις κ.ά.) και 14 εκτός
αρμοδιότητας, όπου ο κανόνας ΔΕΝ πρέπει να δώσει σίγουρη απάντηση.

    python3 scripts/eval_pooling.py            # πριν/μετά
"""
import csv, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import connector as C
from paths import DATASETS
from regression import Bot, key

EVAL = DATASETS / "pooling_holdout.csv"


def dest_of(bot, intent):
    """Πού οδηγεί ένα intent: ("dept", κωδικός) ή ("svc", url)."""
    if intent in bot.ns:
        return ("dept", (bot.ns[intent] or {}).get("code", "?"))
    cand, _ = C.find_candidates(intent, bot.vec, bot.mat, bot.valid, bot.table)
    return ("svc", cand[0][1]["url"]) if cand else None


def answer_dest(bot, it):
    """Πού κατέληξε ο πολίτης μετά το τελευταίο bot.ask."""
    return ("dept", bot.last_dept) if bot.last_dept else dest_of(bot, it)


def evaluate(bot, verbose=False):
    classes = {key(c): c for c in bot.le.classes_}
    res = Counter()
    for r in csv.DictReader(open(EVAL, encoding="utf-8")):
        gold = classes.get(key(r["intent"])) if r["intent"] else None
        it, cf, out = bot.ask(r["text"], translate=False)
        answered = out in ("link", "phone")
        if gold is None:
            v = "σίγουρα λάθος" if answered else "σωστή άρνηση"
        else:
            gd = dest_of(bot, gold)
            if answered:
                v = "σωστό" if answer_dest(bot, it) == gd else "σίγουρα λάθος"
            elif out == "suggest":
                v = ("πρόταση με σωστό" if any(dest_of(bot, o) == gd for o in it)
                     else "πρόταση χωρίς σωστό")
            else:
                v = "άρνηση"
        res[v] += 1
        if verbose:
            shown = it[0] if isinstance(it, list) else it
            print(f"  {v:20} {out:8} {cf:4.2f}  {r['text'][:55]:57} → {shown}")
    return res


if __name__ == "__main__":
    bot = Bot()
    C.POOL_DESTINATIONS = False
    before = evaluate(bot)
    C.POOL_DESTINATIONS = True
    after = evaluate(bot, verbose=True)
    print()
    for k in ("σωστό", "πρόταση με σωστό", "πρόταση χωρίς σωστό", "άρνηση",
              "σωστή άρνηση", "σίγουρα λάθος"):
        print(f"  {k:20} {before[k]:3} → {after[k]:3}")
