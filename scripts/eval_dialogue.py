"""
Σύνδεση μηνυμάτων: σύγκριση στρατηγικών σε ζεύγη διαλόγου (Μ1 → Μ2).

Το context αποθηκεύεται ΜΟΝΟ όταν το Μ1 είναι αβέβαιο (πρόταση ή άρνηση).
Στρατηγικές (από την παρουσίαση «Σύνδεση μηνυμάτων»):
  A   χωρίς context: μόνο το Μ2
  B   πάντα Μ1+Μ2 όταν υπάρχει αποθηκευμένο Μ1
  B'  πρώτα το Μ2· αν δεν είναι σίγουρο, Μ1+Μ2, και κρατάμε ό,τι είναι πιο σίγουρο
  B'+ όπως B', αλλά η ΠΤΥΧΗ του Μ1 («τι δικαιολογητικά») περνάει στην απάντηση
      όταν το Μ2 δεν έχει δική του — το intent και η πτυχή λύνονται χωριστά
  E   ρωτάμε το llama αν σχετίζονται· RELATED → Μ1+Μ2, αλλιώς Μ2
  APP ο ΠΡΑΓΜΑΤΙΚΟΣ κώδικας του παραθύρου (context.Pending +
      connector.combine_with_context) — πρέπει να δίνει ό,τι και το B'+

Σωστό = σωστό intent με βεβαιότητα ≥ κατωφλίου ΚΑΙ, αν το Μ1 ρωτούσε κάτι
συγκεκριμένο («τι δικαιολογητικά»), η απάντηση κρατάει αυτή την πτυχή.
Ζημιά = ζεύγη που το A έλυνε σωστά και η στρατηγική τα χαλάει.

    python3 scripts/eval_dialogue.py            # χωρίς E (δεν χρειάζεται Ollama)
    python3 scripts/eval_dialogue.py --llama    # και με E
"""
import argparse, csv, sys, time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import requests
import connector as C
import context
from aspects import detect_aspect
from paths import DATASETS
from regression import Bot, key

EVAL = DATASETS / "dialogue_eval.csv"
RELATED_PROMPT = """Ένας πολίτης γράφει σε βοηθό του Δήμου Ηρακλείου.
Προηγούμενο μήνυμα: "{m1}"
Τρέχον μήνυμα: "{m2}"
Αφορούν τα δύο μηνύματα το ΙΔΙΟ αίτημα προς τον Δήμο; Απάντησε μόνο RELATED ή NOT_RELATED."""


def llama_related(m1, m2):
    r = requests.post(C.OLLAMA_URL if hasattr(C, "OLLAMA_URL") else "http://localhost:11434/api/generate",
                      json={"model": "llama3.1", "prompt": RELATED_PROMPT.format(m1=m1, m2=m2),
                            "stream": False, "options": {"temperature": 0}}, timeout=120)
    return "NOT" not in r.json().get("response", "").upper()


def run(bot, text):
    it, cf, out = bot.ask(text, translate=False)
    return {"it": it, "cf": cf, "out": out}


def strategy(name, bot, m1, m2, r1):
    stored = r1["out"] in ("suggest", "refuse")
    if name == "APP":
        pass
    elif not stored or name == "A":
        return run(bot, m2), False
    if name == "B":
        return run(bot, f"{m1} {m2}"), True
    if name in ("Bp", "Bpa"):
        r2 = run(bot, m2)
        carry = name == "Bpa"            # η πτυχή του Μ1 περνάει πάντα
        if r2["out"] in ("link", "phone"):
            return r2, carry
        rc = run(bot, f"{m1} {m2}")
        if rc["out"] in ("link", "phone") and rc["cf"] > r2["cf"]:
            return rc, True
        return r2, carry
    if name == "APP":
        ctx = context.Pending()
        if r1["out"] in ("suggest", "refuse"):
            ctx.remember(C.prepare_greek(m1), detect_aspect(C.prepare_greek(m1)))
        pending = ctx.take()
        g2 = C.prepare_greek(m2)
        ints = C.detect_intent(g2, bot.tok, bot.mdl, bot.le)
        _g, _i, used = C.combine_with_context(g2, ints, pending, bot.tok, bot.mdl, bot.le)
        r = run(bot, f"{m1} {m2}") if used else run(bot, m2)
        return r, pending is not None
    if name == "E":
        if llama_related(m1, m2):
            return run(bot, f"{m1} {m2}"), True
        return run(bot, m2), False


def judge(row, res, used_m1):
    allowed = {key(x) for x in row["expect"].split("|")}
    tops = res["it"] if isinstance(res["it"], list) else [res["it"]]
    if row["expect"] == "none":
        return "ok" if res["out"] in ("refuse", "suggest", "smalltalk") else "wrong"
    if res["out"] in ("link", "phone"):
        if key(tops[0]) not in allowed:
            return "wrong"
        aspect = detect_aspect(row["m2"]) or (detect_aspect(row["m1"]) if used_m1 else None)
        return "ok" if not row["aspect"] or aspect == row["aspect"] else "ok_no_aspect"
    if res["out"] == "suggest" and any(key(t) in allowed for t in tops):
        return "partial"
    return "miss"


def score_app(bot, path=None):
    """Για το regression: ποσοστό σωστών με τον πραγματικό κώδικα."""
    rows = list(csv.DictReader(open(path or EVAL, encoding="utf-8")))
    ok = 0
    for row in rows:
        r1 = run(bot, row["m1"])
        r, used = strategy("APP", bot, row["m1"], row["m2"], r1)
        ok += judge(row, r, used) == "ok"
    return ok / len(rows), len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    names = ["A", "B", "Bp", "Bpa", "APP"] + (["E"] if args.llama else [])
    bot = Bot()
    rows = list(csv.DictReader(open(EVAL, encoding="utf-8")))
    res = defaultdict(list)
    ms = Counter()
    for row in rows:
        r1 = run(bot, row["m1"])
        for n in names:
            t0 = time.monotonic()
            r, used = strategy(n, bot, row["m1"], row["m2"], r1)
            ms[n] += (time.monotonic() - t0) * 1000
            res[n].append(judge(row, r, used))
            if args.verbose and n != "A" and res[n][-1] != res["A"][-1]:
                print(f"  {n:2} {res['A'][-1]:>12} → {res[n][-1]:12} «{row['m1']}» + «{row['m2']}»")
    n = len(rows)
    print(f"\n  {n} διάλογοι · στρατηγική: σωστό / σωστό χωρίς πτυχή / μισό / δεν απάντησε / ΛΑΘΟΣ · βοήθησε / ζημιά · χρόνος")
    for s in names:
        c = Counter(res[s])
        good = lambda v: v == "ok"
        helped = sum(1 for a, b in zip(res["A"], res[s]) if not good(a) and good(b))
        harmed = sum(1 for a, b in zip(res["A"], res[s]) if good(a) and not good(b))
        print(f"   {s:2}  {c['ok']:2} / {c['ok_no_aspect']:2} / {c['partial']:2} / {c['miss']:2} / {c['wrong']:2}"
              f"   +{helped:2} / −{harmed:2}   {ms[s]/n:5.0f}ms")
    print("\n  ανά τύπο (σωστά):")
    types = sorted({r["type"] for r in rows})
    for t in types:
        idx = [i for i, r in enumerate(rows) if r["type"] == t]
        print(f"   {t[:38]:38} " + "  ".join(f"{s}:{sum(res[s][i]=='ok' for i in idx)}/{len(idx)}" for s in names))


if __name__ == "__main__":
    main()
