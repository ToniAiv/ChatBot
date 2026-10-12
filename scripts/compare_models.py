"""
Σύγκριση δύο μοντέλων στις ΙΔΙΕΣ ερωτήσεις, με την ίδια λογική με το bot.

ΓΙΑΤΙ: μετά από επανεκπαίδευση το σύνολο των intents αλλάζει (στο v6:
+19, −3), οπότε ένα σκέτο «ποσοστό στο test set» δεν συγκρίνεται. Οι
ερωτήσεις χωρίζονται σε:
  κοινά      intents που υπάρχουν και στα δύο μοντέλα → η δίκαιη σύγκριση
  νέα        intents μόνο του νέου μοντέλου (π.χ. φυσικοθεραπεία)
  αφαιρέθηκαν intents μόνο του παλιού (π.χ. στέγαστρο ΟΑΣΑ)

    python3 scripts/compare_models.py models/intent-model-218 models/intent-model-234
"""
import csv, glob, json, random, re, sys, time, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import connector as C
from paths import DATASETS, ROOT
from regression import Bot, key


def load_tests():
    rows = []
    for f in sorted(glob.glob(str(DATASETS / "tests" / "*.csv"))):
        rows += [dict(r, file=Path(f).name) for r in csv.DictReader(open(f, encoding="utf-8")) if r.get("text")]
    return rows


def run_model(model_dir, rows):
    C.MODEL_DIR = ROOT / model_dir
    bot = Bot()
    classes = {key(c): c for c in bot.le.classes_}
    cache = {}

    def dest(intent):
        c = classes.get(key(intent))
        if c is None:
            return None
        if c not in cache:
            if c in bot.ns:
                cache[c] = "dept:" + (bot.ns[c] or {}).get("name", "")
            else:
                cand, _ = C.find_candidates(c, bot.vec, bot.mat, bot.valid, bot.table)
                cache[c] = cand[0][1]["url"] if cand and cand[0][0] >= C.MIN_TFIDF_SCORE else None
        return cache[c]

    out = []
    t0 = time.monotonic()
    for r in rows:
        gold = key(r["intent"])
        known = gold in classes
        tops = C.detect_intent(C.prepare_greek(r["text"]), bot.tok, bot.mdl, bot.le)
        it, cf, outcome = bot.ask(r["text"], translate=False)
        gd = dest(r["intent"]) if known else None
        if outcome == "suggest":
            reach = gd is not None and gd in {dest(x) for x in it}
            answered_ok = None
        elif outcome in ("link", "phone"):
            got = ("dept:" + bot.depts[bot.last_dept]["name"]) if bot.last_dept else dest(it)
            reach = answered_ok = gd is not None and got == gd
        else:
            reach, answered_ok = False, None
        out.append({"known": known, "top1": key(tops[0][0]) == gold, "conf": tops[0][1],
                    "outcome": outcome, "reach": reach, "answered_ok": answered_ok,
                    "pred": tops[0][0], "gold": r["intent"], "file": r["file"]})
    ms = (time.monotonic() - t0) / len(rows) * 1000
    return bot, out, ms, set(classes)


def summary(res, idx):
    sub = [res[i] for i in idx]
    n = len(sub)
    if not n:
        return {}
    answered = [x for x in sub if x["outcome"] in ("link", "phone")]
    return {"n": n,
            "top1": sum(x["top1"] for x in sub) / n,
            "κάλυψη": len(answered) / n,
            "προσεγγίσιμο": sum(x["reach"] for x in sub) / n,
            "ακρίβεια_όταν_απαντά": (sum(x["answered_ok"] for x in answered) / len(answered)) if answered else 0,
            "λάθος_σίγουρα": sum(1 for x in answered if not x["answered_ok"]) / n}


def main():
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    rows = load_tests()
    bot_a, res_a, ms_a, cls_a = run_model(a_dir, rows)
    bot_b, res_b, ms_b, cls_b = run_model(b_dir, rows)
    gold = [key(r["intent"]) for r in rows]
    groups = {
        "κοινά intents": [i for i, g in enumerate(gold) if g in cls_a and g in cls_b],
        "νέα intents (μόνο νέο)": [i for i, g in enumerate(gold) if g in cls_b and g not in cls_a],
        "αφαιρέθηκαν (μόνο παλιό)": [i for i, g in enumerate(gold) if g in cls_a and g not in cls_b],
    }
    print(f"\n  {a_dir} ({len(cls_a)} κλάσεις, {ms_a:.0f} ms/ερώτηση)  vs  {b_dir} ({len(cls_b)} κλάσεις, {ms_b:.0f} ms/ερώτηση)")
    for name, idx in groups.items():
        sa, sb = summary(res_a, idx), summary(res_b, idx)
        print(f"\n  ── {name} (n={len(idx)})")
        for m in ("top1", "κάλυψη", "προσεγγίσιμο", "ακρίβεια_όταν_απαντά", "λάθος_σίγουρα"):
            va, vb = sa.get(m, 0), sb.get(m, 0)
            print(f"     {m:22} {va:7.1%} → {vb:7.1%}   ({(vb - va) * 100:+.1f})")
    # ανά intent, στα κοινά: τι κέρδισε / έχασε (προσεγγίσιμο)
    per = defaultdict(lambda: [0, 0, 0])
    for i in groups["κοινά intents"]:
        g = rows[i]["intent"]; per[g][0] += res_a[i]["reach"]; per[g][1] += res_b[i]["reach"]; per[g][2] += 1
    diffs = sorted(((b - a, g, a, b, n) for g, (a, b, n) in per.items() if a != b))
    print("\n  κοινά intents — μεγαλύτερες ΑΠΩΛΕΙΕΣ (προσεγγίσιμο):")
    for d, g, a, b, n in diffs[:10]:
        print(f"     {a:2} → {b:2} /{n}  {g}")
    print("  κοινά intents — μεγαλύτερα ΚΕΡΔΗ:")
    for d, g, a, b, n in diffs[::-1][:10]:
        print(f"     {a:2} → {b:2} /{n}  {g}")
    # νέα intents: ανά intent + με τι μπερδεύονται
    print("\n  νέα intents — ανά intent (σωστό top-1 / προσεγγίσιμο):")
    per_new = defaultdict(lambda: [0, 0, 0]); conf = Counter()
    for i in groups["νέα intents (μόνο νέο)"]:
        g = rows[i]["intent"]; x = res_b[i]
        per_new[g][0] += x["top1"]; per_new[g][1] += x["reach"]; per_new[g][2] += 1
        if not x["top1"]:
            conf[(g, x["pred"])] += 1
    for g, (t, r, n) in sorted(per_new.items(), key=lambda kv: kv[1][1]):
        print(f"     {t}/{n}  {r}/{n}  {g}")
    print("  νέα intents — συχνότερες συγχύσεις:")
    for (g, p), c in conf.most_common(8):
        print(f"     {c}× {g} → {p}")
    # τι έκανε το παλιό με τα νέα
    old_new = Counter(res_a[i]["outcome"] for i in groups["νέα intents (μόνο νέο)"])
    print(f"\n  το ΠΑΛΙΟ μοντέλο στις ερωτήσεις των νέων intents: {dict(old_new)}")
    json.dump({"groups": {k: [summary(res_a, v), summary(res_b, v)] for k, v in groups.items()}},
              open(ROOT / "reports" / "model_comparison.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
