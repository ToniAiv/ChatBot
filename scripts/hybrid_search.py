"""
Υβριδική αναζήτηση στο Fetch: BERT + embeddings (bge-m3) + λεκτικό ταίριασμα.
Χωρίς LLM. Όλα μετριούνται ανά ΠΡΟΟΡΙΣΜΟ (σύνδεσμος υπηρεσίας ή τμήμα), όπως
το destination_confidence.

Κανάλια:
  bert   το σημερινό μοντέλο: άθροισμα πιθανοτήτων των top-5 intents ανά προορισμό
  knn    embeddings των 23.202 προτάσεων του dataset· οι 20 πιο κοντινές «ψηφίζουν»
  pages  embeddings των σελίδων των υπηρεσιών (τίτλος + Περιγραφή + Τι χρειάζεται)
  lex    TF-IDF (χαρακτήρες) πάνω στο ίδιο κείμενο των σελίδων
  rrf_*  συγχώνευση με Reciprocal Rank Fusion

Αποφάσεις (πότε απαντάμε σίγουρα):
  today  όπως σήμερα: τμήμα (department_override) ή βεβαιότητα BERT ≥ 0,80
  veto   όπως σήμερα, αλλά όχι σίγουρα όταν το knn δείχνει αλλού με μερίδιο ≥ v
  boost  όπως σήμερα, και επιπλέον σίγουρα όταν BERT και knn συμφωνούν
         (BERT ≥ τ, μερίδιο knn ≥ σ)
  Τα τ, σ, v ρυθμίζονται ΜΟΝΟ στο dev (tune).
  both   veto + boost

Σετ: dev = datasets/tests (ρύθμιση)· τελική μέτρηση = holdout, holdout2,
greeklish_eval. Η σύγκριση (agent/bakeoff_questions.csv) μένει έξω: τα t01–t30
είναι από το datasets/tests.

    PYTHONPATH=scripts:app python scripts/hybrid_search.py           (dev: ρύθμιση)
    PYTHONPATH=scripts:app python scripts/hybrid_search.py --final   (τελική μέτρηση)
"""
from __future__ import annotations

import csv, glob, hashlib, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "app")]
import connector as C
from regression import Bot, key

EMB_MODEL = "bge-m3"
OLLAMA_EMBED = "http://localhost:11434/api/embed"
CACHE = ROOT / "cache"          # στο .gitignore: ξαναφτιάχνεται σε ~8 λεπτά
K_NEIGHBOURS = 20
RRF_K = 60

bot = Bot()
C.SECTIONS.update(C.load_sections())
CLASSES = {key(c): c for c in bot.le.classes_}


def dest_of(intent: str):
    return C.destination_key(intent, bot.table, bot.ns) or ("intent", intent)


# ── embeddings ────────────────────────────────────────────────
def embed(texts: list, batch: int = 128) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch):
        r = requests.post(OLLAMA_EMBED, json={"model": EMB_MODEL, "input": texts[i:i + batch]}, timeout=600)
        r.raise_for_status()
        out += r.json()["embeddings"]
    m = np.asarray(out, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def cached_embed(name: str, texts: list) -> np.ndarray:
    h = hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()[:12]
    path = CACHE / f"{name}_{EMB_MODEL}_{h}.npy"
    if path.exists():
        return np.load(path)
    CACHE.mkdir(exist_ok=True)
    t0 = time.monotonic()
    m = embed(texts)
    np.save(path, m)
    print(f"  embeddings {name}: {len(texts)} κείμενα σε {time.monotonic() - t0:.0f} s", flush=True)
    return m


# ── ευρετήρια ─────────────────────────────────────────────────
def page_text(rec: dict) -> str:
    secs = rec.get("sections") or {}
    parts = [rec.get("title", ""), secs.get("Περιγραφή", ""), secs.get("Τι χρειάζεται", "")]
    return " ".join(p for p in parts if p)[:1500]


PAGES = [(("svc", url), page_text(rec)) for url, rec in C.SECTIONS.items()
         if rec.get("title") and "Καλώς" not in rec["title"]]
PAGE_DEST = [d for d, _ in PAGES]
PAGE_EMB = cached_embed("pages", [t for _, t in PAGES])
LEX = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
LEX_MAT = LEX.fit_transform([C.normalize(t) for _, t in PAGES])

_rows = [r for r in csv.reader(open(ROOT / "datasets/intent_dataset_v6_clean.csv", encoding="utf-8"))][1:]
SENTS = [(t, CLASSES[key(i)]) for t, i in _rows if key(i) in CLASSES]
SENT_DEST = [dest_of(i) for _, i in SENTS]
SENT_EMB = cached_embed("sentences", [t for t, _ in SENTS])


# ── κανάλια ───────────────────────────────────────────────────
def ranking(scores: dict) -> list:
    return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])]


_SEEN: dict = {}


def channels(text: str) -> dict:
    if text not in _SEEN:
        _SEEN[text] = _channels(text)
    return _SEEN[text]


def _channels(text: str) -> dict:
    gq, lang = C.resolve_query(text)
    intents = C.detect_intent(gq, bot.tok, bot.mdl, bot.le)
    bert = defaultdict(float)
    for i, p in intents:
        bert[dest_of(i)] += p
    q = embed([gq])[0]
    sims = SENT_EMB @ q
    top = np.argsort(-sims)[:K_NEIGHBOURS]
    knn = defaultdict(float)
    for j in top:
        knn[SENT_DEST[j]] += float(sims[j])
    psims = PAGE_EMB @ q
    pages = {PAGE_DEST[j]: float(psims[j]) for j in np.argsort(-psims)[:K_NEIGHBOURS]}
    lsims = (LEX_MAT @ LEX.transform([C.normalize(gq)]).T).toarray().ravel()
    lex = {PAGE_DEST[j]: float(lsims[j]) for j in np.argsort(-lsims)[:K_NEIGHBOURS]}
    out = {"lang": lang, "intents": intents, "dept": C.department_override(gq, intents),
           "bert_conf": C.destination_confidence(intents, bot.table, bot.ns),
           "rank": {"bert": ranking(bert), "knn": ranking(knn), "pages": ranking(pages), "lex": ranking(lex)},
           "knn_share": max(knn.values()) / sum(knn.values())}
    if lang == "en":                      # τα αγγλικά και χωρίς μετάφραση (το bge-m3 είναι πολύγλωσσο)
        q2 = embed([text])[0]
        s2 = SENT_EMB @ q2
        k2 = defaultdict(float)
        for j in np.argsort(-s2)[:K_NEIGHBOURS]:
            k2[SENT_DEST[j]] += float(s2[j])
        out["rank"]["knn_raw_en"] = ranking(k2)
    for name, chans in (("rrf_bk", ("bert", "knn")), ("rrf_bkp", ("bert", "knn", "pages")),
                        ("rrf_all", ("bert", "knn", "pages", "lex"))):
        f = defaultdict(float)
        for c in chans:
            for r, d in enumerate(out["rank"][c]):
                f[d] += 1 / (RRF_K + r + 1)
        out["rank"][name] = ranking(f)
    return out


def decide(ch: dict, rule: str, tau: float = 0.5, sigma: float = 0.5, v: float = 0.0):
    """(σίγουρος προορισμός ή None, λίστα για κουμπιά)."""
    bert_top = ch["rank"]["bert"][0]
    knn_top = ch["rank"]["knn"][0]
    buttons = ch["rank"]["rrf_bk" if rule != "today" else "bert"][:3]
    if ch["dept"]:
        return ("dept", ch["dept"]), buttons
    sure = ch["bert_conf"] >= C.MIN_BERT_CONFIDENCE
    agree = bert_top == knn_top
    if rule in ("veto", "both") and sure and not agree and ch["knn_share"] >= v:
        sure = False
    if rule in ("boost", "both") and not sure and agree and ch["bert_conf"] >= tau and ch["knn_share"] >= sigma:
        sure = True
    return (bert_top if sure else None), buttons


# ── σετ ───────────────────────────────────────────────────────
def gold(expect: str, qtype: str):
    if qtype == "department":
        return ("dept", expect)
    c = CLASSES.get(key(expect))
    return dest_of(c) if c else None


def load_sets(final: bool) -> dict:
    if not final:
        rows = [r for f in sorted(glob.glob(str(ROOT / "datasets/tests/*.csv")))
                for r in csv.DictReader(open(f, encoding="utf-8"))]
        return {"dev": [{"id": f"d{i}", "text": r["text"], "type": "service", "expect": r["intent"], "lang": "el"}
                        for i, r in enumerate(rows)]}
    sets = {}
    for name in ("holdout", "holdout2"):
        sets[name] = [{"id": r["id"], "text": r["text"], "type": r["type"], "expect": r["expect"], "lang": r["lang"]}
                      for r in csv.DictReader(open(ROOT / f"agent/{name}_questions.csv", encoding="utf-8"))
                      if "|" not in r["expect"]]          # τα σύνθετα θέλουν χωρισμό, όχι αναζήτηση
    sets["greeklish_eval"] = [{"id": f"g{i:02d}", "text": r["text"], "type": "service", "expect": r["intent"],
                               "lang": "en" if r.get("kind") == "english" else "greeklish"}
                              for i, r in enumerate(csv.DictReader(open(ROOT / "datasets/greeklish_eval.csv",
                                                                        encoding="utf-8")), 1)]
    return sets


RULES = ("today", "veto", "boost", "both")
CHANNELS = ("bert", "knn", "pages", "lex", "rrf_bk", "rrf_bkp", "rrf_all")


def evaluate(qs: list, tau: float, sigma: float, v: float) -> dict:
    res = {"n": 0, "rank": {c: [0, 0] for c in CHANNELS + ("knn_raw_en",)}, "n_en": 0,
           "rules": {r: {"sure": 0, "sure_ok": 0, "buttons_ok": 0, "oos_sure": 0} for r in RULES},
           "n_oos": 0, "by_lang": defaultdict(lambda: {"n": 0, "bert": 0, "knn": 0, "rrf_bk": 0}), "rows": []}
    for q in qs:
        ch = channels(q["text"])
        is_oos = q["type"] in ("out_of_scope", "smalltalk")
        g = None if is_oos else gold(q["expect"], q["type"])
        row = {"id": q["id"], "text": q["text"], "gold": g, "lang": ch["lang"], "bert_conf": round(ch["bert_conf"], 3),
               "knn_share": round(ch["knn_share"], 3), **{f"top_{c}": ch["rank"][c][0] if ch["rank"][c] else None
                                                          for c in CHANNELS}}
        if is_oos:
            res["n_oos"] += 1
            for r in RULES:
                res["rules"][r]["oos_sure"] += decide(ch, r, tau, sigma, v)[0] is not None
            res["rows"].append(row)
            continue
        if g is None:
            continue
        res["n"] += 1
        for c in CHANNELS + ("knn_raw_en",):
            if c in ch["rank"]:
                res["rank"][c][0] += ch["rank"][c][:1] == [g]
                res["rank"][c][1] += g in ch["rank"][c][:3]
        res["n_en"] += "knn_raw_en" in ch["rank"]
        bl = res["by_lang"][q["lang"]]
        bl["n"] += 1
        for c in ("bert", "knn", "rrf_bk"):
            bl[c] += ch["rank"][c][:1] == [g]
        for r in RULES:
            sure, buttons = decide(ch, r, tau, sigma, v)
            st = res["rules"][r]
            if sure is not None:
                st["sure"] += 1
                st["sure_ok"] += sure == g
            else:
                st["buttons_ok"] += g in buttons
            row[f"{r}_sure"] = sure is not None
            row[f"{r}_ok"] = (sure == g) if sure is not None else (g in buttons)
        res["rows"].append(row)
    return res


def report(name: str, res: dict):
    n = res["n"]
    pct = lambda a, b: f"{100 * a / b:5.1f}%" if b else "  -  "
    print(f"\n=== {name}: {n} ερωτήσεις υπηρεσίας, {res['n_oos']} εκτός θέματος")
    print("  κανάλι      top-1    σωστό στα 3")
    for c in CHANNELS:
        t1, t3 = res["rank"][c]
        print(f"  {c:10} {pct(t1, n)}   {pct(t3, n)}")
    if res["n_en"]:
        t1, t3 = res["rank"]["knn_raw_en"]
        print(f"  knn αγγλικά χωρίς μετάφραση ({res['n_en']}): top-1 {pct(t1, res['n_en'])}, στα 3 {pct(t3, res['n_en'])}")
    print("  απόφαση   σίγουρες  ακρίβεια  σίγουρα λάθη  σωστό στα κουμπιά  σίγουρες σε εκτός θέματος")
    for r in RULES:
        st = res["rules"][r]
        rest = n - st["sure"]
        print(f"  {r:8}  {pct(st['sure'], n)}   {pct(st['sure_ok'], st['sure'])}   {st['sure'] - st['sure_ok']:4}"
              f"          {pct(st['buttons_ok'], rest)} ({st['buttons_ok']}/{rest})"
              f"      {st['oos_sure']}/{res['n_oos']}")
    print("  ανά γλώσσα (top-1):", {k: {c: f"{v[c]}/{v['n']}" for c in ("bert", "knn", "rrf_bk")}
                                    for k, v in res["by_lang"].items()})


def tune(dev: list) -> tuple:
    """Στο dev μόνο.
    boost: η μεγαλύτερη κάλυψη όπου οι ΕΠΙΠΛΕΟΝ σίγουρες έχουν ακρίβεια ≥ τη σημερινή.
           (Το «καθόλου επιπλέον λάθη» είναι αδύνατο: κάθε νέα απάντηση έχει κάποιο ρίσκο.)
    veto:  το μικρότερο v όπου οι σίγουρες που απορρίπτονται είναι λάθος ≥ 50% των φορών.
           (Χωρίς όριο, το veto έκοβε 165 σωστές για να γλιτώσει 35 λάθη.)"""
    chs = [(channels(q["text"]), gold(q["expect"], "service")) for q in dev]
    chs = [(c, g) for c, g in chs if g is not None]

    def stats(rule, tau=0.0, sigma=0.0, v=0.0):
        sure = [(decide(c, rule, tau, sigma, v)[0], g) for c, g in chs]
        return sum(s is not None for s, _ in sure), sum(s is not None and s != g for s, g in sure)

    base_sure, base_err = stats("today")
    base_prec = 1 - base_err / base_sure
    print(f"  dev σήμερα: σίγουρες {base_sure}/{len(chs)}, σίγουρα λάθη {base_err}, ακρίβεια {base_prec:.3f}")
    best_boost = (1.0, 1.0, base_sure)                    # (1, 1) = καθόλου boost
    for tau in (0.3, 0.4, 0.5, 0.6, 0.7):
        for sigma in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            s, e = stats("boost", tau, sigma)
            added, added_err = s - base_sure, e - base_err
            ok = added > 0 and 1 - added_err / added >= base_prec
            print(f"    boost τ={tau} σ={sigma}: +{added} σίγουρες, +{added_err} λάθη{'  ✓' if ok else ''}")
            if ok and s > best_boost[2]:
                best_boost = (tau, sigma, s)
    best_v = 1.01                                          # > 1 = καθόλου veto
    for v in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        s, e = stats("veto", v=v)
        removed, removed_err = base_sure - s, base_err - e
        rate = removed_err / removed if removed else 0
        print(f"    veto v={v}: −{removed} σίγουρες, από αυτές λάθος {removed_err} ({rate:.0%})")
        if removed and rate >= 0.5 and v < best_v:
            best_v = v
    print(f"  επιλογή: τ={best_boost[0]} σ={best_boost[1]} v={best_v}")
    return best_boost[0], best_boost[1], best_v


def main():
    final = "--final" in sys.argv
    params = ROOT / "reports/hybrid_params.json"
    if not final:
        dev = load_sets(False)["dev"]
        tau, sigma, v = tune(dev)
        json.dump({"tau": tau, "sigma": sigma, "v": v, "k": K_NEIGHBOURS, "model": EMB_MODEL}, open(params, "w"), indent=1)
        res = evaluate(dev, tau, sigma, v)
        report("dev (datasets/tests)", res)
        out = {"dev": res}
    else:
        p = json.load(open(params))
        tau, sigma, v = p["tau"], p["sigma"], p["v"]
        out = {}
        for name, qs in load_sets(True).items():
            out[name] = evaluate(qs, tau, sigma, v)
            report(name, out[name])
    tag = "final" if final else "dev"
    with open(ROOT / f"reports/hybrid_{tag}_rows.csv", "w", encoding="utf-8", newline="") as f:
        rows = [dict(set=s, **r) for s, res in out.items() for r in res["rows"]]
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: (k != "set", k)))
        w.writeheader(); w.writerows(rows)
    json.dump({s: {k: v for k, v in res.items() if k != "rows"} for s, res in out.items()},
              open(ROOT / f"reports/hybrid_{tag}_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=str)
    print(f"\nαποθηκεύτηκαν: reports/hybrid_{tag}_summary.json, reports/hybrid_{tag}_rows.csv")


if __name__ == "__main__":
    main()
