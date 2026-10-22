"""
Δεύτερο σετ 100 διαλόγων για τη μνήμη υπηρεσίας (agent/followup100b_questions.csv).

Οι διατυπώσεις των ερωτήσεων συνέχειας τις γράφει ΕΝΑ LLM (Krikri), όχι ο
άνθρωπος που γράφει τους κανόνες του aspects.py, ώστε οι κανόνες να μην
προσαρμόζονται στο τεστ. Το σετ ανεβαίνει στο git πριν από τους κανόνες.
Υπηρεσίες, είδη και διατυπώσεις επιλέγονται τυχαία (seed 200). Οι σωστές
απαντήσεις βγαίνουν αυτόματα από τη σελίδα (make_followup100.gold).

  70 ερωτήσεις συνέχειας (60 ελληνικά, 10 αγγλικά), 7 είδη × 10
  15 αλλαγές θέματος (ερωτήσεις των holdout)
  15 εκτός θέματος (επίσης από το LLM)

    PYTHONPATH=scripts:app python scripts/make_followup100b.py                       (100b: seed 200, οδηγία A)
    PYTHONPATH=scripts:app python scripts/make_followup100b.py --seed 300 --out c   (100c: ρύθμιση embeddings)
    PYTHONPATH=scripts:app python scripts/make_followup100b.py --seed 400 --out d --variant B   (100d: τεστ)
"""
from __future__ import annotations

import argparse, csv, json, random, sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "app")]
from make_followup100 import gold
from regression import key

SEED = 200
MODEL = "hf.co/ilsp/Llama-Krikri-8B-Instruct-GGUF:Q4_K_M"
ASPECTS = {
    "docs": "τι δικαιολογητικά ή έγγραφα χρειάζονται",
    "online": "αν γίνεται ηλεκτρονικά ή πρέπει να πάει αυτοπροσώπως",
    "contact": "πώς να επικοινωνήσει: τηλέφωνο, email, πού βρίσκεται το γραφείο, με ποιον να μιλήσει",
    "cost": "αν κοστίζει κάτι ή πόσο πληρώνει",
    "time": "προθεσμίες ή πόσο χρόνο παίρνει",
    "who": "ποιος μπορεί να κάνει την αίτηση ή ποιοι δικαιούνται",
    "process": "τι γίνεται μετά την αίτηση ή πώς προχωρά η διαδικασία",
}
PROMPT = {
    "el": "Ένας πολίτης μιλά με τον εικονικό βοηθό του Δήμου για μια υπηρεσία που ΗΔΗ ανέφερε. Γράψε {n} διαφορετικές, "
          "σύντομες, καθημερινές ερωτήσεις συνέχειας που ρωτούν {what}. Χωρίς να ονομάζουν την υπηρεσία. Ποικιλία στη "
          "διατύπωση, κάποιες χωρίς τόνους, όπως γράφουν οι πολίτες στο κινητό.",
    "en": "A citizen is chatting with a municipality's virtual assistant about a service they ALREADY mentioned. Write {n} "
          "different, short, everyday follow-up questions in English asking {what}. Do not name the service.",
}
EN_WHAT = {"docs": "what documents are needed", "online": "whether it can be done online", "contact": "how to contact them",
           "cost": "whether it costs anything", "time": "deadlines or how long it takes", "who": "who can apply",
           "process": "what happens after applying"}
PROMPT_B = {
    "el": "Φαντάσου ότι είσαι πολίτης που μόλις έμαθε από τον βοηθό του Δήμου ποια υπηρεσία χρειάζεται. Γράψε {n} "
          "μηνύματα που θα έστελνες μετά, για να ρωτήσεις {what}. Γράψε όπως μιλάνε οι άνθρωποι, με διαφορετικές λέξεις "
          "το καθένα, μερικά πολύ σύντομα, μερικά με ορθογραφικά ή χωρίς τόνους. Μην αναφέρεις το όνομα της υπηρεσίας.",
    "en": "Imagine you are a resident who was just told by a municipality chatbot which service you need. Write {n} "
          "messages you would send next to ask {what}. Write like people actually talk, each with different words, "
          "some very short. Do not name the service.",
}
OOS_PROMPT_B = ("Γράψε {n} σύντομα μηνύματα προς έναν βοηθό του Δήμου που ΔΕΝ ζητούν τίποτα από τον Δήμο: χαιρετισμοί, "
                "ευχαριστίες, αστεία, ερωτήσεις για καιρό, αθλητικά ή τηλεόραση, ή προσπάθειες να τον βάλουν να κάνει "
                "κάτι άσχετο. Κανένα να μην αφορά υπηρεσίες, άδειες, πληρωμές ή προβλήματα στην πόλη.")
OOS_PROMPT = ("Γράψε {n} διαφορετικά, σύντομα μηνύματα που θα έγραφε κάποιος σε έναν βοηθό του Δήμου Ηρακλείου και ΔΕΝ "
              "είναι αιτήματα προς τον Δήμο: ευχαριστίες, χαιρετισμοί, ερωτήσεις για άλλα θέματα (καιρός, αθλητικά, "
              "τράπεζες), ή απόπειρες να του αλλάξουν ρόλο.")


def generate(prompt: str, seed: int = SEED) -> list:
    r = requests.post("http://localhost:11434/api/chat", json={
        "model": MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False,
        "format": {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": {"type": "string"}}}},
        "options": {"temperature": 0.7, "seed": seed, "num_predict": 800}}, timeout=300)
    r.raise_for_status()
    return [s.strip() for s in json.loads(r.json()["message"]["content"])["items"] if s.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="b")
    ap.add_argument("--variant", default="A", choices=["A", "B"])
    args = ap.parse_args()
    seed, prefix = args.seed, ("y" if args.out == "b" else args.out)
    rnd = random.Random(seed)
    prompt, oos_prompt = (PROMPT, OOS_PROMPT) if args.variant == "A" else (PROMPT_B, OOS_PROMPT_B)
    pools = {("el", a): generate(prompt["el"].format(n=10, what=w), seed) for a, w in ASPECTS.items()}
    pools.update({("en", a): generate(prompt["en"].format(n=5, what=w), seed) for a, w in EN_WHAT.items()})
    oos = generate(oos_prompt.format(n=15), seed)
    sections = json.load(open(ROOT / "data/service_sections.json", encoding="utf-8"))
    table = list(csv.DictReader(open(ROOT / "mappings/intent_to_service.csv", encoding="utf-8")))
    rows = [r for r in csv.reader(open(ROOT / "datasets/intent_dataset_v6_clean.csv", encoding="utf-8"))][1:]
    by_intent = {}
    for t, i in rows:
        by_intent.setdefault(key(i), []).append(t)
    services = sorted({(r["intent"], r["url"]) for r in table
                       if r["status"] in ("verified", "auto") and r["url"] in sections and key(r["intent"]) in by_intent
                       and "Καλώς" not in sections[r["url"]].get("title", "")})
    el_plan = [("el", a) for a in ASPECTS for _ in range(9)]          # 63 → 60, ισομερώς στα 7 είδη
    rnd.shuffle(el_plan)
    plan = el_plan[:60] + [("en", a) for a in sorted(EN_WHAT)] + [("en", a) for a in rnd.sample(sorted(EN_WHAT), 3)]
    rnd.shuffle(plan)
    out = []
    for n, ((lang, aspect), (intent, url)) in enumerate(zip(plan, rnd.sample(services, 70)), 1):
        out.append({"id": f"{prefix}{n:03d}", "m1": rnd.choice(by_intent[key(intent)]), "m2": rnd.choice(pools[(lang, aspect)]),
                    "type": "followup", "expect": intent, "facts": gold(aspect, url, sections[url]),
                    "note": f"{aspect} {lang}", "m1_intent": intent})
    others = [r for f in ("holdout_questions", "holdout2_questions")
              for r in csv.DictReader(open(ROOT / f"agent/{f}.csv", encoding="utf-8"))
              if r["type"] == "service" and "|" not in r["expect"]]
    for n, (intent, url) in enumerate(rnd.sample(services, 15), 71):
        q = rnd.choice([o for o in others if key(o["expect"]) != key(intent)])
        out.append({"id": f"{prefix}{n:03d}", "m1": rnd.choice(by_intent[key(intent)]), "m2": q["text"], "type": "switch",
                    "expect": q["expect"], "facts": "", "note": f"νέο θέμα ({q['id']})", "m1_intent": intent})
    for n, ((intent, url), text) in enumerate(zip(rnd.sample(services, 15), (oos * 2)[:15]), 86):
        out.append({"id": f"{prefix}{n:03d}", "m1": rnd.choice(by_intent[key(intent)]), "m2": text, "type": "oos",
                    "expect": "", "facts": "", "note": "εκτός θέματος", "m1_intent": intent})
    path = ROOT / f"agent/followup100{args.out}_questions.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"{len(out)} διάλογοι → {path.relative_to(ROOT)}  (διατυπώσεις: {sum(len(v) for v in pools.values())}, εκτός θέματος: {len(oos)})")


if __name__ == "__main__":
    main()
