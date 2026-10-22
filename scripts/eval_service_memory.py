"""
Μνήμη υπηρεσίας (context.Service): το πραγματικό gui.Api με και χωρίς τη μνήμη.

  1. Διάλογοι συνέχειας (agent/followup_questions.csv, agent/followup2_questions.csv):
     Μ1 ονομάζει υπηρεσία, Μ2 ρωτά κάτι γι' αυτήν. Αν το Μ1 δίνει κουμπιά, ο
     «πολίτης» πατά το σωστό (Api.choose).
       followup        σωστό αν η απάντηση έχει ≥ 50% από τις πληροφορίες της σελίδας
       δεν αναφέρεται  σωστό αν παραπέμπει τίμια (σελίδα/τμήμα) χωρίς αριθμούς εκτός σελίδας
       switch          σωστό αν δίνει τον προορισμό του νέου θέματος
       oos             σωστό αν δεν προσθέτει σύνδεσμο άλλης υπηρεσίας
  2. Οι 63 διάλογοι του dialogue_eval / dialogue_holdout: ποιες απαντήσεις στο Μ2 αλλάζουν.

    PYTHONPATH=scripts:app python scripts/eval_service_memory.py
"""
from __future__ import annotations

import csv, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "app")]
import connector as C
import context
import gui
import usage_log
from regression import key

usage_log.ENABLED = False                 # ερωτήσεις δοκιμής, όχι πολιτών
gui._load_pipeline()
S = gui._STATE
CLASSES = {key(c): c for c in S["label_encoder"].classes_}
URL = re.compile(r"https?://[^\s)\]»\"'<>]+")


def _n(s: str) -> str:
    return re.sub(r"\s+", " ", C.normalize(s)).strip()


def target(intent: str) -> dict:
    """Σελίδα (url, κείμενο) ή τμήμα (τηλέφωνο) του intent."""
    c = CLASSES[key(intent)]
    if c in (S["no_service"] or {}):
        d = S["no_service"][c] or {}
        return {"url": "", "phone": re.sub(r"\D", "", d.get("tel", ""))[:10], "text": " ".join(map(str, d.values()))}
    cand, _src = C.find_candidates(c, S["vectorizer"], S["tfidf_matrix"], S["valid_kb"], S["service_table"])
    url = cand[0][1]["url"] if cand else ""
    rec = C.SECTIONS.get(url) or {}
    return {"url": url, "phone": "", "text": " ".join([rec.get("title", "")] + list((rec.get("sections") or {}).values()))}


def converse(m1: str, m2: str, m1_intent: str, memory: bool) -> tuple:
    context.SERVICE_MEMORY = memory
    api = gui.Api()
    first = api.chat(m1)
    if first["options"] and m1_intent:          # ο «πολίτης» πατά το κουμπί της υπηρεσίας του Μ1
        c = CLASSES.get(key(m1_intent))
        if any(o["intent"] == c for o in first["options"]):
            first = api.choose(c)
    return first["text"], api.chat(m2)["text"]


def score(q: dict, first: str, text: str) -> bool:
    if q["type"] == "followup":
        page = target(q["expect"])
        if q["facts"] != "-":
            groups = [g for g in q["facts"].split("|") if g]
            hit = sum(any(_n(a) in _n(text) for a in g.split("/")) for g in groups)
            return hit / len(groups) >= 0.5
        pd = re.sub(r"\D", "", page["text"])
        invented = [x for x in re.findall(r"\d[\d.,]*", text)
                    if re.sub(r"\D", "", x) not in pd and re.sub(r"\D", "", x) not in re.sub(r"\D", "", page["url"])]
        helpful = (page["url"] and page["url"] in text) or "Τηλ" in text or "@" in text
        return bool(helpful) and not invented
    if q["type"] == "switch":
        t = target(q["expect"])
        return bool((t["url"] and t["url"] in text) or (t["phone"] and t["phone"][-6:] in re.sub(r"\D", "", text)))
    return bool(text.strip()) and not (set(URL.findall(text)) - set(URL.findall(first)))


def followups() -> dict:
    out, rows = {}, []
    for name in ("followup_questions", "followup2_questions", "followup100_questions"):
        qs = list(csv.DictReader(open(ROOT / f"agent/{name}.csv", encoding="utf-8")))
        res = {}
        for q in qs:
            kind = q["type"] if q["type"] != "followup" else ("followup" if q["facts"] != "-" else "δεν αναφέρεται")
            m1_intent = q.get("m1_intent") or (q["expect"] if q["type"] == "followup" else "")
            marks = []
            for memory in (False, True):
                first, text = converse(q["m1"], q["m2"], m1_intent, memory)
                ok = score(q, first, text)
                r = res.setdefault(("με" if memory else "χωρίς"), {})
                r.setdefault(kind, [0, 0])
                r[kind][0] += ok; r[kind][1] += 1
                marks.append("✓" if ok else "✗")
                rows.append({"set": name, "id": q["id"], "type": kind, "note": q.get("note", ""), "memory": memory,
                             "m1": q["m1"], "m2": q["m2"], "correct": ok, "answer": text})
            print(f"  {q['id']} χωρίς:{marks[0]} με:{marks[1]}  {q['m2'][:50]}", flush=True)
        out[name] = res
        print(name, json.dumps(res, ensure_ascii=False), flush=True)
    with open(ROOT / "reports/service_memory_answers.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return out


def dialogues() -> list:
    changed = []
    for f in ("dialogue_eval.csv", "dialogue_holdout.csv"):
        for i, r in enumerate(csv.DictReader(open(ROOT / "datasets" / f, encoding="utf-8"))):
            _, off = converse(r["m1"], r["m2"], "", False)
            _, on = converse(r["m1"], r["m2"], "", True)
            if off != on:
                changed.append({"set": f, "i": i, "m1": r["m1"], "m2": r["m2"], "expect": r["expect"],
                                "without": off[:200], "with": on[:200]})
    print(f"\nδιάλογοι dialogue_eval/holdout: άλλαξε η απάντηση σε {len(changed)} από 63")
    for c in changed:
        print(f"  {c['set'][:15]} #{c['i']}: «{c['m1']}» → «{c['m2']}»\n    χωρίς: {c['without'][:110]!r}\n    με:    {c['with'][:110]!r}")
    return changed


def main():
    summary = {"followups": followups(), "dialogue_changes": dialogues()}
    json.dump(summary, open(ROOT / "reports/service_memory_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nαποθηκεύτηκαν: reports/service_memory_summary.json, reports/service_memory_answers.csv")


if __name__ == "__main__":
    main()
