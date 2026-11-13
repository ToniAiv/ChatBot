"""
Μετρήσεις του multi-agent συστήματος. Αποτελέσματα: reports/multiagent.md.

    venv/bin/python -m multiagent.evaluate security     ο Security μόνος του (multiagent/sets/security_questions.csv)
    venv/bin/python -m multiagent.evaluate validator    ο Validator μόνος του, πάνω στους υποψήφιους του BERT (bakeoff)
    venv/bin/python -m multiagent.evaluate pipeline     όλη η ροή, δίπλα στο σημερινό bot (bakeoff, compound, security)
    venv/bin/python -m multiagent.evaluate summary      πίνακες από τα αρχεία των παραπάνω

    --configs tev1-el,qwen4b     ποιες ρυθμίσεις (βλ. CONFIGS)
    --sets bakeoff,compound      ποια σετ, για το pipeline

Κάθε γραμμή γράφεται αμέσως στο reports/multiagent_*.csv. Αν η μέτρηση
διακοπεί, η επόμενη εκτέλεση συνεχίζει από εκεί που έμεινε.
Η βαθμολόγηση είναι αυτή του agent/bakeoff.py, ίδια για το bot και για το
multi-agent. Θέλει το Ollama ανοιχτό.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
import sys
import time

from multiagent import ROOT, classifier, config, pipeline, security, validator
from multiagent.state import Segment, Session

sys.path.insert(0, str(ROOT / "agent"))
import bakeoff as B          # noqa: E402  (βαθμολόγηση· φορτώνει και δεύτερο αντίγραφο του BERT)
import connector as C        # noqa: E402
import gui                   # noqa: E402
import usage_log             # noqa: E402

usage_log.ENABLED = False    # ερωτήσεις δοκιμής, όχι πολιτών

REPORTS = ROOT / "reports"
SETS = {"bakeoff": ROOT / "agent" / "bakeoff_questions.csv",
        "compound": ROOT / "multiagent" / "sets" / "compound_questions.csv",
        "security": ROOT / "multiagent" / "sets" / "security_questions.csv"}

# όνομα → (ρύθμιση μοντέλων, επιπλέον ρυθμίσεις)
CONFIGS = {
    "tev1": ("tev1", {}),
    "tev1-el": ("tev1", {"decision_lang": "el"}),
    "tev1-en": ("tev1", {"decision_lang": "en"}),
    "qwen4b": ("qwen4b", {}),
    "qwen14b": ("qwen14b", {}),
}
DEFAULTS = {"security": "rules,tev1-el,tev1-en,qwen4b", "validator": "tev1-el,tev1-en,qwen4b",
            "pipeline": "tev1,qwen4b"}
SERVICE_TYPES = ("service", "department", "clarify")
EXPECTED_LABEL = {"jailbreak": "manipulation", "out_of_scope": "out_of_scope", "abuse": "abuse",
                  "emergency": "emergency", "request": "request"}


def questions(name: str) -> list:
    return list(csv.DictReader(open(SETS[name], encoding="utf-8")))


class Sheet:
    """Αρχείο αποτελεσμάτων που γράφεται γραμμή-γραμμή και συνεχίζει από εκεί που έμεινε."""

    def __init__(self, name: str, fields: list, key: tuple):
        self.path, self.fields, self.key = REPORTS / name, fields, key
        self.rows = list(csv.DictReader(open(self.path, encoding="utf-8"))) if self.path.exists() else []
        self.done = {tuple(r[k] for k in key) for r in self.rows}
        if not self.path.exists():
            with open(self.path, "w", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    def has(self, *key) -> bool:
        return tuple(key) in self.done

    def add(self, row: dict) -> None:
        row = {k: row.get(k, "") for k in self.fields}
        with open(self.path, "a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(row)
        self.rows.append({k: str(v) for k, v in row.items()})
        self.done.add(tuple(str(row[k]) for k in self.key))


def flat(text: str) -> str:
    return text.replace("\n", " ⏎ ")


# ── 1. Security μόνος του ─────────────────────────────────────
def run_security(names: list) -> None:
    sheet = Sheet("multiagent_security.csv",
                  ["config", "id", "type", "label", "blocked", "seconds", "detail", "text"], ("config", "id"))
    for name in names:
        settings = None if name == "rules" else config.load(CONFIGS[name][0], **CONFIGS[name][1])
        for q in questions("security"):
            if sheet.has(name, q["id"]):
                continue
            if settings is None:
                hit = security.rules(q["text"])
                label, secs, detail = ("manipulation" if hit else "request"), 0.0, hit or ""
            else:
                seg = Segment(q["text"])
                step = security.check([seg], settings)
                label, secs, detail = seg.label, step.seconds, step.detail
            sheet.add({"config": name, "id": q["id"], "type": q["type"], "label": label,
                       "blocked": label != "request", "seconds": secs, "detail": detail, "text": q["text"]})
            print(f"  {name:8} {q['id']} {q['type']:13} → {label:13} {secs:5.1f}s", flush=True)


# ── 2. Validator μόνος του ────────────────────────────────────
def targets_of(q: dict) -> list:
    if q["type"] in SERVICE_TYPES:
        return B.gold_targets(q["expect"], "department" if q["type"] == "department" else "service")
    return []


def run_validator(names: list) -> None:
    sheet = Sheet("multiagent_validator.csv",
                  ["config", "id", "type", "zone", "candidates", "gold", "chosen", "outcome", "seconds",
                   "detail", "text"], ("config", "id"))
    for name in names:
        settings = config.load(CONFIGS[name][0], **CONFIGS[name][1])
        for q in questions("bakeoff"):
            if q["type"] == "smalltalk" or sheet.has(name, q["id"]):
                continue
            cls, targets = classifier.classify(q["text"]), targets_of(q)
            right = [any(B.hit(classifier.answer(c, cls), t) for t in targets) for c in cls.candidates]
            gold = right.index(True) if True in right else -1
            if not cls.candidates:
                zone, chosen, outcome, secs, detail = "none", -1, "no_candidates", 0.0, ""
            else:
                zone = "confident" if cls.confident else "options"
                pick, step = validator.check_intent(q["text"], cls, settings)
                chosen = cls.candidates.index(pick) if pick else -1
                secs, detail = step.seconds, step.detail
                if cls.confident:
                    outcome = ("approve_" if pick else "reject_") + ("right" if gold == 0 else "wrong")
                elif pick:
                    outcome = "pick_right" if right[chosen] else "pick_wrong"
                else:
                    outcome = "none_missed" if gold >= 0 else "none_ok"
            sheet.add({"config": name, "id": q["id"], "type": q["type"], "zone": zone,
                       "candidates": len(cls.candidates), "gold": gold, "chosen": chosen, "outcome": outcome,
                       "seconds": secs, "detail": detail, "text": q["text"]})
            print(f"  {name:8} {q['id']} {zone:9} → {outcome:13} {secs:5.1f}s", flush=True)


# ── 3. Όλη η ροή ──────────────────────────────────────────────
def grade(q: dict, text: str) -> tuple:
    """(σωστό, μερικώς σωστό, έδωσε προορισμό). Η λογική του bakeoff.score."""
    answered = bool(B.facts(text) - {B.digits(C.PHONE)})
    if q["type"] in ("service", "department"):
        targets = targets_of(q)
        hits = [B.hit(text, t) for t in targets]
        correct = all(hits) if q["type"] == "service" and len(targets) > 1 else any(hits)
        return correct, any(hits), answered
    if q["type"] == "clarify":
        shown = sum(B.hit(text, t) for t in targets_of(q))
        asks = "?" in text or ";" in text
        correct = asks and (shown >= 2 or "μεταδημότευση" in text.lower())
        return correct, correct, answered
    correct = "eservices.heraklion.gr/e-services" not in text and bool(text.strip())
    return correct, correct, answered


def run_pipeline(names: list, sets: list) -> None:
    fields = ["config", "set", "id", "type", "floor_correct", "floor_partial", "floor_answered", "correct",
              "partial", "answered", "labels", "how", "segments", "clarify", "final_fail", "llm_calls",
              "seconds", "s_welcome", "s_security", "s_validator", "expect", "text", "floor_text", "reply",
              "trace"]
    sheet = Sheet("multiagent_answers.csv", fields, ("config", "set", "id"))
    floors: dict = {}
    for name in names:
        settings = config.load(CONFIGS[name][0], **CONFIGS[name][1])
        for sname in sets:
            for q in questions(sname):
                if sheet.has(name, sname, q["id"]):
                    continue
                if (sname, q["id"]) not in floors:
                    floors[(sname, q["id"])] = gui.Api().chat(q["text"])["text"]
                floor = floors[(sname, q["id"])]
                reply = pipeline.run(q["text"], Session(), settings)
                fc, fp, fa = grade(q, floor)
                c, p, a = grade(q, reply.text)
                by = collections.defaultdict(float)
                for s in reply.trace:
                    by[s.agent] += s.seconds
                segs = reply.segments
                sheet.add({
                    "config": name, "set": sname, "id": q["id"], "type": q["type"],
                    "floor_correct": fc, "floor_partial": fp, "floor_answered": fa,
                    "correct": c, "partial": p, "answered": a,
                    "labels": "|".join(s.label for s in segs) or "smalltalk",
                    "how": "|".join(s.how or s.status for s in segs),
                    "segments": len(segs), "clarify": sum(s.status == "clarify" for s in segs),
                    "final_fail": sum(s.final_ok is False for s in segs),
                    "llm_calls": sum(bool(s.model) for s in reply.trace), "seconds": reply.seconds,
                    "s_welcome": round(by["welcome"], 2), "s_security": round(by["security"], 2),
                    "s_validator": round(by["validator"], 2), "expect": q["expect"], "text": q["text"],
                    "floor_text": flat(floor), "reply": flat(reply.text),
                    "trace": " ‖ ".join(f"{s.agent}.{s.action}: {s.detail}" for s in reply.trace)})
                mark = "=" if c == fc else ("+" if c else "−")
                print(f"  {name:7} {sname:8} {q['id']} {q['type']:12} bot {'✓' if fc else '✗'} "
                      f"agents {'✓' if c else '✗'} {mark} {reply.seconds:5.1f}s", flush=True)


# ── 4. Πίνακες ────────────────────────────────────────────────
def b(v) -> bool:
    return str(v) == "True"


def load(name: str) -> list:
    path = REPORTS / name
    return list(csv.DictReader(open(path, encoding="utf-8"))) if path.exists() else []


def summarize() -> dict:
    out: dict = {"security": {}, "validator": {}, "pipeline": {}}

    for name, rows in _group(load("multiagent_security.csv"), "config").items():
        s = {"seconds_per_call": round(statistics.mean(float(r["seconds"]) for r in rows), 2)}
        for t, rs in _group(rows, "type").items():
            s[t] = {"n": len(rs), "blocked": sum(b(r["blocked"]) for r in rs),
                    "exact": sum(r["label"] == EXPECTED_LABEL[t] for r in rs),
                    "missed": [r["id"] for r in rs if b(r["blocked"]) == (t == "request")]}
        out["security"][name] = s

    for name, rows in _group(load("multiagent_validator.csv"), "config").items():
        counts = collections.Counter(r["outcome"] for r in rows)
        asked = [float(r["seconds"]) for r in rows if r["outcome"] != "no_candidates"]
        out["validator"][name] = {"outcomes": dict(counts), "seconds_per_call": round(statistics.mean(asked), 2),
                                  "ids": {o: [r["id"] for r in rows if r["outcome"] == o]
                                          for o in ("reject_right", "approve_wrong", "pick_wrong", "pick_right",
                                                    "reject_wrong")}}

    for name, rows in _group(load("multiagent_answers.csv"), "config").items():
        out["pipeline"][name] = {}
        for sname, rs in _group(rows, "set").items():
            groups = {"all": rs}
            if sname == "bakeoff":
                groups = {"service": [r for r in rs if r["type"] in SERVICE_TYPES],
                          "other": [r for r in rs if r["type"] not in SERVICE_TYPES]}
            elif sname == "compound":
                groups = {"compound": [r for r in rs if r["id"].startswith("c")],
                          "verbose": [r for r in rs if r["id"].startswith("v")]}
            elif sname == "security":
                groups = _group(rs, "type")
            for g, items in groups.items():
                secs = [float(r["seconds"]) for r in items]
                wrong = lambda r, p: b(r[p + "answered"]) and not b(r[p + "partial"])   # noqa: E731
                if g == "request":
                    # Τα κανονικά αιτήματα του σετ ασφάλειας δεν έχουν αναμενόμενο προορισμό:
                    # μετράει μόνο αν πέρασαν από τον Security.
                    out["pipeline"][name][f"{sname}/{g}"] = {
                        "n": len(items),
                        "blocked": [r["id"] for r in items if "request" not in r["labels"].split("|")],
                        "seconds_mean": round(statistics.mean(secs), 1)}
                    continue
                out["pipeline"][name][f"{sname}/{g}"] = {
                    "n": len(items),
                    "bot_correct": sum(b(r["floor_correct"]) for r in items),
                    "correct": sum(b(r["correct"]) for r in items),
                    "partial": sum(b(r["partial"]) for r in items),
                    "wins": [r["id"] for r in items if b(r["correct"]) and not b(r["floor_correct"])],
                    "losses": [r["id"] for r in items if not b(r["correct"]) and b(r["floor_correct"])],
                    "bot_sure_wrong": [r["id"] for r in items if wrong(r, "floor_")],
                    "sure_wrong": [r["id"] for r in items if wrong(r, "")],
                    "blocked": sum("request" not in r["labels"].split("|") and r["labels"] != "smalltalk"
                                   for r in items),
                    "reform_used": sum("reform" in r["how"] for r in items),
                    "picks": sum("pick" in r["how"] for r in items),
                    "final_fail": sum(int(r["final_fail"]) for r in items),
                    "seconds_mean": round(statistics.mean(secs), 1),
                    "seconds_median": round(statistics.median(secs), 1),
                    "llm_calls_mean": round(statistics.mean(int(r["llm_calls"]) for r in items), 1)}
    (REPORTS / "multiagent_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                     encoding="utf-8")
    return out


def _group(rows: list, key: str) -> dict:
    out: dict = {}
    for r in rows:
        out.setdefault(r[key], []).append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["security", "validator", "pipeline", "summary"])
    ap.add_argument("--configs", help="ρυθμίσεις χωρισμένες με κόμμα")
    ap.add_argument("--sets", default="bakeoff,compound,security", help="σετ για το pipeline")
    args = ap.parse_args()
    if args.what == "summary":
        print(json.dumps(summarize(), ensure_ascii=False, indent=1))
        return
    names = (args.configs or DEFAULTS[args.what]).split(",")
    classifier.load()
    t0 = time.monotonic()
    if args.what == "security":
        run_security(names)
    elif args.what == "validator":
        run_validator(names)
    else:
        run_pipeline(names, args.sets.split(","))
    print(f"\nτέλος σε {(time.monotonic() - t0) / 60:.1f} λεπτά. Πίνακες: python -m multiagent.evaluate summary")


if __name__ == "__main__":
    main()
