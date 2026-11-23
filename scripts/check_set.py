"""
Έλεγχος ενός σετ προτάσεων για έναν agent, πριν γίνει commit και πριν τρέξει μοντέλο.

    venv/bin/python scripts/check_set.py multiagent/sets/security/maria_dev.csv
    venv/bin/python scripts/check_set.py multiagent/sets/validator/nikos_dev.csv --agent validator

Ο agent βγαίνει από τον φάκελο του αρχείου (security, validator, split, reform) ή δίνεται
με --agent. Οδηγίες για τα σετ: multiagent/sets/README.md. Ελέγχει:
  - τις στήλες, τα μοναδικά id, τα κενά κείμενα και τα επιτρεπτά type
  - ότι κάθε intent του expect υπάρχει στο μοντέλο, και κάθε τμήμα στο mappings/departments.csv
  - ότι καμία πρόταση δεν μοιάζει πολύ (από 0,80 και πάνω) με πρόταση των δεδομένων
    εκπαίδευσης ή άλλου σετ, ούτε μετά τη μεταγραφή των greeklish σε ελληνικά
Τυπώνει και πόσες προτάσεις έχει από κάθε είδος και γλώσσα. Τελειώνει με κωδικό 1 αν
βρει πρόβλημα, ώστε να διορθωθεί πριν από το commit.
"""
from __future__ import annotations

import argparse
import collections
import csv
import difflib
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "scripts", ROOT / "app"):
    sys.path.insert(0, str(_p))

from prepare_dataset import EVAL_FILES, notebook_label, text_key     # noqa: E402

SIMILAR = 0.80
TRAINING = ROOT / "datasets" / "intent_dataset_v6_clean.csv"
AGENTS = {
    # επιτρεπτά type και η στήλη που χρειάζεται κάθε agent
    "security": ({"request", "service", "out_of_scope", "jailbreak", "manipulation", "abuse", "emergency"}, None),
    "validator": ({"service", "department", "out_of_scope"}, "expect"),
    "split": ({"service"}, "requests"),
    "reform": ({"service", "department"}, "expect"),
}


def intents() -> set:
    import joblib
    model = ROOT / os.environ.get("BOT_MODEL", "models/intent-model-233")
    return {str(c) for c in joblib.load(model / "label_encoder.joblib").classes_}


def departments() -> set:
    with open(ROOT / "mappings" / "departments.csv", encoding="utf-8") as f:
        return {r["code"] for r in csv.DictReader(f)}


def corpus(skip: Path) -> dict:
    """κανονικοποιημένο κείμενο → (κείμενο, αρχείο), από την εκπαίδευση και όλα τα άλλα σετ."""
    out = {}
    for path in [TRAINING] + [p for p in EVAL_FILES if p.exists() and p.resolve() != skip.resolve()]:
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                for col in ("text", "m1", "m2"):
                    if r.get(col):
                        out.setdefault(text_key(r[col]), (r[col], path.name))
    return out


def language(text: str, greeklish) -> str:
    letters = re.findall(r"[^\W\d_]", text)
    latin = sum(c.isascii() for c in letters)
    if not letters or latin < len(letters) / 2:
        return "ελληνικά"
    return "greeklish" if greeklish.to_greek(text) else "αγγλικά ή άλλα λατινικά"


def check(path: Path, agent: str) -> int:
    import greeklish

    types, needs = AGENTS[agent]
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = set(rows[0].keys()) if rows else set()
    problems = []
    for col in ("id", "text", "type"):
        if col not in cols:
            problems.append(f"λείπει η στήλη «{col}»")
    if agent == "split" and not {"requests", "expect"} & cols:
        problems.append("λείπει η στήλη «requests» (πόσα αιτήματα έχει κάθε μήνυμα)")
    elif needs and agent != "split" and needs not in cols:
        problems.append(f"λείπει η στήλη «{needs}»")
    if problems:
        return report(path, agent, rows, problems, {}, {})

    known, depts = intents(), departments()
    ids = collections.Counter(r["id"] for r in rows)
    seen, said = {}, set()
    for r in rows:
        rid, text, kind = r["id"], (r.get("text") or "").strip(), (r.get("type") or "").strip()
        if (not rid or ids[rid] > 1) and rid not in said:
            said.add(rid)
            problems.append(f"{rid or '(χωρίς id)'}: το id πρέπει να υπάρχει και να είναι μοναδικό")
        if not text:
            problems.append(f"{rid}: κενό κείμενο")
            continue
        if kind not in types:
            problems.append(f"{rid}: type «{kind}»· γίνεται {', '.join(sorted(types))}")
        if text_key(text) in seen:
            problems.append(f"{rid}: ίδιο κείμενο με το {seen[text_key(text)]}")
        seen.setdefault(text_key(text), rid)
        expect = (r.get("expect") or "").strip()
        if kind in ("service", "department") and agent in ("validator", "reform") and not expect:
            problems.append(f"{rid}: χρειάζεται expect (η σωστή υπηρεσία)")
        for part in filter(None, expect.split("|")):
            for name in filter(None, part.split("/")):
                pool = depts if kind == "department" else known
                if name not in pool:
                    near = notebook_label(name)
                    hint = f" (μήπως «{near}»;)" if near in pool and near != name else ""
                    problems.append(f"{rid}: «{name}» δεν είναι {'τμήμα' if kind == 'department' else 'intent'}{hint}")
        if agent == "split":
            req = (r.get("requests") or "").strip()
            if req and not (req.isdigit() and int(req) >= 1):
                problems.append(f"{rid}: requests «{req}»· γράφεται αριθμός αιτημάτων (1, 2, 3)")
            if not req and not expect:
                problems.append(f"{rid}: χρειάζεται requests (πόσα αιτήματα έχει το μήνυμα)")

    print(f"Σύγκριση με {TRAINING.name} και τα άλλα σετ (μπορεί να πάρει μισό λεπτό)…", flush=True)
    others = corpus(path)
    keys = list(others)
    for r in rows:
        text = (r.get("text") or "").strip()
        variants = {text_key(text)}
        converted = greeklish.to_greek(text)
        if converted:
            variants.add(text_key(converted))
        for v in variants:
            match = difflib.get_close_matches(v, keys, n=1, cutoff=SIMILAR)
            if match:
                score = difflib.SequenceMatcher(None, v, match[0]).ratio()
                original, source = others[match[0]]
                problems.append(f"{r['id']}: μοιάζει ({score:.2f}) με «{original[:70]}» ({source}) · γράψτε το αλλιώς")
                break
    kinds = collections.Counter((r.get("type") or "").strip() for r in rows)
    langs = collections.Counter(language(r.get("text") or "", greeklish) for r in rows)
    return report(path, agent, rows, problems, kinds, langs)


def report(path: Path, agent: str, rows: list, problems: list, kinds: dict, langs: dict) -> int:
    print(f"\n{path} · agent: {agent} · {len(rows)} προτάσεις")
    if kinds:
        print("   είδη:    " + " · ".join(f"{k} {n}" for k, n in kinds.most_common()))
    if langs:
        print("   γλώσσες: " + " · ".join(f"{k} {n}" for k, n in langs.most_common()))
    if not problems:
        print("✓ Κανένα πρόβλημα. Κάντε commit το αρχείο πριν τρέξει μοντέλο.")
        return 0
    print(f"✗ {len(problems)} προβλήματα:")
    for p in problems:
        print(f"   {p}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Έλεγχος ενός σετ προτάσεων για έναν agent")
    ap.add_argument("path", type=Path)
    ap.add_argument("--agent", choices=list(AGENTS), help="αν λείπει, ο φάκελος του αρχείου")
    args = ap.parse_args()
    agent = args.agent or args.path.resolve().parent.name
    if agent not in AGENTS:
        ap.error(f"δεν ξέρω για ποιον agent είναι («{agent}»)· δώστε --agent {'|'.join(AGENTS)}")
    return check(args.path, agent)


if __name__ == "__main__":
    sys.exit(main())
