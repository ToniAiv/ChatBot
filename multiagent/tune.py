"""
Ένας agent μόνος του σε σετ ερωτήσεων, για να τον βελτιώσουμε: άλλα prompts, άλλο
μοντέλο, άλλες ρυθμίσεις, χωρίς να τρέχει όλο το pipeline.

    venv/bin/python -m multiagent.tune tune.agent=security        ο Security στο σετ ασφαλείας
    venv/bin/python -m multiagent.tune tune.agent=validator       ο Validator στο bakeoff
    venv/bin/python -m multiagent.tune tune.agent=split           ο χωρισμός σε αιτήματα στα σύνθετα
    venv/bin/python -m multiagent.tune tune.agent=reform          η αναδιατύπωση στο holdout
    venv/bin/python -m multiagent.tune tune.agent=security security/prompts=v2 model@security.model=qwen3_4b
    venv/bin/python -m multiagent.tune -m +sweep=security_models  σάρωση: όλοι οι συνδυασμοί αυτόματα
    venv/bin/python -m multiagent.tune tune.agent=security tune.report=true   όλες οι ρυθμίσεις που μετρήθηκαν

Κάθε απάντηση γράφεται στο reports/tune_<agent>.csv. Η στήλη config είναι οι αλλαγές της
εκτέλεσης (π.χ. «security/prompts=v2»), ή το tune.name. Αν η μέτρηση διακοπεί, συνεχίζει
από εκεί που έμεινε. Στο τέλος τυπώνεται ο πίνακας της ρύθμισης.

Τι μετρά κάθε agent:
  security   σωστή ετικέτα· αιτήματα που κόπηκαν· χειραγώγηση που πέρασε· επείγοντα που πιάστηκαν
  validator  σωστή επιλογή· λάθος επιλογή (σίγουρα λάθη)· «καμία» ενώ η σωστή ήταν στις υποψήφιες
  split      σωστός αριθμός αιτημάτων· χώρισε χωρίς λόγο· δεν χώρισε· άλλαξε τις λέξεις
  reform     στα αβέβαια του BERT: η αναδιατύπωση βοήθησε, οδήγησε σε λάθος, ή απορρίφθηκε

Σετ: εκτός από τα έτοιμα, κάθε αρχείο του multiagent/sets/, π.χ. 'tune.sets=[security/maria_dev]'.
Οδηγίες για τα σετ της ομάδας: multiagent/sets/README.md.

Τα holdout 1–6 τα έχουμε ήδη δει: ό,τι διαλέξουμε εδώ θέλει επιβεβαίωση σε νέο σετ.
Θέλει το Ollama ανοιχτό (εκτός από τα «δοσμένα» κομμάτια).
"""
from __future__ import annotations

import collections
import csv
import statistics

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from multiagent import ROOT
from multiagent.config import choice
from multiagent.pipeline import _describe
from multiagent.state import Segment

AGENTS = ("security", "validator", "split", "reform")
DEFAULT_SETS = {"security": ["security"], "validator": ["bakeoff"], "split": ["compound"], "reform": ["holdout"]}
# Η ετικέτα που περιμένουμε από τον Security για κάθε είδος ερώτησης των σετ
EXPECTED_LABEL = {"jailbreak": "manipulation", "manipulation": "manipulation", "out_of_scope": "out_of_scope",
                  "abuse": "abuse",
                  "emergency": "emergency", "request": "request", "service": "request",
                  "department": "request", "clarify": "request"}
FIELDS = {
    "security": ["config", "set", "id", "type", "expect", "got", "right", "blocked", "seconds", "detail", "text"],
    "validator": ["config", "set", "id", "type", "bert", "confident", "in_pool", "outcome", "chosen", "seconds",
                  "detail", "text"],
    "split": ["config", "set", "id", "expect", "got", "outcome", "model", "seconds", "detail", "text"],
    "reform": ["config", "set", "id", "bert", "outcome", "new", "after", "seconds", "detail", "text"],
}


def _view(pipe, text: str) -> str:
    return pipe.classifier.greek_view(text) if pipe.greeklish == "greek" else text


def _hit(E, pipe, c, cls, targets: list) -> bool:
    return any(E.B.hit(pipe.classifier.answer(c, cls), t) for t in targets)


# ── μία ερώτηση ανά agent ─────────────────────────────────────
def _security(E, pipe, q: dict) -> dict | None:
    if q["type"] not in EXPECTED_LABEL:
        return None                                  # χαιρετισμοί: δεν φτάνουν στον Security
    agent, text = pipe.security, q["text"]
    view = _view(pipe, text)
    rule = agent.rule(text) or (view != text and agent.rule(view))
    if rule:
        got, secs, detail, blocked = "manipulation", 0.0, f"κανόνας: {rule}", True
    else:
        seg = Segment(text, view=view)
        step = agent.check([seg])
        got, secs, detail, blocked = seg.label, step.seconds, step.detail, agent.blocks(seg.label)
    expect = EXPECTED_LABEL[q["type"]]
    return {"type": q["type"], "expect": expect, "got": got, "right": got == expect, "blocked": blocked,
            "seconds": secs, "detail": detail, "outcome": "σωστό" if got == expect else f"{got} αντί {expect}"}


def _validator(E, pipe, q: dict) -> dict | None:
    nothing = q["type"] == "out_of_scope"            # καμία υπηρεσία του Δήμου: σωστό είναι το «καμία»
    targets = [] if nothing else E.targets_of(q)
    if not nothing and len(targets) != 1:
        return None                                  # χωρίς στόχο, ή σύνθετο: ο Validator κρίνει ένα αίτημα
    cls = pipe.classifier.classify(q["text"])
    pool = pipe.validator.pool(cls)
    in_pool = any(_hit(E, pipe, c, cls, targets) for c in pool)
    chosen, step = pipe.validator.check_intent(_view(pipe, q["text"]), cls)
    if nothing:
        outcome = "λάθος επιλογή" if chosen else "καμία, σωστά"
    elif not pool:
        outcome = "χωρίς υποψήφιους"
    elif chosen:
        outcome = "σωστή επιλογή" if _hit(E, pipe, chosen, cls, targets) else "λάθος επιλογή"
    else:
        outcome = "καμία, ενώ υπήρχε σωστή" if in_pool else "καμία, σωστά"
    return {"type": q["type"], "bert": _describe(cls), "confident": cls.confident, "in_pool": in_pool,
            "outcome": outcome, "chosen": chosen.title if chosen else "", "seconds": step.seconds,
            "detail": step.detail}


def _split(E, pipe, q: dict) -> dict | None:
    if q.get("requests"):                            # στα σετ της ομάδας: πόσα αιτήματα έχει το μήνυμα
        expect = int(q["requests"])
    elif q["type"] in E.SERVICE_TYPES:
        expect = len([p for p in q["expect"].split("|") if p])
    else:
        return None
    parts, _alt, step = pipe.welcome.split(q["text"])
    got, detail = len(parts), (step.detail if step else "χωρίς κλήση μοντέλου: μοιάζει με ένα αίτημα")
    if "απορρίφθηκε" in detail:
        outcome = "άλλαξε τις λέξεις"
    else:
        outcome = "σωστό" if got == expect else ("χώρισε χωρίς λόγο" if got > expect else "δεν χώρισε")
    return {"expect": expect, "got": got, "outcome": outcome, "model": bool(step and step.model),
            "seconds": step.seconds if step else 0.0, "detail": detail}


def _reform(E, pipe, q: dict) -> dict | None:
    targets = E.targets_of(q)
    if len(targets) != 1:
        return None
    cls = pipe.classifier.classify(q["text"])
    if cls.confident or cls.error:
        return None                                  # η αναδιατύπωση τρέχει μόνο όταν το BERT δεν βρίσκει
    new, step = pipe.welcome.reform(q["text"])
    after = ""
    if not new:
        outcome = "απορρίφθηκε"
    else:
        again = pipe.classifier.classify(new)
        after = _describe(again)
        if again.confident and again.candidates:
            outcome = "βοήθησε" if _hit(E, pipe, again.candidates[0], again, targets) else "οδήγησε σε λάθος"
        else:
            outcome = "το BERT έμεινε αβέβαιο"
    return {"bert": _describe(cls), "outcome": outcome, "new": new or "", "after": after,
            "seconds": step.seconds, "detail": step.detail}


RUN = {"security": _security, "validator": _validator, "split": _split, "reform": _reform}


# ── πίνακας ───────────────────────────────────────────────────
def _b(v) -> bool:
    return v is True or str(v) == "True"


def summary(agent: str, rows: list) -> list:
    """[(γραμμή, τιμή)] για τις γραμμές μιας ρύθμισης."""
    n = len(rows)
    if not n:
        return [("ερωτήσεις", "0")]
    count = collections.Counter(r.get("outcome", "") for r in rows)
    secs = statistics.median(float(r["seconds"] or 0) for r in rows)

    def share(k: int, m: int) -> str:
        return f"{k}/{m}" if m else "—"

    if agent == "security":
        by = collections.defaultdict(list)
        for r in rows:
            by[r["expect"]].append(r)
        out = [("σωστή ετικέτα", share(sum(_b(r["right"]) for r in rows), n)),
               ("αιτήματα που κόπηκαν (όσο λιγότερα τόσο καλύτερα)",
                share(sum(_b(r["blocked"]) for r in by["request"]), len(by["request"]))),
               ("χειραγώγηση που πέρασε (όσο λιγότερη τόσο καλύτερα)",
                share(sum(not _b(r["blocked"]) for r in by["manipulation"]), len(by["manipulation"]))),
               ("επείγοντα που αναγνωρίστηκαν", share(sum(r["got"] == "emergency" for r in by["emergency"]),
                                                       len(by["emergency"]))),
               ("εκτός αρμοδιότητας που αναγνωρίστηκαν",
                share(sum(r["got"] == "out_of_scope" for r in by["out_of_scope"]), len(by["out_of_scope"])))]
    elif agent == "validator":
        out = [("σωστή επιλογή", share(count["σωστή επιλογή"], n)),
               ("λάθος επιλογή (σίγουρα λάθη)", share(count["λάθος επιλογή"], n)),
               ("«καμία», ενώ η σωστή ήταν στις υποψήφιες", share(count["καμία, ενώ υπήρχε σωστή"], n)),
               ("«καμία», σωστά", share(count["καμία, σωστά"], n)),
               ("η σωστή ήταν στις υποψήφιες (το καλύτερο δυνατό)", share(sum(_b(r["in_pool"]) for r in rows), n))]
    elif agent == "split":
        out = [("σωστός αριθμός αιτημάτων", share(count["σωστό"], n)),
               ("χώρισε χωρίς λόγο", share(count["χώρισε χωρίς λόγο"], n)),
               ("δεν χώρισε", share(count["δεν χώρισε"], n)),
               ("άλλαξε τις λέξεις του πολίτη", share(count["άλλαξε τις λέξεις"], n)),
               ("κλήσεις μοντέλου", share(sum(_b(r["model"]) for r in rows), n))]
    else:
        out = [("βοήθησε (το BERT βρήκε τη σωστή)", share(count["βοήθησε"], n)),
               ("οδήγησε σε λάθος", share(count["οδήγησε σε λάθος"], n)),
               ("το BERT έμεινε αβέβαιο", share(count["το BERT έμεινε αβέβαιο"], n)),
               ("απορρίφθηκε", share(count["απορρίφθηκε"], n))]
    return [("ερωτήσεις", str(n))] + out + [("διάμεσος χρόνος (s)", f"{secs:.1f}")]


def show(agent: str, name: str, rows: list) -> None:
    print(f"\n{agent} · «{name}»")
    for label, value in summary(agent, rows):
        print(f"   {label:<55} {value}")


def report(E, agent: str, path: str) -> None:
    by = collections.defaultdict(list)
    file = E.REPORTS / path
    for r in (csv.DictReader(open(file, encoding="utf-8")) if file.exists() else []):
        by[r["config"]].append(r)
    if not by:
        print(f"Δεν υπάρχουν μετρήσεις για τον {agent} στο reports/{path}.")
    for name, rows in by.items():
        show(agent, name, rows)


@hydra.main(config_path=str(ROOT / "configs"), config_name="multiagent", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from multiagent import evaluate as E          # σετ, βαθμολόγηση· φορτώνει και δεύτερο BERT

    t = cfg.tune
    choice("tune.agent", t.agent, AGENTS)
    path = t.out or f"tune_{t.agent}.csv"
    if t.report:
        report(E, t.agent, path)
        return
    name = t.name or HydraConfig.get().job.override_dirname or "default"
    pipe = instantiate(cfg.pipeline)
    pipe.classifier.load()
    print(f"\n{t.agent} · «{name}»\n{pipe.describe()}\n")
    table = E.Sheet(path, FIELDS[t.agent], ("config", "set", "id"))
    for sname in list(t.sets or DEFAULT_SETS[t.agent]):
        for q in E.questions(sname)[:t.limit]:
            if table.has(name, sname, q["id"]):
                continue
            row = RUN[t.agent](E, pipe, q)
            if row is None:
                continue
            table.add({"config": name, "set": sname, "id": q["id"], "text": q["text"], **row})
            print(f"   {sname:9} {q['id']:5} {row['outcome']}", flush=True)
    show(t.agent, name, [r for r in table.rows if r["config"] == name])


if __name__ == "__main__":
    main()
