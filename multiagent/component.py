"""
Ένα κομμάτι του multi-agent μόνο του, σε ένα μήνυμα: τι βγάζει, πριν από το υπόλοιπο pipeline.

    venv/bin/python -m multiagent.component component=security 'once="πήρε φωτιά το διπλανό σπίτι"'
    venv/bin/python -m multiagent.component component=classifier 'once="θέλω βεβαίωση κατοικίας"'
    venv/bin/python -m multiagent.component component=validator 'once="…"'   πρώτα το BERT, μετά ο Validator
    venv/bin/python -m multiagent.component component=welcome 'once="…"'     χωρισμός σε αιτήματα
    venv/bin/python -m multiagent.component component=pipeline 'once="…"'    όλη η ροή, όπως το chat

Δέχεται τις ίδιες αλλαγές με το multiagent.chat, π.χ. model@security.model=qwen3_4b.
Για να δεις τι κάνει το υπόλοιπο pipeline όταν ένα κομμάτι βγάλει κάτι συγκεκριμένο,
τρέξε όλη τη ροή με το κομμάτι «δοσμένο», π.χ. security=fixed security.label=emergency.
"""
from __future__ import annotations

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from multiagent import ROOT
from multiagent.cli import converse
from multiagent.config import choice
from multiagent.pipeline import _describe
from multiagent.state import Segment

PARTS = ("welcome", "security", "classifier", "validator", "pipeline")


def _step(step) -> None:
    if step is None:
        return
    tail = f"  [{step.model} {step.seconds:.1f}s]" if step.model else ""
    print(f"   · {step.agent:<10} {step.action:<9} {step.detail}{tail}")


def _candidates(title: str, cands: list) -> None:
    print(f"{title}:" if cands else f"{title}: καμία")
    for i, c in enumerate(cands, 1):
        print(f"   {i}. {c.title}  ({c.intent}, {c.score:.2f})")


@hydra.main(config_path=str(ROOT / "configs"), config_name="multiagent", version_base="1.3")
def main(cfg: DictConfig) -> None:
    choice("component", cfg.component, PARTS)
    if not cfg.once:
        raise SystemExit("Χρειάζεται μήνυμα, π.χ. 'once=\"θέλω βεβαίωση κατοικίας\"'")
    pipe, text = instantiate(cfg.pipeline), str(cfg.once)
    if cfg.component == "pipeline":
        converse(pipe, text)
        return
    pipe.classifier.load()
    view = pipe.classifier.greek_view(text) if pipe.greeklish == "greek" else text
    print(f"\nΜήνυμα: {text}" + (f"\nΣε ελληνικά: {view}" if view != text else "") + "\n")

    if cfg.component == "welcome":
        parts, alt, step = pipe.welcome.split(text)
        _step(step)
        print(f"Αιτήματα ({len(parts)}):")
        for p in parts:
            print(f"   · {p}")
        if alt:
            print(f"Απόσπασμα για δεύτερη ευκαιρία στο BERT: {alt}")

    elif cfg.component == "security":
        rule = pipe.security.rule(text) or (view != text and pipe.security.rule(view))
        if rule:
            print(f"Κανόνας: {rule} → το μήνυμα κόβεται πριν από το μοντέλο")
            return
        seg = Segment(text, view=view)
        _step(pipe.security.check([seg]))
        print(f"Ετικέτα: {seg.label} · κόβεται πριν από το BERT: {'ναι' if pipe.security.blocks(seg.label) else 'όχι'}")

    elif cfg.component == "classifier":
        cls = pipe.classifier.classify(text)
        print(_describe(cls))
        for intent, p in cls.intents[:5]:
            print(f"   {p:.2f}  {intent}")
        _candidates("Υποψήφιοι του bot", cls.candidates)

    else:                                                   # validator
        cls = pipe.classifier.classify(text)
        print(f"BERT: {_describe(cls)}")
        _candidates("Υποψήφιοι για τον Validator", pipe.validator.pool(cls))
        chosen, step = pipe.validator.check_intent(view, cls)
        _step(step)
        print(f"Εγκρίθηκε: {chosen.title if chosen else 'καμία'}")


if __name__ == "__main__":
    main()
