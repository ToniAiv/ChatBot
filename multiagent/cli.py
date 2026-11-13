"""
Συζήτηση με το multi-agent σύστημα στο τερματικό, με το ίχνος κάθε μηνύματος.

    venv/bin/python -m multiagent.cli                     συζήτηση
    venv/bin/python -m multiagent.cli --once "ερώτηση"    ένα μήνυμα
    venv/bin/python -m multiagent.cli --profile qwen4b    άλλη ρύθμιση μοντέλων

Μέσα στη συζήτηση: /νεα ξεχνά το αίτημα που περιμένει διευκρίνιση, /τελος κλείνει.
Θέλει το Ollama ανοιχτό.
"""
from __future__ import annotations

import argparse

from multiagent import classifier, config, pipeline
from multiagent.state import Session


def show(reply) -> None:
    for s in reply.trace:
        tail = f"  [{s.model} {s.seconds:.1f}s]" if s.model else (f"  [{s.seconds:.2f}s]" if s.seconds else "")
        print(f"   · {s.agent:<10} {s.action:<9} {s.detail}{tail}")
    print(f"\n{reply.text}")
    for i, o in enumerate(reply.options, 1):
        print(f"   {i}. {o['title']}")
    print(f"\n   ({reply.seconds:.1f}s)\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-agent βοηθός του Δήμου Ηρακλείου")
    ap.add_argument("--profile", choices=list(config.PROFILES), help="ρύθμιση μοντέλων (προεπιλογή: tev1)")
    ap.add_argument("--once", metavar="ΜΗΝΥΜΑ", help="απάντησε σε ένα μήνυμα και κλείσε")
    args = ap.parse_args()

    settings = config.load(args.profile)
    classifier.load()
    print("\nΜοντέλα: " + ", ".join(f"{r}={settings.model(r)}" for r in config.ROLES)
          + f"\n«Εκτός αρμοδιότητας»: {settings.scope} BERT · ερωτήσεις decision model: {settings.decision_lang}"
          + f" · «επείγον»: {settings.emergency} · χωρισμός: {settings.split_langs}\n")
    session = Session()
    if args.once:
        show(pipeline.run(args.once, session, settings))
        return
    while True:
        try:
            message = input("Πολίτης> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if message in ("/τελος", "/τέλος", "/q"):
            return
        if message in ("/νεα", "/νέα"):
            session.reset()
            print("   (νέα συζήτηση)\n")
            continue
        if message:
            show(pipeline.run(message, session, settings))


if __name__ == "__main__":
    main()
