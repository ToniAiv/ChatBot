"""
Συζήτηση με το multi-agent σύστημα, με ρυθμίσεις από τις μεταβλητές MA_* (ο παλιός τρόπος).

    venv/bin/python -m multiagent.cli                     συζήτηση
    venv/bin/python -m multiagent.cli --once "ερώτηση"    ένα μήνυμα
    venv/bin/python -m multiagent.cli --profile qwen4b    άλλη ρύθμιση μοντέλων

Ο κύριος τρόπος είναι πια το python -m multiagent.chat, με ρυθμίσεις από τα
configs/ (Hydra). Η συζήτηση είναι ίδια και στους δύο.

Μέσα στη συζήτηση: /νεα ξεχνά το αίτημα που περιμένει διευκρίνιση, /τελος κλείνει.
Θέλει το Ollama ανοιχτό.
"""
from __future__ import annotations

import argparse

from multiagent import config
from multiagent.pipeline import Pipeline
from multiagent.state import Session


def show(reply) -> None:
    for s in reply.trace:
        tail = f"  [{s.model} {s.seconds:.1f}s]" if s.model else (f"  [{s.seconds:.2f}s]" if s.seconds else "")
        print(f"   · {s.agent:<10} {s.action:<9} {s.detail}{tail}")
    print(f"\n{reply.text}")
    for i, o in enumerate(reply.options, 1):
        print(f"   {i}. {o['title']}")
    print(f"\n   ({reply.seconds:.1f}s)\n")


def converse(pipe: Pipeline, once: str | None = None) -> None:
    """Ένα μήνυμα (once), ή συζήτηση μέχρι το /τελος."""
    pipe.classifier.load()
    print("\n" + pipe.describe() + "\n")
    session = Session()
    if once:
        show(pipe.run(once, session))
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
            show(pipe.run(message, session))


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-agent βοηθός του Δήμου Ηρακλείου (μεταβλητές MA_*)")
    ap.add_argument("--profile", choices=list(config.PROFILES), help="ρύθμιση μοντέλων (προεπιλογή: tev1)")
    ap.add_argument("--once", metavar="ΜΗΝΥΜΑ", help="απάντησε σε ένα μήνυμα και κλείσε")
    args = ap.parse_args()
    converse(Pipeline.from_settings(config.load(args.profile)), args.once)


if __name__ == "__main__":
    main()
