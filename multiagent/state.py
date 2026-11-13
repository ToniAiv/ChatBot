"""Τα δεδομένα που περνούν από agent σε agent μέσα σε ένα μήνυμα."""
from __future__ import annotations

from dataclasses import dataclass, field

import context


@dataclass
class Step:
    """Μία γραμμή του ίχνους: ποιος agent έκανε τι, με ποιο μοντέλο, σε πόσο χρόνο."""
    agent: str
    action: str
    detail: str = ""
    seconds: float = 0.0
    model: str = ""


@dataclass
class Candidate:
    """Ένας πιθανός προορισμός: σελίδα υπηρεσίας ή τμήμα."""
    intent: str
    score: float
    kind: str                       # link | phone | dept (ρητή ερώτηση για τμήμα)
    title: str
    url: str = ""
    description: str = ""
    dept: dict | None = None
    topic: str = ""                 # σύντομη περιγραφή για τον Validator (validator_text=clean)


@dataclass
class Classification:
    """Η έξοδος του BERT για ένα αίτημα, με τις αποφάσεις του σημερινού bot."""
    current: str                    # το αίτημα στα ελληνικά, όπως το είδε το BERT
    lang: str
    intents: list = field(default_factory=list)       # top-5 (intent, πιθανότητα)
    confidence: float = 0.0         # βεβαιότητα προορισμού
    confident: bool = False
    candidates: list = field(default_factory=list)    # 1 αν είναι σίγουρο, αλλιώς οι επιλογές του bot
    top: list = field(default_factory=list)           # οι 5 πρώτες του BERT, χωρίς διπλό προορισμό
    family: str = ""                # τίτλος οικογένειας, αν οι επιλογές είναι οικογένεια
    aspect: str | None = None       # δικαιολογητικά, επικοινωνία κ.λπ. (aspects.py)
    error: str = ""


@dataclass
class Segment:
    """Ένα αίτημα του μηνύματος, με τις λέξεις του πολίτη."""
    text: str
    view: str = ""                  # ό,τι βλέπουν Security και Validator (greeklish=greek: σε ελληνικά)
    alt: str = ""                   # απόσπασμα του Welcome από φλύαρο μήνυμα
    label: str = "request"          # Security: request | out_of_scope | manipulation | abuse | emergency
    cls: Classification | None = None
    chosen: Candidate | None = None
    how: str = ""                   # confident | pick | switch (άλλη από την 1η του σίγουρου BERT) | reform
    reformed: str = ""              # το κείμενο του case 1, αν έγινε δεκτό
    status: str = ""                # answered | clarify | blocked | out_of_scope | emergency | error
    answer: str = ""
    options: list = field(default_factory=list)
    final_ok: bool | None = None    # ο τελικός Validator


@dataclass
class Reply:
    text: str
    options: list
    lang: str
    segments: list
    trace: list
    seconds: float

    def to_dict(self) -> dict:
        """Η μορφή που επιστρέφει και το gui.Api.chat."""
        return {"text": self.text, "options": self.options, "lang": self.lang}


class Session:
    """Η μνήμη μιας συζήτησης: το αίτημα που περιμένει διευκρίνιση (case 2)."""

    def __init__(self):
        self.pending = context.Pending()

    def reset(self) -> None:
        self.pending.clear()
