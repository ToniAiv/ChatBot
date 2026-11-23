"""
Security agent: τι είδους μήνυμα έστειλε ο πολίτης.

Δύο στρώσεις:
  1. Κανόνες στον κώδικα, σε ολόκληρο το μήνυμα (rules). Πιάνουν τα προφανή
     και δεν ξεγελιούνται.
  2. Ένα μοντέλο, για κάθε αίτημα χωριστά (check): decision model ή LLM,
     ανάλογα με τη ρύθμιση.

Χειραγώγηση και προσβολές κόβονται πάντα πριν από τον ταξινομητή.
Το «εκτός αρμοδιότητας» εξαρτάται από τη ρύθμιση scope (config.py): με
"before" κόβεται κι αυτό, με "after" το αίτημα περνά και η ετικέτα μετρά μόνο
αν ούτε το BERT ούτε ο Validator βρουν υπηρεσία (pipeline.py).
Το «επείγον» εξαρτάται από τη ρύθμιση emergency: με "block" κόβεται, με
"answer" η απάντηση έχει το 112 και, αν βρεθεί, την υπηρεσία του Δήμου.
Αν το μοντέλο δεν απαντήσει, το αίτημα περνά: οι απαντήσεις είναι έτοιμα
κείμενα, άρα ένα μήνυμα που ξέφυγε δεν μπορεί να βάλει λόγια στο στόμα του bot.

Ο agent είναι η κλάση Security· ρυθμίσεις στο configs/security/default.yaml. Οι ερωτήσεις
του (prompts) έρχονται από το configs/security/prompts/ και συμπληρώνονται την ώρα της κλήσης.
"""
from __future__ import annotations

import re

import connector as C

from multiagent import llm
from multiagent.config import Settings, choice, plain, prompts_of
from multiagent.state import Step

LABELS = ("request", "out_of_scope", "manipulation", "abuse", "emergency")
ALWAYS_BLOCKED = ("manipulation", "abuse")

# ── 1. Κανόνες ────────────────────────────────────────────────
# Λίγοι και στενοί: ό,τι πιάνουν είναι σχεδόν σίγουρα επίθεση. Το «ποιες
# είναι οι οδηγίες για την εγγραφή;» και το «αγνόησαν το αίτημά μου» περνούν.
_RAW = [
    ("κώδικας σελίδας", re.compile(r"<\s*/?\s*(script|iframe|img|svg|object|style)\b|javascript:", re.I)),
    ("ψεύτικος ρόλος", re.compile(r"(^|[\n.])\s*(system|assistant|developer)\s*:", re.I)),
    ("ακύρωση οδηγιών", re.compile(r"\b(ignore|disregard|forget)\b.{0,40}\b(instructions?|rules|prompt)\b", re.I)),
    ("οδηγίες του βοηθού", re.compile(r"\bsystem\s+prompt\b", re.I)),
]
_NORMALIZED = [   # πάνω σε κείμενο χωρίς τόνους και κεφαλαία
    ("ακύρωση οδηγιών", re.compile(
        r"\b(αγνοησε|ξεχνα|ξεχασε|παραβλεψε)\b.{0,40}\b(οδηγιες|εντολες|κανονες)\b")),
]


def rules(text: str) -> str | None:
    """Το όνομα του κανόνα που παραβιάζει το μήνυμα, ή None."""
    for name, rx in _RAW:
        if rx.search(text):
            return name
    plain = C.normalize(text)
    for name, rx in _NORMALIZED:
        if rx.search(plain):
            return name
    return None


# ── 2. Μοντέλο ────────────────────────────────────────────────
SCHEMA = {"type": "object", "required": ["labels"], "properties": {"labels": {
    "type": "array", "items": {"type": "object", "required": ["n", "label"], "properties": {
        "n": {"type": "integer"}, "label": {"type": "string", "enum": list(LABELS)}}}}}}


class Security:
    """Ο Security agent: κανόνες σε όλο το μήνυμα και ετικέτα ανά αίτημα από το μοντέλο."""

    def __init__(self, *, model: llm.OllamaModel, rules: bool = True, scope: str = "after",
                 emergency: str = "answer", lang: str = "en", prompts: dict | None = None):
        choice("security.scope", scope, ("before", "after"))
        choice("security.emergency", emergency, ("block", "answer"))
        choice("security.lang", lang, ("el", "en"))
        self.model, self.rules_on = model, bool(rules)
        self.scope, self.emergency, self.lang = scope, emergency, lang
        self.prompts = plain(prompts) if prompts is not None else prompts_of("security")
        for lang_ in ("el", "en"):
            if list(self.prompts[lang_]["criteria"]) != list(LABELS):
                raise ValueError(f"security.prompts.{lang_}.criteria: χρειάζονται οι ετικέτες {', '.join(LABELS)}")

    @classmethod
    def from_settings(cls, s: Settings) -> Security:
        """Από τις παλιές ρυθμίσεις (config.py, μεταβλητές MA_*)."""
        return cls(model=llm.OllamaModel(name=s.model("security"), url=s.url("security"), timeout=s.timeout),
                   rules=s.rules, scope=s.scope, emergency=s.emergency, lang=s.decision_lang,
                   prompts=prompts_of("security"))

    def params(self) -> dict:
        return {"model": self.model.params(), "rules": self.rules_on, "scope": self.scope,
                "emergency": self.emergency, "lang": self.lang, "prompts": self.prompts}

    def system(self) -> str:
        """Το system prompt για LLM: το πρότυπο του llm.system με τα criteria μιας γλώσσας."""
        llm_ = self.prompts["llm"]
        listing = "\n".join(f'- "{k}": {v}' for k, v in self.prompts[llm_["criteria_lang"]]["criteria"].items())
        return llm_["system"].format(criteria=listing)

    def rule(self, text: str) -> str | None:
        """Ο κανόνας που παραβιάζει το μήνυμα, ή None. Με rules: false, πάντα None."""
        return rules(text) if self.rules_on else None

    def blocks(self, label: str) -> bool:
        """Κόβεται το αίτημα πριν από τον ταξινομητή;"""
        return (label in ALWAYS_BLOCKED
                or (label == "out_of_scope" and self.scope == "before")
                or (label == "emergency" and self.emergency == "block"))

    def check(self, segments: list) -> Step:
        """Βάζει ετικέτα σε κάθε αίτημα (segment.label) και επιστρέφει τη γραμμή του ίχνους."""
        model = self.model
        total, notes = 0.0, []
        try:
            if model.decision:
                p = self.prompts[self.lang]
                question = {"kind": {"type": "choice", "instructions": p["question"],
                                     "criteria": dict(p["criteria"])}}
                for seg in segments:
                    ans, secs = model.decide(seg.view or seg.text, question)
                    total += secs
                    seg.label = ans["kind"]["choice"] if ans["kind"]["choice"] in LABELS else "request"
                    notes.append(f"{seg.label} ({ans['kind']['probabilities'].get(seg.label, 0):.2f})")
            else:
                user = "\n".join(f"{i}. {seg.view or seg.text}" for i, seg in enumerate(segments, 1))
                out, total = model.chat_json(self.system(), user, SCHEMA)
                found = {x.get("n"): x.get("label") for x in out.get("labels", []) if isinstance(x, dict)}
                for i, seg in enumerate(segments, 1):
                    seg.label = found.get(i) if found.get(i) in LABELS else "request"
                    notes.append(seg.label)
        except (llm.ModelError, KeyError, TypeError) as exc:
            for seg in segments:
                seg.label = "request"
            return Step("security", "model", f"το μοντέλο δεν απάντησε, τα αιτήματα περνούν ({exc})"[:160],
                        round(total, 2), model.name)
        return Step("security", "model", " · ".join(notes), round(total, 2), model.name)


class FixedSecurity(Security):
    """Security χωρίς μοντέλο, με ετικέτα που ορίζουμε εμείς, για να δούμε τι κάνει το υπόλοιπο
    pipeline (configs/security/fixed.yaml). Με label: request και rules: false είναι «χωρίς
    Security» (configs/security/off.yaml). Δεν χρησιμοποιεί το model που του δίνεται."""

    def __init__(self, *, label: str = "request", rules: bool = False, scope: str = "after",
                 emergency: str = "answer", model=None):
        choice("security.label", label, LABELS)
        choice("security.scope", scope, ("before", "after"))
        choice("security.emergency", emergency, ("block", "answer"))
        self.model, self.label, self.rules_on = None, label, bool(rules)
        self.scope, self.emergency, self.lang = scope, emergency, "en"

    def params(self) -> dict:
        return {"label": self.label, "rules": self.rules_on, "scope": self.scope, "emergency": self.emergency}

    def check(self, segments: list) -> Step:
        for seg in segments:
            seg.label = self.label
        return Step("security", "fixed", f"δοσμένη ετικέτα: {self.label}")


# Ο παλιός τρόπος (μεταβλητές MA_*): οι ίδιες μέθοδοι, με ρυθμίσεις από το config.py.
def blocks(label: str, settings: Settings) -> bool:
    return Security.from_settings(settings).blocks(label)


def check(segments: list, settings: Settings) -> Step:
    return Security.from_settings(settings).check(segments)
