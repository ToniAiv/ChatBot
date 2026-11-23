"""
Validator agent: δύο έλεγχοι, όπως στο σχήμα.

  check_intent  μετά το BERT: εξυπηρετεί κάποιος από τους υποψήφιους
                προορισμούς αυτό που ζητά ο πολίτης; Αν όχι, ο Welcome
                αναδιατυπώνει (case 1) ή ρωτά τον πολίτη (case 2).
  check_final   πριν φύγει η απάντηση: απαντά το κείμενο στο αίτημα;

Ο Validator κρίνει πάντα με τις λέξεις του πολίτη, ποτέ με την
αναδιατύπωση του Welcome: έτσι μια αναδιατύπωση που άλλαξε το νόημα κόβεται.
Αν το μοντέλο δεν απαντήσει, ισχύει ό,τι θα έκανε το σημερινό bot.

Με mode: choose ο Validator δεν κρίνει μόνο την πρώτη υπηρεσία του
BERT: διαλέγει ανάμεσα στις 5 πρώτες, και όταν το BERT είναι σίγουρο
(reports/multiagent.md, 4η μέτρηση).

Ο agent είναι η κλάση Validator· ρυθμίσεις στο configs/validator/default.yaml.
Οι ερωτήσεις του (prompts) δεν είναι στον κώδικα: έρχονται από το
configs/validator/prompts/ και συμπληρώνονται την ώρα της κλήσης. Ο FixedValidator
δίνει απάντηση που ορίζουμε εμείς (configs/validator/off.yaml, fixed.yaml).
"""
from __future__ import annotations

import re

import connector as C
import greeklish
import yaml
from omegaconf import DictConfig, OmegaConf

from multiagent import ROOT, llm
from multiagent.config import Settings, choice
from multiagent.state import Candidate, Classification, Step

# ── οι ερωτήσεις (prompts) ───────────────────────────────────
PROMPTS_FILE = ROOT / "configs" / "validator" / "prompts" / "default.yaml"


def default_prompts() -> dict:
    """Οι ερωτήσεις του configs/validator/prompts/default.yaml, για τον παλιό τρόπο (χωρίς Hydra)."""
    return yaml.safe_load(PROMPTS_FILE.read_text(encoding="utf-8"))


# Η μορφή της απάντησης όταν ο Validator είναι LLM (κομμάτι του κώδικα, όχι του prompt)
SCHEMA_INTENT = {"type": "object", "required": ["choice", "evidence"], "properties": {
    "choice": {"type": "integer"}, "evidence": {"type": "string"}}}


def _stems(text: str) -> set:
    text = greeklish.to_greek(text) or text      # «metadimotefsi» = «μεταδημότευση»
    return {w[:5] for w in re.findall(r"\w+", C.normalize(text)) if len(w) >= 4}


def _line(c: Candidate, clean: bool = False) -> str:
    """Η περιγραφή μιας επιλογής για τον Validator. Με clean: χωρίς τυποποιημένες φράσεις,
    και για τα τμήματα τα θέματα που εξυπηρετούν (classifier._clean, _dept_topics)."""
    text = c.topic if clean else c.description
    return f"{c.title}: {text}" if text else c.title


SCHEMA_FINAL = {"type": "object", "required": ["answers", "reason"], "properties": {
    "answers": {"type": "boolean"}, "reason": {"type": "string"}}}


class Validator:
    """Ο Validator agent: διαλέγει την υπηρεσία μετά το BERT και ελέγχει την απάντηση πριν φύγει."""

    def __init__(self, *, model: llm.OllamaModel, final_model: llm.OllamaModel | None = None,
                 mode: str = "choose", text: str = "plain", hint: bool = False, threshold: float = 0.5,
                 final: bool = True, lang: str = "en", prompts: dict | None = None):
        choice("validator.mode", mode, ("fits", "choose"))
        choice("validator.text", text, ("plain", "clean"))
        choice("validator.lang", lang, ("el", "en"))
        if final and final_model is None:
            raise ValueError("validator.final_model: χρειάζεται μοντέλο όταν final: true")
        self.model, self.final_model = model, final_model
        self.mode, self.text, self.hint = mode, text, bool(hint)
        self.threshold, self.final_on, self.lang = float(threshold), bool(final), lang
        if isinstance(prompts, DictConfig):
            prompts = OmegaConf.to_container(prompts, resolve=True)
        self.prompts = prompts if prompts is not None else default_prompts()
        if lang not in self.prompts or "llm" not in self.prompts:
            raise ValueError(f"validator.prompts: χρειάζονται τα «{lang}» και «llm»")

    @classmethod
    def from_settings(cls, s: Settings) -> Validator:
        """Από τις παλιές ρυθμίσεις (config.py, μεταβλητές MA_*)."""
        return cls(model=llm.OllamaModel(name=s.model("validator"), url=s.url("validator"), timeout=s.timeout),
                   final_model=llm.OllamaModel(name=s.model("final"), url=s.url("final"), timeout=s.timeout),
                   mode=s.validator_mode, text=s.validator_text, hint=s.validator_hint,
                   threshold=s.threshold, final=s.final, lang=s.decision_lang, prompts=default_prompts())

    def params(self) -> dict:
        return {"model": self.model.params(),
                "final_model": self.final_model.params() if self.final_model else None,
                "mode": self.mode, "text": self.text, "hint": self.hint, "threshold": self.threshold,
                "final": self.final_on, "lang": self.lang, "prompts": self.prompts}

    def pool(self, cls: Classification) -> list:
        """Οι υποψήφιοι που κρίνει: του bot (fits), ή οι 5 πρώτες του BERT (choose).
        Μια ρητή ερώτηση για τμήμα («τηλέφωνο της πολεοδομίας») μένει πάντα όπως είναι."""
        explicit = bool(cls.candidates) and cls.candidates[0].kind == "dept"
        if self.mode == "choose" and cls.top and not explicit:
            return cls.top
        return cls.candidates

    def check_intent(self, message: str, cls: Classification) -> tuple:
        """(ο προορισμός που εγκρίθηκε ή None, γραμμή ίχνους)."""
        model, cands = self.model, self.pool(cls)
        if not cands:
            return None, Step("validator", "intent", "κανένας υποψήφιος προορισμός")
        try:
            if model.decision:
                p, clean = self.prompts[self.lang], self.text == "clean"
                if len(cands) == 1:
                    state = {"citizen_message": message, "service": cands[0].title,
                             "description": cands[0].topic if clean else cands[0].description}
                    q = {"fits": {"type": "noul", "instructions": p["fit"]["question"],
                                  "criteria": dict(p["fit"]["criteria"])}}
                    ans, secs = model.decide(state, q)
                    p = float(ans["fits"]["noul"])
                    chosen = cands[0] if p >= self.threshold else None
                    note = f"«{cands[0].title}» ταιριάζει με πιθανότητα {p:.2f}"
                else:
                    keys = "abcdefgh"[:len(cands)]
                    criteria = {k: _line(c, clean) for k, c in zip(keys, cands)}
                    criteria["none"] = p["pick"]["none_strict" if clean else "none"]
                    instructions = p["pick"]["question"]
                    if self.hint:
                        instructions += p["pick"]["hint"].format(conf=f"{cls.confidence:.2f}")
                    q = {"pick": {"type": "choice", "instructions": instructions, "criteria": criteria}}
                    ans, secs = model.decide(message, q)
                    key = ans["pick"]["choice"]
                    p = float(ans["pick"]["probabilities"].get(key, 0.0))
                    chosen = cands[keys.index(key)] if key in keys and p >= self.threshold else None
                    note = (f"διάλεξε «{cands[keys.index(key)].title}» ({p:.2f})" if key in keys
                            else f"καμία από τις {len(cands)} ({p:.2f})")
            else:
                user = (f"Μήνυμα: {message}\nΥπηρεσίες:\n"
                        + "\n".join(f"{i}. {_line(c, self.text == 'clean')}" for i, c in enumerate(cands, 1)))
                out, secs = model.chat_json(self.prompts["llm"]["intent"], user, SCHEMA_INTENT)
                n, ev = out.get("choice"), str(out.get("evidence", ""))
                # Οι λέξεις που επικαλείται πρέπει να υπάρχουν στο μήνυμα του πολίτη.
                quoted = bool(_stems(ev)) and _stems(ev) <= _stems(message)
                chosen = cands[n - 1] if isinstance(n, int) and 1 <= n <= len(cands) and quoted else None
                note = (f"διάλεξε «{chosen.title}» («{ev}»)" if chosen
                        else f"απάντησε {n} («{ev}»){'' if quoted else ', οι λέξεις δεν είναι του πολίτη'}")
        except (llm.ModelError, KeyError, TypeError, ValueError) as exc:
            chosen = cls.candidates[0] if cls.confident and cls.candidates else None    # όπως το σημερινό bot
            return chosen, Step("validator", "intent", f"το μοντέλο δεν απάντησε ({exc})"[:160], 0.0, model.name)
        return chosen, Step("validator", "intent", note, secs, model.name)

    def check_final(self, message: str, reply: str) -> tuple:
        """(True αν η απάντηση καλύπτει το αίτημα, γραμμή ίχνους)."""
        model, reply = self.final_model, reply[:1500]
        try:
            if model.decision:
                p = self.prompts[self.lang]
                q = {"answers": {"type": "noul", "instructions": p["answers"]["question"],
                                 "criteria": dict(p["answers"]["criteria"])}}
                ans, secs = model.decide({"citizen_message": message, "reply": reply}, q)
                p = float(ans["answers"]["noul"])
                ok, note = p >= self.threshold, f"απαντά στο αίτημα με πιθανότητα {p:.2f}"
            else:
                out, secs = model.chat_json(self.prompts["llm"]["final"], f"Αίτημα: {message}\n\nΑπάντηση:\n{reply}", SCHEMA_FINAL)
                ok = bool(out.get("answers"))
                note = ("απαντά στο αίτημα" if ok else "δεν απαντά στο αίτημα") + f": {out.get('reason', '')}"[:120]
        except (llm.ModelError, KeyError, TypeError, ValueError) as exc:
            return True, Step("validator", "final", f"το μοντέλο δεν απάντησε ({exc})"[:160], 0.0, model.name)
        return ok, Step("validator", "final", note, secs, model.name)


class FixedValidator(Validator):
    """Validator χωρίς μοντέλο, με απάντηση που ορίζουμε εμείς, για να δούμε τι κάνει το
    υπόλοιπο pipeline. Δεν χρησιμοποιεί τα model και final_model που του δίνονται.

        bert   εγκρίνει τη σίγουρη απάντηση του BERT, όπως το σημερινό bot· αλλιώς καμία
        first  εγκρίνει πάντα την πρώτη υποψήφια υπηρεσία
        none   δεν εγκρίνει ποτέ: το pipeline αναδιατυπώνει (case 1) ή ρωτά τον πολίτη (case 2)
    """

    def __init__(self, *, pick: str = "bert", mode: str = "fits", final: bool = False,
                 final_ok: bool = True, model=None, final_model=None):
        choice("validator.pick", pick, ("bert", "first", "none"))
        choice("validator.mode", mode, ("fits", "choose"))
        self.model = self.final_model = None
        self.mode, self.text, self.hint, self.threshold, self.lang = mode, "plain", False, 0.5, "en"
        self.final_on, self.final_ok, self.pick, self.prompts = bool(final), bool(final_ok), pick, {}

    def params(self) -> dict:
        return {"pick": self.pick, "mode": self.mode, "final": self.final_on, "final_ok": self.final_ok}

    def check_intent(self, message: str, cls: Classification) -> tuple:
        cands = self.pool(cls)
        if self.pick == "bert":
            chosen = cls.candidates[0] if cls.confident and cls.candidates else None
        else:
            chosen = cands[0] if self.pick == "first" and cands else None
        note = f"δοσμένη απάντηση ({self.pick}): " + (f"«{chosen.title}»" if chosen else "καμία")
        return chosen, Step("validator", "fixed", note)

    def check_final(self, message: str, reply: str) -> tuple:
        return self.final_ok, Step("validator", "fixed", f"τελικός έλεγχος: δοσμένη απάντηση {self.final_ok}")


# Ο παλιός τρόπος (μεταβλητές MA_*): οι ίδιες μέθοδοι, με ρυθμίσεις από το config.py.
def pool(cls: Classification, settings: Settings) -> list:
    return Validator.from_settings(settings).pool(cls)


def check_intent(message: str, cls: Classification, settings: Settings) -> tuple:
    return Validator.from_settings(settings).check_intent(message, cls)


def check_final(message: str, reply: str, settings: Settings) -> tuple:
    return Validator.from_settings(settings).check_final(message, reply)
