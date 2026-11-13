"""
Validator agent: δύο έλεγχοι, όπως στο σχήμα.

  check_intent  μετά το BERT: εξυπηρετεί κάποιος από τους υποψήφιους
                προορισμούς αυτό που ζητά ο πολίτης; Αν όχι, ο Welcome
                αναδιατυπώνει (case 1) ή ρωτά τον πολίτη (case 2).
  check_final   πριν φύγει η απάντηση: απαντά το κείμενο στο αίτημα;

Ο Validator κρίνει πάντα με τις λέξεις του πολίτη, ποτέ με την
αναδιατύπωση του Welcome: έτσι μια αναδιατύπωση που άλλαξε το νόημα κόβεται.
Αν το μοντέλο δεν απαντήσει, ισχύει ό,τι θα έκανε το σημερινό bot.
"""
from __future__ import annotations

import re

import connector as C
import greeklish

from multiagent import llm
from multiagent.config import Settings, is_decision_model
from multiagent.state import Candidate, Classification, Step

# ── μετά το BERT ──────────────────────────────────────────────
FIT = {
    "el": {"q": "Εξυπηρετεί η υπηρεσία του πεδίου `service` αυτό που ζητά ο πολίτης στο `citizen_message`;",
           "true": "Η υπηρεσία αφορά ακριβώς το θέμα που ζητά ο πολίτης.",
           "false": "Ο πολίτης ζητά κάτι διαφορετικό, κάτι που δεν είναι υπηρεσία του Δήμου, ή το μήνυμα "
                    "δεν αρκεί για να κριθεί."},
    "en": {"q": "Does the municipal service in `service` handle what the citizen asks for in `citizen_message`?",
           "true": "The service is about exactly the matter the citizen asks for.",
           "false": "The citizen asks for something different, for something that is not a municipal "
                    "service, or the message is not enough to tell."},
}
PICK = {
    "el": {"q": "Ποια υπηρεσία του Δήμου Ηρακλείου ζητά ο πολίτης;",
           "none": "Καμία από τις παραπάνω, ή το μήνυμα δεν αρκεί για να ξεχωρίσει μία."},
    "en": {"q": "Which service of the Municipality of Heraklion is the citizen asking for?",
           "none": "None of the above, or the message is not enough to single one out."},
}

SYSTEM_INTENT = """Είσαι ο ελεγκτής του εικονικού βοηθού του Δήμου Ηρακλείου. Ο πολίτης έγραψε ένα μήνυμα και η αναζήτηση πρότεινε υπηρεσίες του Δήμου.
Διάλεξε τον αριθμό της υπηρεσίας που εξυπηρετεί ΑΚΡΙΒΩΣ αυτό που ζητά ο πολίτης. Απάντησε 0 αν καμία δεν ταιριάζει, αν το θέμα δεν είναι υπηρεσία του Δήμου, ή αν το μήνυμα δεν αρκεί για να ξεχωρίσεις.
Στο "evidence" αντέγραψε 1 έως 4 λέξεις του μηνύματος που δείχνουν ποια υπηρεσία εννοεί ο πολίτης."""

SCHEMA_INTENT = {"type": "object", "required": ["choice", "evidence"], "properties": {
    "choice": {"type": "integer"}, "evidence": {"type": "string"}}}


def _stems(text: str) -> set:
    text = greeklish.to_greek(text) or text      # «metadimotefsi» = «μεταδημότευση»
    return {w[:5] for w in re.findall(r"\w+", C.normalize(text)) if len(w) >= 4}


def _line(c: Candidate) -> str:
    return f"{c.title}: {c.description}" if c.description else c.title


def check_intent(message: str, cls: Classification, settings: Settings) -> tuple:
    """(ο προορισμός που εγκρίθηκε ή None, γραμμή ίχνους)."""
    model, url, cands = settings.model("validator"), settings.url("validator"), cls.candidates
    if not cands:
        return None, Step("validator", "intent", "κανένας υποψήφιος προορισμός")
    try:
        if is_decision_model(model):
            t = settings.decision_lang
            if len(cands) == 1:
                state = {"citizen_message": message, "service": cands[0].title,
                         "description": cands[0].description}
                q = {"fits": {"type": "noul", "instructions": FIT[t]["q"],
                              "criteria": {"true": FIT[t]["true"], "false": FIT[t]["false"]}}}
                ans, secs = llm.decide(url, model, state, q, settings.timeout)
                p = float(ans["fits"]["noul"])
                chosen = cands[0] if p >= settings.threshold else None
                note = f"«{cands[0].title}» ταιριάζει με πιθανότητα {p:.2f}"
            else:
                keys = "abcdefgh"[:len(cands)]
                criteria = {k: _line(c) for k, c in zip(keys, cands)}
                criteria["none"] = PICK[t]["none"]
                q = {"pick": {"type": "choice", "instructions": PICK[t]["q"], "criteria": criteria}}
                ans, secs = llm.decide(url, model, message, q, settings.timeout)
                key = ans["pick"]["choice"]
                p = float(ans["pick"]["probabilities"].get(key, 0.0))
                chosen = cands[keys.index(key)] if key in keys and p >= settings.threshold else None
                note = (f"διάλεξε «{cands[keys.index(key)].title}» ({p:.2f})" if key in keys
                        else f"καμία από τις {len(cands)} ({p:.2f})")
        else:
            user = (f"Μήνυμα: {message}\nΥπηρεσίες:\n"
                    + "\n".join(f"{i}. {_line(c)}" for i, c in enumerate(cands, 1)))
            out, secs = llm.chat_json(url, model, SYSTEM_INTENT, user, SCHEMA_INTENT, settings.timeout)
            n, ev = out.get("choice"), str(out.get("evidence", ""))
            # Οι λέξεις που επικαλείται πρέπει να υπάρχουν στο μήνυμα του πολίτη.
            quoted = bool(_stems(ev)) and _stems(ev) <= _stems(message)
            chosen = cands[n - 1] if isinstance(n, int) and 1 <= n <= len(cands) and quoted else None
            note = (f"διάλεξε «{chosen.title}» («{ev}»)" if chosen
                    else f"απάντησε {n} («{ev}»){'' if quoted else ', οι λέξεις δεν είναι του πολίτη'}")
    except (llm.ModelError, KeyError, TypeError, ValueError) as exc:
        chosen = cands[0] if cls.confident else None      # όπως το σημερινό bot
        return chosen, Step("validator", "intent", f"το μοντέλο δεν απάντησε ({exc})"[:160], 0.0, model)
    return chosen, Step("validator", "intent", note, secs, model)


# ── πριν τον πολίτη ───────────────────────────────────────────
ANSWERS = {
    "el": {"q": "Απαντά το κείμενο του πεδίου `reply` σε αυτό που ζητά ο πολίτης στο `citizen_message`;",
           "true": "Το κείμενο δίνει την υπηρεσία, τα δικαιολογητικά ή τα στοιχεία επικοινωνίας για αυτό "
                   "ακριβώς που ζητά ο πολίτης.",
           "false": "Το κείμενο αφορά άλλο θέμα ή δεν δίνει αυτό που ζήτησε ο πολίτης."},
    "en": {"q": "Does the text in `reply` answer what the citizen asks for in `citizen_message`?",
           "true": "The text gives the service, the required documents or the contact details for exactly "
                   "what the citizen asks.",
           "false": "The text is about another matter or does not give what the citizen asked for."},
}

SYSTEM_FINAL = """Είσαι ο τελικός ελεγκτής του εικονικού βοηθού του Δήμου Ηρακλείου. Θα δεις το αίτημα ενός πολίτη και την απάντηση που ετοιμάστηκε.
Κρίνε αν η απάντηση αφορά αυτό ακριβώς που ζήτησε ο πολίτης. Μια απάντηση που δίνει τη σωστή υπηρεσία με τον σύνδεσμο ή το τηλέφωνό της είναι σωστή, ακόμα κι αν είναι σύντομη.
Στο "reason" γράψε μία σύντομη πρόταση."""

SCHEMA_FINAL = {"type": "object", "required": ["answers", "reason"], "properties": {
    "answers": {"type": "boolean"}, "reason": {"type": "string"}}}


def check_final(message: str, reply: str, settings: Settings) -> tuple:
    """(True αν η απάντηση καλύπτει το αίτημα, γραμμή ίχνους)."""
    model, url, reply = settings.model("final"), settings.url("final"), reply[:1500]
    try:
        if is_decision_model(model):
            t = settings.decision_lang
            q = {"answers": {"type": "noul", "instructions": ANSWERS[t]["q"],
                             "criteria": {"true": ANSWERS[t]["true"], "false": ANSWERS[t]["false"]}}}
            ans, secs = llm.decide(url, model, {"citizen_message": message, "reply": reply}, q,
                                   settings.timeout)
            p = float(ans["answers"]["noul"])
            ok, note = p >= settings.threshold, f"απαντά στο αίτημα με πιθανότητα {p:.2f}"
        else:
            out, secs = llm.chat_json(url, model, SYSTEM_FINAL, f"Αίτημα: {message}\n\nΑπάντηση:\n{reply}",
                                      SCHEMA_FINAL, settings.timeout)
            ok = bool(out.get("answers"))
            note = ("απαντά στο αίτημα" if ok else "δεν απαντά στο αίτημα") + f": {out.get('reason', '')}"[:120]
    except (llm.ModelError, KeyError, TypeError, ValueError) as exc:
        return True, Step("validator", "final", f"το μοντέλο δεν απάντησε ({exc})"[:160], 0.0, model)
    return ok, Step("validator", "final", note, secs, model)
