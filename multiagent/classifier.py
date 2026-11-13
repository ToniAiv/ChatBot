"""
Ο ταξινομητής του σχήματος: το BERT του σημερινού bot, με τις ίδιες αποφάσεις.

Δεν έχει δικό του μοντέλο ούτε δικά του κείμενα. Καλεί τις συναρτήσεις του
connector.py με τη σειρά που τις καλεί το gui.Api.chat, ώστε ένα αίτημα που
περνά όλους τους ελέγχους να παίρνει ακριβώς την απάντηση του σημερινού bot.
Η μνήμη υπηρεσίας (ερωτήσεις συνέχειας) δεν έχει συνδεθεί ακόμα εδώ.
"""
from __future__ import annotations

import re

import connector as C
import gui

from multiagent.state import Candidate, Classification


def load() -> dict:
    """Το pipeline του bot (BERT, πίνακες), φορτωμένο μία φορά."""
    if not gui._STATE["ready"]:
        gui._load_pipeline()
        if gui._STATE["error"]:
            raise RuntimeError(gui._STATE["error"])
    return gui._STATE


def _trim(s: str, n: int = 220) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


def _candidate(intent: str, score: float) -> Candidate | None:
    """Ο προορισμός ενός intent, ή None όταν το bot θα έλεγε «εκτός αρμοδιότητας»."""
    S = load()
    if intent in (S["no_service"] or {}):
        dept = S["no_service"][intent]
        if not dept:
            return None
        return Candidate(intent, score, "phone", dept.get("name", ""), dept=dept,
                         description=f"Τηλεφωνική επικοινωνία με το τμήμα «{dept.get('name', '')}».")
    cand, _src = C.find_candidates(intent, S["vectorizer"], S["tfidf_matrix"], S["valid_kb"],
                                   S["service_table"])
    if not cand or cand[0][0] < C.MIN_TFIDF_SCORE:
        return None
    rec = cand[0][1]
    described = (C.SECTIONS.get(rec["url"]) or {}).get("sections", {}).get("Περιγραφή", "")
    return Candidate(intent, score, "link", rec["title"], url=rec["url"], description=_trim(described))


def same_place(a: Candidate, b: Candidate) -> bool:
    """Ίδιος προορισμός: ίδια σελίδα, ή ίδιο τμήμα."""
    return (a.url or (a.kind, a.title)) == (b.url or (b.kind, b.title))


def _top(intents: list) -> list:
    """Οι 5 πρώτες υπηρεσίες του BERT ως υποψήφιοι, χωρίς δύο φορές τον ίδιο προορισμό.
    Με αυτές διαλέγει ο Validator όταν validator_mode=choose."""
    out = []
    for intent, score in intents[:5]:
        c = _candidate(intent, score)
        if c and not any(same_place(c, o) for o in out):
            out.append(c)
    return out


def classify(text: str, pending: dict | None = None) -> Classification:
    """Ένα αίτημα → top-5 intents, βεβαιότητα προορισμού και υποψήφιοι προορισμοί.

    pending: το αίτημα που περίμενε διευκρίνιση από το προηγούμενο μήνυμα
    (context.Pending), με τον κανόνα «πρώτα το τρέχον» του bot.
    """
    S = load()
    try:
        greek, lang = C.resolve_query(text)
    except C.TranslationUnavailable as exc:
        return Classification(text, "en", error=str(exc)[:120])
    tok, mdl, le = S["tokenizer"], S["model"], S["label_encoder"]
    vec, mat, valid = S["vectorizer"], S["tfidf_matrix"], S["valid_kb"]
    table, ns = S["service_table"], S["no_service"]

    intents = C.detect_intent(greek, tok, mdl, le)
    out = Classification(greek, lang, aspect=C.carried_aspect(greek, pending))
    greek, intents, _used = C.combine_with_context(greek, intents, pending, tok, mdl, le, table, ns)
    out.intents = intents
    out.confidence = C.destination_confidence(intents, table, ns)
    out.top = _top(intents)

    # «τηλέφωνο της πολεοδομίας» → στοιχεία του τμήματος (departments.py)
    code = C.department_override(greek, intents, pending)
    dept = S["departments"].get(code) if code else None
    if dept:
        out.confident = True
        out.candidates = [Candidate(intents[0][0], 1.0, "dept", dept.get("name", ""), dept=dept,
                                    description=f"Στοιχεία επικοινωνίας του τμήματος «{dept.get('name', '')}».")]
        out.top = list(out.candidates)          # ρητή ερώτηση για τμήμα: καμία άλλη επιλογή
        return out

    if out.confidence >= C.MIN_BERT_CONFIDENCE:
        out.confident = True
        top = _candidate(intents[0][0], out.confidence)
        out.candidates = [top] if top else []
        return out

    # Αβέβαιο: οι επιλογές που θα έδειχνε το bot ως κουμπιά.
    fam = C.family_suggestions(intents, S["groups"], vec, mat, valid, table, ns)
    opts = fam[1] if fam else (C.suggestions(intents, vec, mat, valid, table, ns)
                               if C.should_suggest(intents) else [])
    out.family = fam[0][lang] if fam else ""
    out.candidates = [c for c in (_candidate(o["intent"], o.get("score", 0.0)) for o in opts) if c]
    return out


def answer(c: Candidate, cls: Classification) -> str:
    """Η απάντηση του bot για έναν προορισμό: έτοιμο κείμενο και στοιχεία από τους πίνακες."""
    S = load()
    if c.kind == "dept":
        return C.department_answer(c.dept, cls.lang, contact=True)
    if c.kind == "phone":
        return C.department_answer(c.dept, cls.lang)
    cand, src = C.find_candidates(c.intent, S["vectorizer"], S["tfidf_matrix"], S["valid_kb"],
                                  S["service_table"])
    return C.compose_answer(cls.current, c.intent, cand, src, cls.lang, aspect=cls.aspect)


def options(cls: Classification, pool: list | None = None) -> list:
    """Οι υποψήφιοι ως κουμπιά, στη μορφή του gui.Api.chat (pool: άλλοι από τους του bot)."""
    return [{"intent": c.intent, "title": c.title, "kind": "link" if c.kind == "link" else "phone",
             "url": c.url} for c in (cls.candidates if pool is None else pool)]


def clarify_text(cls: Classification) -> str:
    """Το μήνυμα του bot όταν δεν είναι σίγουρο (case 2 του σχήματος)."""
    m = C.MSG[cls.lang]
    if cls.error:
        return m["tr_fail"]
    if cls.candidates:
        return m["suggest_family"].format(title=cls.family) if cls.family else m["suggest"]
    return m["clarify_aspect" if cls.aspect else "clarify"]
