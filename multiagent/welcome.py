"""
Welcome agent: παραλαμβάνει το μήνυμα και δίνει την απάντηση.

  smalltalk  χαιρετισμοί και «ευχαριστώ», όπως στο σημερινό bot (χωρίς μοντέλο)
  split      χωρίζει το μήνυμα σε αιτήματα
  reform     case 1 του σχήματος: ξαναγράφει ένα αίτημα που δεν αναγνωρίστηκε
  compose    ενώνει τις απαντήσεις των αιτημάτων σε ένα κείμενο

Ο Welcome δεν γράφει ο ίδιος το κείμενο της απάντησης: τα κείμενα και τα
στοιχεία επικοινωνίας έρχονται από τους πίνακες του bot. Το μοντέλο του
χρησιμοποιείται μόνο για να κόψει και να αναδιατυπώσει.
"""
from __future__ import annotations

import re

import connector as C
import greeklish
import smalltalk as chitchat

from multiagent import classifier, llm
from multiagent.config import Settings
from multiagent.state import Step

TEXT = {
    "el": {
        "blocked": "Μπορώ να βοηθήσω μόνο με υπηρεσίες του Δήμου Ηρακλείου: πιστοποιητικά, βεβαιώσεις, "
                   "άδειες, αιτήματα για καθαριότητα, δρόμους και φωτισμό, πληρωμές. Γράψτε μου τι "
                   "χρειάζεστε από τον Δήμο.",
        "emergency": "Αν κινδυνεύει άνθρωπος αυτή τη στιγμή, καλέστε αμέσως το 112. Ο εικονικός βοηθός "
                     "δεν εξυπηρετεί επείγοντα περιστατικά.",
        "out_of_scope": "Δεν βρήκα υπηρεσία του Δήμου Ηρακλείου γι' αυτό το θέμα· ίσως αφορά άλλον φορέα. "
                        f"Αν πιστεύετε ότι αφορά τον Δήμο, καλέστε στο {C.PHONE}.",
        "for": "Για το «{text}»:",
        "final_note": f"Αν αυτό δεν καλύπτει ό,τι ρωτήσατε, γράψτε το με άλλα λόγια ή καλέστε στο {C.PHONE}.",
        "empty": "Παρακαλώ γράψτε μια ερώτηση.",
    },
    "en": {
        "blocked": "I can only help with services of the Municipality of Heraklion: certificates, permits, "
                   "requests about cleaning, roads and lighting, payments. Tell me what you need from the "
                   "Municipality.",
        "emergency": "If someone is in danger right now, call 112 immediately. This virtual assistant does "
                     "not handle emergencies.",
        "out_of_scope": "I could not find a Municipality of Heraklion service for this; it may concern "
                        f"another authority. If you think it concerns the Municipality, please call {C.PHONE}.",
        "for": "About “{text}”:",
        "final_note": f"If this does not cover what you asked, please rephrase it or call {C.PHONE}.",
        "empty": "Please type a question.",
    },
}
# Η ετικέτα του Security → το έτοιμο κείμενο
CANNED = {"manipulation": "blocked", "abuse": "blocked", "emergency": "emergency",
          "out_of_scope": "out_of_scope"}


def language_of(text: str) -> str:
    """Η γλώσσα του πολίτη, όπως την αποφασίζει το bot (τα greeklish μετρούν ως ελληνικά)."""
    if C.ENGLISH_SUPPORT and C.detect_language(text) == "en" and not greeklish.to_greek(text):
        return "en"
    return "el"


def smalltalk(text: str) -> dict | None:
    S = classifier.load()
    return C.smalltalk_answer(text, S["vectorizer"], S["tfidf_matrix"], S["valid_kb"],
                              S["service_table"], S["no_service"])


# ── χωρισμός ──────────────────────────────────────────────────
SYSTEM_SPLIT = """Είσαι ο βοηθός υποδοχής του Δήμου Ηρακλείου. Ο πολίτης έγραψε ένα μήνυμα. Βρες τα ξεχωριστά αιτήματα που περιέχει.
Κανόνες:
- Γράψε κάθε αίτημα ΜΕ ΤΙΣ ΙΔΙΕΣ ΛΕΞΕΙΣ του πολίτη, αντιγραμμένες από το μήνυμα. Μην αλλάζεις, μην διορθώνεις και μην προσθέτεις λέξεις.
- Χώρισε ΜΟΝΟ αν ο πολίτης ζητά δύο ή περισσότερα διαφορετικά πράγματα. Ερωτήσεις για το ίδιο θέμα, π.χ. τα δικαιολογητικά και το τηλέφωνο της ίδιας υπηρεσίας, είναι ΕΝΑ αίτημα.
- Χαιρετισμοί και προτάσεις που μόνο περιγράφουν την κατάσταση δεν είναι ξεχωριστά αιτήματα. Κράτα μαζί με το αίτημα μόνο όσες λέξεις χρειάζονται για να γίνει κατανοητό.
- Αν το μήνυμα έχει ένα αίτημα, δώσε ένα."""

SCHEMA_SPLIT = {"type": "object", "required": ["requests"], "properties": {"requests": {
    "type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "string"}}}}

_COMPOSITE = re.compile(r"\b(και|επισης|ακομα|επιπλεον)\b")
_SENTENCE_END = re.compile(r"[.!?;;·]\s+\S")


def _tokens(s: str) -> list:
    return C.normalize(re.sub(r"[^\w\s]", " ", s)).split()


def may_be_composite(text: str) -> bool:
    """Αξίζει να ρωτήσουμε το μοντέλο; Τα σύντομα, απλά μηνύματα δεν πληρώνουν κλήση."""
    return len(text.split()) >= 6 and bool(_SENTENCE_END.search(text) or _COMPOSITE.search(C.normalize(text)))


def verbatim(part: str, message: str) -> bool:
    """Οι λέξεις του κομματιού υπάρχουν στο μήνυμα, με την ίδια σειρά. Το μοντέλο
    επιτρέπεται να κόψει, όχι να ξαναγράψει."""
    rest = iter(_tokens(message))
    return all(any(t == m for m in rest) for t in _tokens(part))


def split(text: str, settings: Settings) -> tuple:
    """(αιτήματα, απόσπασμα, γραμμή ίχνους ή None).

    Με δύο ή περισσότερα αιτήματα επιστρέφει τα κομμάτια. Με ένα, επιστρέφει
    ολόκληρο το μήνυμα και, αν το μοντέλο απομόνωσε το αίτημα μέσα από φλύαρο
    κείμενο, το απόσπασμα ως δεύτερη ευκαιρία για τον ταξινομητή.
    """
    if not settings.split or not may_be_composite(text):
        return [text], "", None
    model = settings.model("split")
    try:
        out, secs = llm.chat_json(settings.url("split"), model, SYSTEM_SPLIT, text, SCHEMA_SPLIT,
                                  settings.timeout)
    except llm.ModelError as exc:
        return [text], "", Step("welcome", "split", f"το μοντέλο δεν απάντησε ({exc})"[:160], 0.0, model)
    parts = [str(p).strip() for p in out.get("requests", []) if str(p).strip()]
    if not parts or not all(verbatim(p, text) for p in parts):
        return [text], "", Step("welcome", "split", "απορρίφθηκε: άλλαξε τις λέξεις του πολίτη", secs, model)
    parts = [p for p in parts if len(_tokens(p)) >= 2 and not chitchat.detect(p)]
    parts = list(dict.fromkeys(parts))
    if len(parts) >= 2:
        return parts, "", Step("welcome", "split", f"{len(parts)} αιτήματα: " + " | ".join(parts), secs, model)
    if len(parts) == 1 and len(_tokens(parts[0])) <= 0.7 * len(_tokens(text)):
        return [text], parts[0], Step("welcome", "split", f"1 αίτημα, απόσπασμα: {parts[0]}", secs, model)
    return [text], "", Step("welcome", "split", "1 αίτημα", secs, model)


# ── case 1: αναδιατύπωση ─────────────────────────────────────
SYSTEM_REFORM = """Ο πολίτης έγραψε στον εικονικό βοηθό του Δήμου Ηρακλείου, αλλά η αναζήτηση δεν βρήκε την υπηρεσία που ζητά.
Ξαναγράψε το αίτημά του σε ΜΙΑ απλή, σωστή ελληνική πρόταση, όπως θα το έγραφε η σελίδα του Δήμου. Διόρθωσε ορθογραφία και greeklish, και μετάφρασε αν είναι σε άλλη γλώσσα.
Μην προσθέτεις κάτι που δεν είπε ο πολίτης και μην απαντάς στο αίτημα."""

SCHEMA_REFORM = {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}}


def reform(text: str, settings: Settings) -> tuple:
    """(νέα διατύπωση ή None, γραμμή ίχνους)."""
    model = settings.model("reform")
    try:
        # 300 tokens: με 120 το JSON κοβόταν στη μέση στα φλύαρα μηνύματα (8 φορές στη μέτρηση 13/11).
        out, secs = llm.chat_json(settings.url("reform"), model, SYSTEM_REFORM, text, SCHEMA_REFORM,
                                  settings.timeout, num_predict=300)
    except llm.ModelError as exc:
        return None, Step("welcome", "reform", f"το μοντέλο δεν απάντησε ({exc})"[:160], 0.0, model)
    new = re.sub(r"\s+", " ", str(out.get("text", ""))).strip()
    words = _tokens(new)
    # Ελληνική πρόταση, χωρίς αριθμούς και χωρίς να είναι το ίδιο το αίτημα.
    greek = sum("α" <= c <= "ω" for c in C.normalize(new)) >= 0.6 * max(1, sum(c.isalpha() for c in new))
    if not (2 <= len(words) <= 30) or not greek or re.search(r"\d{3,}", new) or words == _tokens(text):
        return None, Step("welcome", "reform", f"απορρίφθηκε: {new}"[:160], secs, model)
    return new, Step("welcome", "reform", new, secs, model)


# ── σύνθεση ───────────────────────────────────────────────────
def _short(text: str, n: int = 70) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"


def compose(segments: list, lang: str) -> tuple:
    """(κείμενο, κουμπιά). Με περισσότερα από ένα αιτήματα, κάθε απάντηση
    ξεκινά με τις λέξεις του αιτήματός της."""
    t = TEXT[lang]
    many = sum(s.status in ("answered", "clarify") for s in segments) >= 2
    blocks, options, said = [], [], set()
    for s in segments:
        if s.status in ("answered", "clarify", "error"):
            body = s.answer
            if s.status == "answered" and s.final_ok is False:
                body += "\n\n" + t["final_note"]
            if s.status == "clarify" and not options:
                options = s.options
            if many:
                body = t["for"].format(text=_short(s.text)) + "\n" + body
            blocks.append(body)
        else:                                   # κομμένο από τον Security: ένα έτοιμο κείμενο ανά είδος
            key = CANNED.get(s.label, "blocked")
            if key not in said:
                said.add(key)
                blocks.append(t[key])
    return "\n\n".join(blocks), options
