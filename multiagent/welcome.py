"""
Welcome agent: παραλαμβάνει το μήνυμα και δίνει την απάντηση.

  smalltalk  χαιρετισμοί και «ευχαριστώ», όπως στο σημερινό bot (χωρίς μοντέλο)
  split      χωρίζει το μήνυμα σε αιτήματα
  reform     case 1 του σχήματος: ξαναγράφει ένα αίτημα που δεν αναγνωρίστηκε
  compose    ενώνει τις απαντήσεις των αιτημάτων σε ένα κείμενο

Ο Welcome δεν γράφει ο ίδιος το κείμενο της απάντησης: τα κείμενα και τα
στοιχεία επικοινωνίας έρχονται από τους πίνακες του bot. Το μοντέλο του
χρησιμοποιείται μόνο για να κόψει και να αναδιατυπώσει.

Ο agent είναι η κλάση Welcome· ρυθμίσεις στο configs/welcome/default.yaml.
"""
from __future__ import annotations

import re

import connector as C
import greeklish
import smalltalk as chitchat

from multiagent import classifier, llm
from multiagent.config import Settings, choice
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
        "or_these": "Αν εννοείτε κάποιο από τα παρακάτω, διαλέξτε το.",
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
        "or_these": "If you meant one of the following, please choose it.",
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
# αγγλικά και greeklish (split_langs=all)
_COMPOSITE_LATIN = re.compile(r"\b(and|also|plus|kai|ke|episis|epishs|episeis|akoma|epipleon)\b")
_SENTENCE_END = re.compile(r"[.!?;;·]\s+\S")


def _tokens(s: str) -> list:
    return C.normalize(re.sub(r"[^\w\s]", " ", s)).split()


def may_be_composite(text: str, any_lang: bool = False) -> bool:
    """Αξίζει να ρωτήσουμε το μοντέλο; Τα σύντομα, απλά μηνύματα δεν πληρώνουν κλήση."""
    plain = C.normalize(text)
    joined = _COMPOSITE.search(plain) or (any_lang and _COMPOSITE_LATIN.search(plain))
    return len(text.split()) >= 6 and bool(_SENTENCE_END.search(text) or joined)


def verbatim(part: str, message: str) -> bool:
    """Οι λέξεις του κομματιού υπάρχουν στο μήνυμα, με την ίδια σειρά. Το μοντέλο
    επιτρέπεται να κόψει, όχι να ξαναγράψει."""
    rest = iter(_tokens(message))
    return all(any(t == m for m in rest) for t in _tokens(part))


# ── case 1: αναδιατύπωση ─────────────────────────────────────
SYSTEM_REFORM = """Ο πολίτης έγραψε στον εικονικό βοηθό του Δήμου Ηρακλείου, αλλά η αναζήτηση δεν βρήκε την υπηρεσία που ζητά.
Ξαναγράψε το αίτημά του σε ΜΙΑ απλή, σωστή ελληνική πρόταση, όπως θα το έγραφε η σελίδα του Δήμου. Διόρθωσε ορθογραφία και greeklish, και μετάφρασε αν είναι σε άλλη γλώσσα.
Μην προσθέτεις κάτι που δεν είπε ο πολίτης και μην απαντάς στο αίτημα."""

SCHEMA_REFORM = {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}}


# ── σύνθεση ───────────────────────────────────────────────────
def _short(text: str, n: int = 70) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"


def compose(segments: list, lang: str) -> tuple:
    """(κείμενο, κουμπιά). Με περισσότερα από ένα αιτήματα, κάθε απάντηση
    ξεκινά με τις λέξεις του αιτήματός της."""
    t = TEXT[lang]
    many = sum(s.status in ("answered", "clarify", "out_of_scope") for s in segments) >= 2
    blocks, options, said = [], [], set()
    for s in segments:
        if s.status in ("blocked", "emergency"):    # ένα έτοιμο κείμενο ανά είδος
            key = CANNED.get(s.label, "blocked")
            if key not in said:
                said.add(key)
                blocks.append(t[key])
            continue
        if s.status == "out_of_scope":          # scope=after: ούτε BERT ούτε Validator βρήκαν υπηρεσία
            body = t["out_of_scope"]
            if s.options and not options:
                options, body = s.options, body + "\n" + t["or_these"]
        else:
            body = s.answer
            if s.status == "answered" and s.final_ok is False:
                body += "\n\n" + t["final_note"]
            if s.status == "answered" and s.label == "emergency" and "emergency" not in said:
                said.add("emergency")               # emergency=answer: πρώτα το 112
                body = t["emergency"] + "\n\n" + body
            if s.status == "clarify" and not options:
                options = s.options
        if many:
            body = t["for"].format(text=_short(s.text)) + "\n" + body
        blocks.append(body)
    return "\n\n".join(blocks), options


class Welcome:
    """Ο Welcome agent: παραλαβή, χωρισμός σε αιτήματα, αναδιατύπωση, σύνθεση της απάντησης."""

    def __init__(self, *, split_model: llm.OllamaModel | None = None,
                 reform_model: llm.OllamaModel | None = None, split: bool = True,
                 split_langs: str = "all", reform: bool = True):
        choice("welcome.split_langs", split_langs, ("el", "all"))
        if split and split_model is None:
            raise ValueError("welcome.split_model: χρειάζεται μοντέλο όταν split: true")
        if reform and reform_model is None:
            raise ValueError("welcome.reform_model: χρειάζεται μοντέλο όταν reform: true")
        self.split_model, self.reform_model = split_model, reform_model
        self.split_on, self.split_langs, self.reform_on = bool(split), split_langs, bool(reform)

    @classmethod
    def from_settings(cls, s: Settings) -> Welcome:
        """Από τις παλιές ρυθμίσεις (config.py, μεταβλητές MA_*)."""
        return cls(split_model=llm.OllamaModel(name=s.model("split"), url=s.url("split"), timeout=s.timeout),
                   reform_model=llm.OllamaModel(name=s.model("reform"), url=s.url("reform"), timeout=s.timeout),
                   split=s.split, split_langs=s.split_langs, reform=s.reform)

    def params(self) -> dict:
        return {"split_model": self.split_model.params() if self.split_model else None,
                "reform_model": self.reform_model.params() if self.reform_model else None,
                "split": self.split_on, "split_langs": self.split_langs, "reform": self.reform_on}

    # Χωρίς μοντέλο: ίδια σε κάθε ρύθμιση
    language_of = staticmethod(language_of)
    smalltalk = staticmethod(smalltalk)
    compose = staticmethod(compose)

    def split(self, text: str) -> tuple:
        """(αιτήματα, απόσπασμα, γραμμή ίχνους ή None).

        Με δύο ή περισσότερα αιτήματα επιστρέφει τα κομμάτια. Με ένα, επιστρέφει
        ολόκληρο το μήνυμα και, αν το μοντέλο απομόνωσε το αίτημα μέσα από φλύαρο
        κείμενο, το απόσπασμα ως δεύτερη ευκαιρία για τον ταξινομητή.
        """
        if not self.split_on or not may_be_composite(text, self.split_langs == "all"):
            return [text], "", None
        model = self.split_model
        try:
            out, secs = model.chat_json(SYSTEM_SPLIT, text, SCHEMA_SPLIT)
        except llm.ModelError as exc:
            return [text], "", Step("welcome", "split", f"το μοντέλο δεν απάντησε ({exc})"[:160], 0.0, model.name)
        parts = [str(p).strip() for p in out.get("requests", []) if str(p).strip()]
        if not parts or not all(verbatim(p, text) for p in parts):
            return [text], "", Step("welcome", "split", "απορρίφθηκε: άλλαξε τις λέξεις του πολίτη", secs, model.name)
        parts = [p for p in parts if len(_tokens(p)) >= 2 and not chitchat.detect(p)]
        parts = list(dict.fromkeys(parts))
        if len(parts) >= 2:
            return parts, "", Step("welcome", "split", f"{len(parts)} αιτήματα: " + " | ".join(parts), secs,
                                   model.name)
        if len(parts) == 1 and len(_tokens(parts[0])) <= 0.7 * len(_tokens(text)):
            return [text], parts[0], Step("welcome", "split", f"1 αίτημα, απόσπασμα: {parts[0]}", secs, model.name)
        return [text], "", Step("welcome", "split", "1 αίτημα", secs, model.name)

    def reform(self, text: str) -> tuple:
        """Case 1: (νέα διατύπωση ή None, γραμμή ίχνους)."""
        model = self.reform_model
        try:
            # 300 tokens: με 120 το JSON κοβόταν στη μέση στα φλύαρα μηνύματα (8 φορές στη μέτρηση 13/11).
            out, secs = model.chat_json(SYSTEM_REFORM, text, SCHEMA_REFORM, num_predict=300)
        except llm.ModelError as exc:
            return None, Step("welcome", "reform", f"το μοντέλο δεν απάντησε ({exc})"[:160], 0.0, model.name)
        new = re.sub(r"\s+", " ", str(out.get("text", ""))).strip()
        words = _tokens(new)
        # Ελληνική πρόταση, χωρίς αριθμούς και χωρίς να είναι το ίδιο το αίτημα.
        greek = sum("α" <= c <= "ω" for c in C.normalize(new)) >= 0.6 * max(1, sum(c.isalpha() for c in new))
        if not (2 <= len(words) <= 30) or not greek or re.search(r"\d{3,}", new) or words == _tokens(text):
            return None, Step("welcome", "reform", f"απορρίφθηκε: {new}"[:160], secs, model.name)
        return new, Step("welcome", "reform", new, secs, model.name)


# Ο παλιός τρόπος (μεταβλητές MA_*): οι ίδιες μέθοδοι, με ρυθμίσεις από το config.py.
def split(text: str, settings: Settings) -> tuple:
    return Welcome.from_settings(settings).split(text)


def reform(text: str, settings: Settings) -> tuple:
    return Welcome.from_settings(settings).reform(text)
