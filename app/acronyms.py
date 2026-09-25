"""
Ενοποίηση ακρωνυμίων πριν τον διορθωτή και το μοντέλο.

ΓΙΑΤΙ: το dataset γράφει κάθε ακρωνύμιο με ΜΙΑ μορφή («ΤΑΠ», ποτέ
«Τ.Α.Π.» ή «τέλος ακίνητης περιουσίας»). Ο πολίτης γράφει όλες τις
μορφές. Τρία πράγματα έσπαγαν:
  · τελείες — ο tokenizer κάνει το «Τ.Α.Π.» μεμονωμένα γράμματα
    (μη οφειλής ΤΑΠ 0,89 → Τ.Α.Π. 0,07)
  · αναπτύγματα και λατινικά που δεν υπάρχουν στο dataset
    («τελών ακίνητης περιουσίας» → κοιμητήρια με 0,90)
  · ο διορθωτής «διόρθωνε» ακρωνύμια που δεν ήξερε
    («κδαπ» → «καπη» → ΚΑΠΗ με 0,95)

ΛΥΣΗ: πίνακας mappings/acronyms.csv, κάθε γραφή → η μορφή που είδε το
μοντέλο στην εκπαίδευση. Η σύγκριση αγνοεί τόνους, κεφαλαία και
τελείες ανάμεσα στα γράμματα. Κάθε λέξη του πίνακα προστατεύεται από τον
διορθωτή (βλ. spellfix.correct).

ΤΙ ΔΕΝ ΚΑΝΕΙ: δεν αγγίζει ό,τι δεν είναι στον πίνακα. Ο ΚΟΚ λείπει
σκόπιμα: το dataset γράφει «ΚΟΚ» στα πρόστιμα και «Κ.Ο.Κ.» στη γενική
αίτηση και στις καταγγελίες, οπότε κάθε ενοποίηση χαλάει τη μία πλευρά.
Διορθώνεται μόνο στο dataset, με επανεκπαίδευση.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

ENABLED = True
TABLE = Path(__file__).resolve().parent.parent / "mappings" / "acronyms.csv"

_ACCENTS = {"α": "αά", "ε": "εέ", "η": "ηή", "ι": "ιίϊΐ", "ο": "οό",
            "υ": "υύϋΰ", "ω": "ωώ", "σ": "σς"}


def _strip(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s))
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()


def _pattern(variant: str) -> str:
    """
    «ταπ» → ταιριάζει ΤΑΠ, ταπ, Τ.Α.Π., τ.α.π — τελεία προαιρετική μετά
    από κάθε γράμμα. «τελ*» → οποιαδήποτε κατάληξη (τέλος, τελών, τέλη).
    """
    parts = []
    for word in _strip(variant).split():
        star = word.endswith("*")
        word = word.rstrip("*")
        letters = []
        for ch in word:
            if ch == "-":
                letters.append(r"[-\s]?")
            elif ch in _ACCENTS:
                letters.append(f"[{_ACCENTS[ch]}]\\.?")
            else:
                letters.append(re.escape(ch) + r"\.?")
        parts.append("".join(letters) + (r"\w*" if star else ""))
    # Όχι κομμάτι μεγαλύτερου ακρωνυμίου: το «ΚΕ.Π.Α.» (Κέντρο
    # Πιστοποίησης Αναπηρίας) δεν πρέπει να γίνει «ΚΕΠ.Α.».
    return r"(?<![\w.])" + r"\s+".join(parts) + r"(?!\w|\.\w)"


def _load() -> tuple[list, set]:
    rules, protected = [], set()
    if not TABLE.exists():
        return rules, protected
    with TABLE.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rewrite = row["rewrite"].strip()
            for v in row["variants"].split("|"):
                v = v.strip()
                if v:
                    rules.append((re.compile(_pattern(v), re.IGNORECASE), rewrite))
                    if " " not in v and "*" not in v:
                        protected.add(_strip(v).replace("-", ""))
            protected.update(_strip(w) for w in re.findall(r"\w+", rewrite))
    return rules, protected


_RULES, PROTECTED = _load()


def expand(text: str) -> str:
    """Κάθε γνωστή γραφή ακρωνυμίου → η μορφή του dataset."""
    if not ENABLED:
        return text
    text = str(text)
    for rx, rewrite in _RULES:
        # Αν ο πολίτης έγραψε ήδη το ανάπτυγμα («ΠΕΚ (Πιστοποιητικό
        # Ελέγχου Κατασκευής)»), να μην το διπλασιάσουμε.
        if "(" in rewrite and _strip(rewrite.split("(", 1)[1]).rstrip(")") in _strip(text):
            continue
        text = rx.sub(lambda m, rw=rewrite: _keep_period(m.group(0), rw), text)
    return text


def _keep_period(matched: str, rewrite: str) -> str:
    """«πληρωμή ΤΑΠ.» — η τελεία είναι της πρότασης, όχι του ακρωνυμίου.
    Μόνη τελεία στο τέλος = τελεία πρότασης, και επιστρέφεται."""
    if matched.endswith(".") and matched.count(".") == 1:
        return rewrite + "."
    return rewrite
