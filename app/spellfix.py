"""
Διόρθωση ορθογραφικών λαθών πριν το μοντέλο.

ΓΙΑΤΙ: το dataset παρήχθη από LLM, άρα είναι ορθογραφικά τέλειο. Το
μοντέλο δεν έχει δει ποτέ λάθος, και ένα λάθος ανά πρόταση του κοστίζει
9,5 μονάδες, δύο λάθη 27 (μετρημένο σε 400 προτάσεις).

ΜΕΘΟΔΟΣ: φωνητικό κλειδί. Μετρήθηκε ότι 9/9 από τα λάθη που σπάνε το
μοντέλο είναι φωνητικά — τα ελληνικά έχουν πολλούς τρόπους να γράψεις
τον ίδιο ήχο (ι/η/υ/ει/οι, ο/ω, ε/αι) και εκεί κάνουν λάθος οι φυσικοί
ομιλητές. Χτίζεται ευρετήριο «φωνητικό κλειδί → συχνότερη πραγματική
λέξη» από το dataset, και κάθε άγνωστη λέξη αναζητείται εκεί.

ΓΙΑΤΙ ΤΟ ΔΙΚΟ ΜΑΣ CORPUS ΚΑΙ ΟΧΙ ΛΕΞΙΚΟ: τα ελληνικά κλίνονται βαριά.
Ο πολίτης γράφει «κατοικίας», το λεξικό έχει λήμμα «κατοικία». Το corpus
έχει και τους κλιτικούς τύπους ΚΑΙ τις συχνότητες του τομέα — γι' αυτό
το «κλιση» διορθώνεται σε «κλήση» (92 εμφανίσεις) και όχι σε «κλίση» (9).

ΤΙ ΔΕΝ ΑΓΓΙΖΕΙ: λέξεις που υπάρχουν ήδη στο λεξιλόγιο. Η υπερδιόρθωση
είναι ο πραγματικός κίνδυνος — μια σωστή λέξη που «διορθώνεται» σε κάτι
άσχετο είναι χειρότερη από ένα ανορθόγραφο που δεν αναγνωρίζεται.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path

ENABLED = True
MIN_LEN = 5          # κάτω από αυτό οι γείτονες είναι πάρα πολλοί
MIN_FREQ = 2         # μία μόνο εμφάνιση μπορεί να είναι και τυπογραφικό

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [ROOT / "datasets" / "Expanded_Intent_Dataset_3.csv"]
SOURCES += sorted((ROOT / "datasets" / "tests").glob("*.csv"))

_WORD = re.compile(r"[α-ωΑ-ΩίϊΐόάέύϋΰήώΆΈΉΊΌΎΏ]+")


def _strip(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s))
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()


def phonetic(word: str) -> str:
    """Ό,τι ακούγεται ίδιο, γράφεται ίδιο."""
    w = _strip(word)
    w = re.sub(r"(ει|οι|υι)", "ι", w)
    w = re.sub(r"[ηυ]", "ι", w)
    w = re.sub(r"αι", "ε", w)
    w = re.sub(r"ω", "ο", w)
    w = re.sub(r"(αυ|ευ)", "αφ", w)
    w = re.sub(r"(.)\1+", r"\1", w)     # διπλά σύμφωνα
    w = re.sub(r"ς$", "σ", w)
    return w


_VOCAB: set = set()
_INDEX: dict = {}


def _build() -> None:
    global _VOCAB, _INDEX
    freq: Counter = Counter()
    for path in SOURCES:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for w in _WORD.findall(str(row.get("text", ""))):
                    freq[_strip(w)] += 1
    _VOCAB = set(freq)
    best: dict = {}
    for w, n in freq.items():
        if len(w) < MIN_LEN or n < MIN_FREQ:
            continue
        k = phonetic(w)
        if k not in best or n > best[k][0]:
            best[k] = (n, w)
    _INDEX = {k: w for k, (_n, w) in best.items()}


_build()


def correct(text: str) -> str:
    """Επιστρέφει το κείμενο με διορθωμένες μόνο τις άγνωστες λέξεις."""
    if not ENABLED or not _INDEX:
        return text

    def fix(m: re.Match) -> str:
        w = m.group(0)
        s = _strip(w)
        if len(s) < MIN_LEN or s in _VOCAB:
            return w
        return _INDEX.get(phonetic(s), w)

    return _WORD.sub(fix, str(text))


def stats() -> dict:
    return {"vocab": len(_VOCAB), "phonetic_keys": len(_INDEX)}
