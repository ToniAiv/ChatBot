"""
Αναγνώριση τμήματος σε ερωτήσεις επικοινωνίας («τηλέφωνο της πολεοδομίας»).

ΓΙΑΤΙ: στο dataset v6 το intent «επικοινωνία» ξαναγράφτηκε, και ~65/100
προτάσεις του ρωτάνε για ΣΥΓΚΕΚΡΙΜΕΝΟ τμήμα. Το intent λέει «θέλει στοιχεία
επικοινωνίας», αλλά όχι ΠΟΙΟΥ τμήματος, οπότε η απάντηση ήταν πάντα το
τηλεφωνικό κέντρο. Επιλογή Β της ομάδας (12/10): το τμήμα το βρίσκει το bot
από το κείμενο, όπως η πτυχή («τι δικαιολογητικά») στο aspects.py.

ΠΟΤΕ: μόνο σε ερώτηση επικοινωνίας (τηλέφωνο, email, πού βρίσκεται,
ωράριο). Το «θέλω να κάνω καταγγελία στη δημοτική αστυνομία» είναι
αίτημα, όχι ερώτηση επικοινωνίας, και πάει στην υπηρεσία.
Το αν θα αντικαταστήσει την απάντηση του μοντέλου το αποφασίζει το
connector.department_override.

ΠΗΓΕΣ: τμήματα και τηλέφωνα στο mappings/departments.csv (επίσημος
κατάλογος του δήμου), τρόποι γραφής στο mappings/department_aliases.csv.
Νέο τμήμα ή νέος τρόπος γραφής = μία γραμμή στο CSV, χωρίς κώδικα.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

ENABLED = True
ALIASES = Path(__file__).resolve().parent.parent / "mappings" / "department_aliases.csv"

# Ερώτηση επικοινωνίας. Το «διεύθυνση» μόνο με άρθρο: «η διεύθυνση της
# πολεοδομίας» (πού βρίσκεται) ≠ «Διεύθυνση Καθαριότητας» (υπηρεσία).
CONTACT = re.compile(
    r"τηλεφων|\be ?mail\b|\bμειλ|επικοινων|\bφαξ\b|\bfax\b"
    r"|διευθυνση (του|της|των)\b|(ποια ειναι η|σε ποια) διευθυνση"
    r"|\bπου (ειναι|βρισκ|στεγαζ)|ωραρι|ωρες λειτουργ|τι ωρες|ποτε ανοιγ|ανοιχτ"
    r"|ποιος ειναι (ο )?υπευθυνος|να (μιλησω|καλεσω)\b")
# ΟΧΙ σκέτο «να πάρω»: «να πάρω αντίγραφα» = να αποκτήσω. Το «να πάρω
# τηλέφωνο» το πιάνει ήδη το «τηλεφων».


def _norm(text: str) -> str:
    s = unicodedata.normalize("NFD", str(text))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def _load() -> list:
    if not ALIASES.exists():
        return []
    with ALIASES.open(encoding="utf-8") as f:
        return [(r["code"], re.compile(r["patterns"]), bool(r.get("generic")), bool(r.get("org")))
                for r in csv.DictReader(f)]


_RULES = _load()


# «το ΓΡΑΦΕΙΟ αδεσπότων», «το ΤΜΗΜΑ κυκλοφορίας»: ο πολίτης ζητά ρητά το
# τμήμα, όχι την υπηρεσία. Χωρίς αυτό, το «θέλω να μιλήσω με το γραφείο
# αδεσπότων» έδινε σύνδεσμο για καταγγελία αδέσποτου.
OFFICE = re.compile(r"\b(γραφει|τμημα|υπηρεσι|διευθυνσ)\w*")


def is_contact_question(text: str) -> bool:
    return bool(CONTACT.search(_norm(text)))


def names_office(text: str) -> bool:
    return bool(OFFICE.search(_norm(text)))


def detect(text: str) -> str | None:
    """Ο κωδικός του τμήματος που αναφέρεται στο κείμενο, ή None."""
    m = match(text)
    return m[0] if m else None


def match(text: str):
    """
    (κωδικός, είναι_φορέας) για το τμήμα που αναφέρεται στο κείμενο, ή None.
    «είναι_φορέας» αφορά τη ΓΡΑΜΜΗ που ταίριαξε: το «ΔΕΥΑΗ» είναι φορέας,
    το «ύδρευση» θέμα, αν και τα δύο οδηγούν στη ΔΕΥΑΗ.

    Αν ταιριάζουν πολλά, κερδίζει το ΕΙΔΙΚΟ έναντι του γενικού («τεχνική
    υπηρεσία ηλεκτροφωτισμού» → ηλεκτροφωτισμός), και μετά το μακρύτερο
    ταίριασμα («κοινωνικό φαρμακείο» → φαρμακείο, όχι κοινωνικά θέματα).
    """
    if not ENABLED:
        return None
    s = _norm(text)
    best = None
    for code, rx, generic, org in _RULES:
        m = rx.search(s)
        if m:
            score = (not generic, m.end() - m.start())
            if best is None or score > best[0]:
                best = (score, code, org)
    return (best[1], best[2]) if best else None
