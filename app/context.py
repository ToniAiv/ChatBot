"""
Σύνδεση μηνυμάτων: μνήμη ΕΝΟΣ αβέβαιου μηνύματος.

ΓΙΑΤΙ: ο πολίτης συχνά χωρίζει το αίτημα σε δύο μηνύματα: «τι
δικαιολογητικά χρειάζονται;» και μετά «για την άδεια παιδότοπου». Κάθε
μήνυμα μόνο του είναι ελλιπές.

ΤΙ ΚΡΑΤΑΜΕ: μόνο το τελευταίο μήνυμα που το bot ΔΕΝ κατάλαβε σίγουρα, και
την πτυχή του («δικαιολογητικά», «κόστος»…). Όχι όλη τη συζήτηση: ένα
άσχετο πρώτο μήνυμα θα «μόλυνε» τα επόμενα (βλ. παρουσίαση «Σύνδεση
μηνυμάτων»). Σβήνει μόλις δοθεί απάντηση, όταν πατηθεί κουμπί, ή μετά από
TTL_SECONDS.

ΠΩΣ ΣΥΝΔΕΕΤΑΙ: connector.combine_with_context (κανόνας «πρώτα το Μ2»).
Μετρημένο σε scripts/eval_dialogue.py: 45 διάλογοι 21 → 32 σωστοί, με
μηδέν ζημιά· ανεξάρτητο σετ 18 διαλόγων 9 → 14, με μία ζημιά.

ΔΙΑΚΟΠΤΗΣ: BOT_CONTEXT=0 ./run.sh. Κλειστό, το bot συμπεριφέρεται ακριβώς
όπως πριν (ελέγχεται στο regression).
"""
from __future__ import annotations

import os
import time

ENABLED = os.environ.get("BOT_CONTEXT", "1").strip().lower() not in ("0", "false", "off", "no")
TTL_SECONDS = 300


class Pending:
    """Το εκκρεμές μήνυμα μιας συνομιλίας. Ένα αντικείμενο ανά παράθυρο."""

    def __init__(self):
        self._p = None

    def remember(self, greek: str, aspect: str | None) -> None:
        if ENABLED:
            self._p = {"greek": greek, "aspect": aspect, "t": time.monotonic()}

    def take(self) -> dict | None:
        """Επιστρέφει και ΣΒΗΝΕΙ το εκκρεμές: χρησιμοποιείται μία φορά."""
        p, self._p = self._p, None
        if not ENABLED or not p or time.monotonic() - p["t"] > TTL_SECONDS:
            return None
        return p

    def clear(self) -> None:
        self._p = None
