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


# ══════════════════════════════════════════════════════════════
# Μνήμη ΥΠΗΡΕΣΙΑΣ
# ══════════════════════════════════════════════════════════════
# ΓΙΑΤΙ: το Pending θυμάται μόνο ό,τι ΔΕΝ καταλάβαμε. Μετά από σίγουρη
# απάντηση το bot ξεχνούσε την υπηρεσία: «θέλω βεβαίωση κατοικίας» →
# σύνδεσμος, «τι χαρτιά θέλει;» → «Για ποια υπηρεσία ρωτάτε;» (0/25 στο
# agent/followup_questions.csv, reports/service_memory.md).
# ΤΙ ΚΡΑΤΑΜΕ: την υπηρεσία (ή το τμήμα) της τελευταίας σίγουρης απάντησης, ή
# του κουμπιού που πάτησε ο πολίτης.
# ΠΩΣ ΧΡΗΣΙΜΟΠΟΙΕΙΤΑΙ: gui.Api.chat → connector.service_memory_answer, μόνο όταν
# το μήνυμα ρωτά κάτι συγκεκριμένο (aspects.py) και το BERT δεν δείχνει άλλη
# συγκεκριμένη υπηρεσία. Χωρίς μοντέλο: το κείμενο της σελίδας αυτούσιο.
# ΣΒΗΝΕΙ: με νέα σίγουρη υπηρεσία (αντικαθίσταται), με κουμπιά για άλλες
# υπηρεσίες (KEEP_ON_UNCLEAR), με «Νέα συζήτηση», ή μετά από SERVICE_TTL_SECONDS.
# ΔΙΑΚΟΠΤΗΣ: BOT_SERVICE_MEMORY=0 ./run.sh. Κλειστό, το bot συμπεριφέρεται ακριβώς
# όπως πριν.
SERVICE_MEMORY = os.environ.get("BOT_SERVICE_MEMORY", "1").strip().lower() not in ("0", "false", "off", "no")
SERVICE_TTL_SECONDS = 600
# Όταν το bot ΔΕΝ καταλαβαίνει ένα μήνυμα, η μνήμη μένει, εκτός αν προτείνει
# κουμπιά για ΑΛΛΕΣ υπηρεσίες (ένδειξη νέου θέματος). Πριν (23/10) έσβηνε πάντα,
# και μια ερώτηση συνέχειας που δεν αναγνωρίστηκε («ως πότε το προλαβαίνω;»)
# έκανε την επόμενη («τηλέφωνο;») να χάσει την υπηρεσία. BOT_MEMORY_KEEP_UNCLEAR=0:
# όπως πριν.
KEEP_ON_UNCLEAR = os.environ.get("BOT_MEMORY_KEEP_UNCLEAR", "1").strip().lower() not in ("0", "false", "off", "no")


class Service:
    """Η υπηρεσία της συζήτησης. Ένα αντικείμενο ανά παράθυρο / session."""

    def __init__(self):
        self._s = None

    def remember_service(self, rec: dict) -> None:
        if SERVICE_MEMORY and rec and rec.get("url"):
            self._s = {"kind": "svc", "rec": rec, "t": time.monotonic()}

    def remember_department(self, dept: dict) -> None:
        if SERVICE_MEMORY and dept:
            self._s = {"kind": "dept", "dept": dept, "t": time.monotonic()}

    def get(self) -> dict | None:
        """Χωρίς να τη σβήνει: ισχύει για όσες ερωτήσεις συνέχειας χρειαστεί."""
        if not SERVICE_MEMORY or not self._s or time.monotonic() - self._s["t"] > SERVICE_TTL_SECONDS:
            return None
        return self._s

    def clear(self) -> None:
        self._s = None
