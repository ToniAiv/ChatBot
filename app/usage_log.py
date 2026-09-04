"""
Καταγραφή χρήσης: μία γραμμή JSON ανά ερώτηση.

ΓΙΑΤΙ: όλα τα νούμερα του συστήματος προέρχονται από προτάσεις που
γράψαμε εμείς ή παρήγαγε LLM. Οι πραγματικές ερωτήσεις πολιτών και
υπαλλήλων είναι διαφορετικές — και η παρουσίαση στον δήμο είναι η
πρώτη ευκαιρία να τις δούμε. Χωρίς καταγραφή χάνονται.

ΠΡΟΣΩΠΙΚΑ ΔΕΔΟΜΕΝΑ: καταγράφεται το κείμενο της ερώτησης, που μπορεί
να περιέχει στοιχεία (όνομα, διεύθυνση, ΑΦΜ) αν ο πολίτης τα γράψει.
Το αρχείο μένει ΤΟΠΙΚΑ, δεν φεύγει πουθενά, και η καταγραφή σβήνει με
ENABLED = False. Πριν από οποιαδήποτε πραγματική χρήση από πολίτες
χρειάζεται απόφαση του δήμου για διατήρηση και ανωνυμοποίηση.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ENABLED = True
LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "usage.jsonl"


def log(query: str, lang: str, greek_query: str, intent: str | None,
        confidence: float | None, outcome: str, detail: str = "",
        elapsed_ms: float | None = None) -> None:
    """
    Ποτέ δεν πετάει σφάλμα προς τα πάνω: μια αποτυχία καταγραφής δεν
    επιτρέπεται να χαλάσει την απάντηση προς τον πολίτη.

    outcome: link | phone | refuse_low_confidence | refuse_out_of_scope
             | translation_failed
    """
    if not ENABLED:
        return
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "query": query,
            "lang": lang,
            "greek_query": greek_query if lang != "el" else None,
            "intent": intent,
            "confidence": round(confidence, 4) if confidence is not None else None,
            "outcome": outcome,
            "detail": detail,
            "ms": round(elapsed_ms) if elapsed_ms is not None else None,
        }
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
