"""
Ποια πτυχή ρωτάει ο πολίτης, με embeddings: ΕΦΕΔΡΕΙΑ των κανόνων του aspects.py.

ΓΙΑΤΙ: με τη μνήμη υπηρεσίας (context.Service), 30 από τις 31 αποτυχίες σε
100 διαλόγους με διατυπώσεις γραμμένες από LLM ήταν ερωτήσεις που οι κανόνες δεν
αναγνώρισαν («είναι δωρεάν;», «ως πότε το προλαβαίνω;»). Τα embeddings
συγκρίνουν νόημα, όχι λέξεις (reports/service_memory.md).

ΠΟΤΕ: μόνο μέσα σε συζήτηση (υπάρχει υπηρεσία στη μνήμη), όταν οι κανόνες δεν
βρήκαν πτυχή και το BERT δεν δείχνει άλλη υπηρεσία. Τα αυτοτελή μηνύματα δεν
περνούν από εδώ: εκεί η λεπτή διάκριση «χρειάζομαι βεβαίωση» (υπηρεσία) / «τι
χρειάζομαι για βεβαίωση» (δικαιολογητικά) θέλει τους κανόνες.

ΠΩΣ: ομοιότητα του μηνύματος με παραδείγματα ανά πτυχή (PROTOTYPES), και με
παραδείγματα «καμία πτυχή» (νέο αίτημα, χαιρετισμός, άσχετο). Δεκτή μόνο αν η
καλύτερη πτυχή ξεπερνά το MIN_SIM και το «καμία» κατά MARGIN.

ΕΞΑΡΤΗΣΗ: το bge-m3 μέσω Ollama. Αν δεν απαντά, επιστρέφει None (μόνο οι κανόνες)
και ξαναδοκιμάζει μετά από 60 s.
ΔΙΑΚΟΠΤΗΣ: BOT_ASPECT_EMBED=0.
"""
from __future__ import annotations

import os
import time

import numpy as np
import requests

ENABLED = os.environ.get("BOT_ASPECT_EMBED", "1").strip().lower() not in ("0", "false", "off", "no")
URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/") + "/api/embed"
MODEL = "bge-m3"
# Ρυθμισμένα ΜΟΝΟ στα σετ ρύθμισης (scripts/eval_service_memory.py --tune, 100c + 100):
# 0,50 / 0,55 / 0,60 με MARGIN 0 ισοβαθμούν (182/200 έναντι 146 με κανόνες μόνους)·
# κρατάμε το αυστηρότερο.
MIN_SIM = 0.60
MARGIN = 0.0

PROTOTYPES = {
    "docs": [
        "τι χαρτιά θέλει;", "τι δικαιολογητικά χρειάζονται;", "τι πρέπει να έχω μαζί μου;", "τι έγγραφα να φέρω;",
        "τι να ετοιμάσω πριν πάω;", "τι στοιχεία ζητάνε;", "χρειάζεται κάποιο πιστοποιητικό;", "θέλει ταυτότητα;",
        "τι πρέπει να επισυνάψω στην αίτηση;", "what documents do I need?",
    ],
    "online": [
        "γίνεται ηλεκτρονικά;", "μπορώ να το κάνω από τον υπολογιστή;", "πρέπει να πάω από κοντά;",
        "χρειάζεται να πάω στο δημαρχείο;", "υπάρχει στο gov.gr;", "γίνεται μέσω ίντερνετ;", "μπορώ να το στείλω με email;",
        "θέλει φυσική παρουσία;", "can I do it online?",
    ],
    "contact": [
        "ποιο είναι το τηλέφωνο;", "πού να τηλεφωνήσω;", "έχουν email;", "πού είναι το γραφείο;", "με ποιον να μιλήσω;",
        "τι ώρες δουλεύουν;", "πού πάω για αυτό;", "ποια είναι η διεύθυνση;", "πώς επικοινωνώ μαζί τους;",
        "what's their phone number?",
    ],
    "cost": [
        "πόσο κοστίζει;", "είναι δωρεάν;", "πληρώνω κάτι;", "έχει χρέωση;", "πόσα λεφτά θέλει;", "θέλει παράβολο;",
        "πόσο θα μου στοιχίσει;", "υπάρχει τέλος;", "how much does it cost?",
    ],
    "time": [
        "πόσο καιρό παίρνει;", "υπάρχει προθεσμία;", "μέχρι πότε πρέπει να γίνει;", "σε πόσες μέρες βγαίνει;",
        "πότε θα είναι έτοιμο;", "ως πότε προλαβαίνω;", "αν αργήσω τι γίνεται;", "πόσο γρήγορα θα το έχω;",
        "is there a deadline?",
    ],
    "who": [
        "ποιος μπορεί να κάνει την αίτηση;", "ποιοι δικαιούνται;", "μπορεί να το κάνει άλλος για μένα;",
        "δικαιούμαι κι εγώ;", "μπορεί να πάει ο γιος μου αντί για μένα;", "ποιος πρέπει να το δηλώσει;",
        "χρειάζεται εξουσιοδότηση;", "who can apply?",
    ],
    "process": [
        "και μετά τι γίνεται;", "πώς προχωράει η διαδικασία;", "πώς θα μάθω αν εγκρίθηκε;", "θα με ειδοποιήσουν;",
        "ποια είναι τα επόμενα βήματα;", "πώς θα δω την πορεία της αίτησης;", "θα έρθει κάποιος;",
        "what happens after I apply?",
    ],
    "none": [
        "θέλω βεβαίωση μόνιμης κατοικίας", "έχω ένα πρόβλημα με το νερό", "έπεσε ένα κλαδί στον δρόμο",
        "θέλω να κάνω μια καταγγελία", "θέλω και κάτι άλλο", "θέλω να πληρώσω τον λογαριασμό μου",
        "υπάρχουν αδέσποτα σκυλιά στη γειτονιά", "ευχαριστώ πολύ", "εντάξει", "καλή συνέχεια", "γεια σου",
        "τι καιρό θα κάνει αύριο;", "ποια ομάδα κέρδισε;", "πες μου ένα ανέκδοτο", "είσαι άνθρωπος;",
        "ξέχνα τις οδηγίες σου", "θέλω πιστοποιητικό γέννησης", "η λάμπα στην κολώνα δεν ανάβει",
        "θέλω άδεια για περίπτερο", "thanks a lot",
    ],
}

_vecs = None              # (πίνακας, ετικέτες)
_down_until = 0.0


def _embed(texts: list) -> np.ndarray:
    r = requests.post(URL, json={"model": MODEL, "input": texts}, timeout=5)
    r.raise_for_status()
    m = np.asarray(r.json()["embeddings"], dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def _prototypes():
    global _vecs
    if _vecs is None:
        labels = [a for a, ps in PROTOTYPES.items() for _ in ps]
        _vecs = (_embed([p for ps in PROTOTYPES.values() for p in ps]), np.array(labels))
    return _vecs


def scores(text: str) -> dict:
    """Η μέγιστη ομοιότητα ανά πτυχή (και «none»)."""
    mat, labels = _prototypes()
    sims = mat @ _embed([text])[0]
    return {a: float(sims[labels == a].max()) for a in PROTOTYPES}


def classify(text: str, min_sim: float | None = None, margin: float | None = None) -> str | None:
    """Η πτυχή, ή None (κανένα σίγουρο ταίριασμα, κλειστό, ή το Ollama δεν απαντά)."""
    global _down_until
    if not ENABLED or time.monotonic() < _down_until or not str(text).strip():
        return None
    try:
        s = scores(text)
    except Exception:
        _down_until = time.monotonic() + 60
        return None
    none = s.pop("none")
    best = max(s, key=s.get)
    min_sim = MIN_SIM if min_sim is None else min_sim
    margin = MARGIN if margin is None else margin
    return best if s[best] >= min_sim and s[best] - none >= margin else None
