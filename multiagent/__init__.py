"""
Multi-agent σύστημα του εικονικού βοηθού (σχήμα της ομάδας, 10/11/2026):

    πολίτης → Welcome → Security → BERT → Validator → Welcome → Validator → πολίτης

Τρέχει δίπλα στο σημερινό bot (app/) χωρίς να το αλλάζει: ξαναχρησιμοποιεί
το pipeline του (connector.py) για την ταξινόμηση και για τα κείμενα των
απαντήσεων. Οδηγίες: multiagent/README.md. Μετρήσεις: reports/multiagent.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Τα modules του bot εισάγονται με το σκέτο όνομά τους (import connector),
# όπως στο agent/ και στα scripts/.
for _p in (ROOT / "scripts", ROOT / "app"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
