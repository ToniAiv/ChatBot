"""Κοινές διαδρομές του project, αγκυρωμένες στη ρίζα.

Τα scripts έτρεχαν με σχετικά paths και απαιτούσαν cwd = New-praktiki/.
Μετά την αναδιοργάνωση σε υποφακέλους αυτό έσπασε. Εδώ ορίζεται μία
φορά η ρίζα και όλα τα υπόλοιπα προκύπτουν από αυτήν, ώστε τα scripts
να τρέχουν από οπουδήποτε.
"""
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
DATA     = ROOT / "data"
DATASETS = ROOT / "datasets"
MODELS   = ROOT / "models"
MAPPINGS = ROOT / "mappings"
REPORTS  = ROOT / "reports"

MODEL_218 = MODELS / "intent-model-218"
KB        = DATA / "heraklion_eservices.json"
KB_RICH   = DATA / "heraklion_eservices_enriched.json"
