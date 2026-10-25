"""
Taxonomy Consistency Check
===========================
Το σύστημα έχει τέσσερα κομμάτια που πρέπει να συμφωνούν για τα ίδια intents:

    1. Expanded_Intent_Dataset_2.csv   — τι εκπαιδεύεται   (άτομο #3)
    2. dimos-intent-model/             — τι προβλέπει
    3. intent_map.csv                  — πού καταλήγει     (mapping)
    4. heraklion_eservices.json        — τι υπάρχει όντως   (KB)

Αν κάποιο ξεφύγει, το σύστημα σπάει ΣΙΩΠΗΛΑ: το μοντέλο προβλέπει ένα intent
που δεν υπάρχει στο mapping και ο πολίτης παίρνει λάθος URL — χωρίς exception,
χωρίς log. Αυτό το script το πιάνει.

Τρέξ' το κάθε φορά που αλλάζει οποιοδήποτε από τα τέσσερα.

Έξοδος: 0 = όλα εντάξει, 1 = βρέθηκαν σφάλματα (κατάλληλο για CI / pre-commit)

Χρήση:
    python3 validate_taxonomy.py
    python3 validate_taxonomy.py --map intent_map.csv --strict
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from clean_dataset import canon_label, strip_accents


def norm(label: str) -> str:
    """Η μορφή που βλέπει το μοντέλο: canonical + χωρίς τόνους."""
    return strip_accents(canon_label(label)).lower()


class Checker:
    def __init__(self) -> None:
        self.errors: list = []
        self.warnings: list = []
        self.notes: list = []

    def error(self, section: str, msg: str, items=None) -> None:
        self.errors.append((section, msg, list(items or [])))

    def warn(self, section: str, msg: str, items=None) -> None:
        self.warnings.append((section, msg, list(items or [])))

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    def report(self) -> int:
        for msg in self.notes:
            print(f"  {msg}")

        for label, bucket, icon in (
            ("ΣΦΑΛΜΑΤΑ", self.errors, "❌"),
            ("ΠΡΟΕΙΔΟΠΟΙΗΣΕΙΣ", self.warnings, "⚠️ "),
        ):
            if not bucket:
                continue
            print(f"\n{'─' * 66}")
            print(f"  {label}")
            print("─" * 66)
            for section, msg, items in bucket:
                print(f"  {icon} [{section}] {msg}")
                for it in items[:12]:
                    print(f"        {it}")
                if len(items) > 12:
                    print(f"        … και άλλα {len(items) - 12}")

        print("\n" + "═" * 66)
        if self.errors:
            print(f"  ❌ ΑΠΕΤΥΧΕ — {len(self.errors)} σφάλμα(τα), "
                  f"{len(self.warnings)} προειδοποίηση(εις)")
            print("═" * 66)
            return 1
        print(f"  ✅ ΟΚ — {len(self.warnings)} προειδοποίηση(εις)")
        print("═" * 66)
        return 0


# ══════════════════════════════════════════════════════════════
# Φορτώσεις
# ══════════════════════════════════════════════════════════════
def load_dataset_labels(path: Path, chk: Checker) -> set:
    raw_labels, labels = set(), set()
    with path.open(encoding="utf-8") as f:
        for lineno, r in enumerate(csv.DictReader(f), start=2):
            raw = r.get("intent")
            if raw is None or not str(raw).strip():
                chk.warn("dataset", f"κενό intent στη γραμμή {lineno}")
                continue
            raw_labels.add(str(raw))
            labels.add(norm(raw))

    # Ίδιο intent γραμμένο με πολλούς τρόπους
    variants: dict = {}
    for raw in raw_labels:
        variants.setdefault(norm(raw), set()).add(raw)
    for canonical, forms in variants.items():
        if len(forms) > 1:
            chk.warn("dataset",
                     f"το intent «{canonical}» εμφανίζεται με {len(forms)} γραφές",
                     sorted(repr(f) for f in forms))

    chk.note(f"dataset  : {len(labels)} intents  ({path.name})")
    return labels


def load_model_labels(model_dir: Path, chk: Checker) -> set:
    enc = model_dir / "label_encoder.joblib"
    if not enc.exists():
        chk.warn("model", f"δεν βρέθηκε {enc} — παραλείπω τους ελέγχους μοντέλου")
        return set()

    import joblib
    classes = {str(c) for c in joblib.load(enc).classes_}

    junk = {c for c in classes if c in {"nan", "none", ""} or not c.strip()}
    if junk:
        chk.error("model",
                  "το μοντέλο έχει κλάσεις-σκουπίδια (εκπαιδεύτηκε σε κενές γραμμές)",
                  sorted(junk))
    classes -= junk

    chk.note(f"μοντέλο  : {len(classes)} κλάσεις")
    return {norm(c) for c in classes}


def load_mapping(path: Path, chk: Checker) -> dict:
    if not path.exists():
        chk.warn("mapping",
                 f"δεν βρέθηκε {path} — παραλείπω τους ελέγχους mapping")
        return {}

    mapping: dict = {}
    with path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "intent" not in (reader.fieldnames or []):
            chk.error("mapping", f"λείπει στήλη 'intent' στο {path.name}")
            return {}
        if "url" not in (reader.fieldnames or []):
            chk.error("mapping", f"λείπει στήλη 'url' στο {path.name}")
            return {}

        for lineno, r in enumerate(reader, start=2):
            intent = norm(r.get("intent", ""))
            if not intent:
                continue
            if intent in mapping:
                chk.error("mapping",
                          f"διπλή εγγραφή για «{intent}» (γραμμή {lineno})")
            mapping[intent] = (r.get("url") or "").strip()

    chk.note(f"mapping  : {len(mapping)} intents  ({path.name})")
    return mapping


def load_kb_urls(path: Path, chk: Checker) -> set:
    if not path.exists():
        chk.warn("kb", f"δεν βρέθηκε {path} — παραλείπω τους ελέγχους KB")
        return set()
    kb = json.loads(path.read_text(encoding="utf-8"))
    chk.note(f"KB       : {len(kb)} υπηρεσίες  ({path.name})")
    return {r["url"] for r in kb}


# ══════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description="Έλεγχος συνέπειας taxonomy")
    ap.add_argument("--dataset", default="Expanded_Intent_Dataset_2.csv")
    ap.add_argument("--model-dir", default="dimos-intent-model")
    ap.add_argument("--map", default="intent_map.csv")
    ap.add_argument("--kb", default="data/heraklion_eservices.json")
    ap.add_argument("--strict", action="store_true",
                    help="Θεώρησε και τις προειδοποιήσεις ως σφάλματα")
    args = ap.parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.exists():
        sys.exit(f"Δεν βρέθηκε το dataset: {ds_path}")

    print("═" * 66)
    print("  ΕΛΕΓΧΟΣ ΣΥΝΕΠΕΙΑΣ TAXONOMY")
    print("═" * 66)

    chk = Checker()
    ds_labels = load_dataset_labels(ds_path, chk)
    model_labels = load_model_labels(Path(args.model_dir), chk)
    mapping = load_mapping(Path(args.map), chk)
    kb_urls = load_kb_urls(Path(args.kb), chk)

    # ── dataset ↔ μοντέλο ───────────────────────────────────────────────────
    if model_labels:
        missing = ds_labels - model_labels
        if missing:
            chk.error("dataset↔μοντέλο",
                      "intents στο dataset που το μοντέλο ΔΕΝ ξέρει "
                      "(χρειάζεται retrain)", sorted(missing))
        stale = model_labels - ds_labels
        if stale:
            chk.error("dataset↔μοντέλο",
                      "κλάσεις που το μοντέλο προβλέπει αλλά έφυγαν από το dataset "
                      "(θα δίνουν λάθος URL)", sorted(stale))

    # ── dataset ↔ mapping ───────────────────────────────────────────────────
    if mapping:
        unmapped = ds_labels - set(mapping)
        if unmapped:
            chk.error("dataset↔mapping",
                      "intents χωρίς αντιστοίχιση σε URL", sorted(unmapped))
        orphan = set(mapping) - ds_labels
        if orphan:
            chk.error("dataset↔mapping",
                      "εγγραφές mapping για intents που δεν υπάρχουν πια",
                      sorted(orphan))

        # ── mapping ↔ KB ────────────────────────────────────────────────────
        blank = sorted(i for i, u in mapping.items() if not u)
        if blank:
            chk.warn("mapping",
                     "intents με κενό URL (δηλωμένα ως μη παρεχόμενα)", blank)

        if kb_urls:
            bad = sorted(f"{i} → {u}" for i, u in mapping.items()
                         if u and u not in kb_urls)
            if bad:
                chk.error("mapping↔KB",
                          "URLs που δεν υπάρχουν στο KB", bad)

        # Πολλά intents στο ίδιο URL είναι θεμιτό, αλλά αξίζει να το ξέρεις
        by_url: dict = {}
        for intent, url in mapping.items():
            if url:
                by_url.setdefault(url, []).append(intent)
        shared = {u: v for u, v in by_url.items() if len(v) > 1}
        if shared:
            chk.warn("mapping",
                     f"{len(shared)} URLs εξυπηρετούν πολλαπλά intents "
                     f"(εντάξει, αλλά έλεγξε για διπλότυπα intents)",
                     [f"{', '.join(v)}  →  {u.split('/')[-1]}"
                      for u, v in list(shared.items())])

    code = chk.report()
    if args.strict and chk.warnings:
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
