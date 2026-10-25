"""
Συγχώνευση των ανά-ενότητα αρχείων σε ένα dataset
==================================================
Τα αρχεία γράφονται ανά ενότητα στο dataset_new/ (20_prasino.csv, 23_adespota.csv,
…) για να δουλεύουμε μία κατηγορία τη φορά. Αυτό το script τα ενώνει σε ένα
αρχείο, έτοιμο να συγχωνευθεί με τη δουλειά των υπόλοιπων συναδέλφων.

Διατηρεί τη σειρά των ενοτήτων (αριθμητικά) και ελέγχει για διπλότυπα.

Χρήση:
    python3 merge_dataset.py
    python3 merge_dataset.py --out dataset_new/new_intents_dataset.csv
"""

import argparse
import csv
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from paths import ROOT, DATA, DATASETS, MODELS, MAPPINGS, MODEL_218, KB, KB_RICH


def sort_key(path: Path):
    """20_prasino → (20, '20_prasino'), 24a → (24, '24a_...') ώστε 24a πριν 24b."""
    m = re.match(r"(\d+)", path.stem)
    return (int(m.group(1)) if m else 999, path.stem)


def main() -> None:
    ap = argparse.ArgumentParser(description="Συγχώνευση αρχείων dataset")
    ap.add_argument("--dir", default=str(DATASETS / "dataset_new"))
    ap.add_argument("--out", default=str(DATASETS / "dataset_new" / "new_intents_dataset.csv"))
    args = ap.parse_args()

    out_path = Path(args.out)
    files = [p for p in sorted(Path(args.dir).glob("*.csv"), key=sort_key)
             if p.resolve() != out_path.resolve()]
    if not files:
        sys.exit(f"Δεν βρέθηκαν αρχεία στο {args.dir}/")

    merged: list = []
    per_file: OrderedDict = OrderedDict()

    for path in files:
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        merged.extend({"text": r["text"], "intent": r["intent"]} for r in rows)
        per_file[path.name] = rows

    dupes = [t for t, c in Counter(r["text"] for r in merged).items() if c > 1]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "intent"], quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(merged)

    counts = Counter(r["intent"] for r in merged)

    print("═" * 66)
    print("  ΣΥΓΧΩΝΕΥΣΗ DATASET")
    print("═" * 66)
    for name, rows in per_file.items():
        intents = sorted({r["intent"] for r in rows})
        print(f"  {name:<26} {len(rows):>4} προτάσεις  ({len(intents)} intents)")
    print("─" * 66)
    print(f"  ΣΥΝΟΛΟ: {len(merged)} προτάσεις σε {len(counts)} intents")
    if dupes:
        print(f"\n  ⚠  {len(dupes)} διπλότυπες προτάσεις:")
        for d in dupes[:5]:
            print(f"       {d[:60]}")
    else:
        print("  ✅ κανένα διπλότυπο")

    odd = {i: c for i, c in counts.items() if c != 100}
    if odd:
        print(f"\n  ⚠  intents χωρίς ακριβώς 100 προτάσεις:")
        for i, c in odd.items():
            print(f"       {c:>4}  {i}")

    print(f"\n  💾 {out_path}")
    print("═" * 66)


if __name__ == "__main__":
    main()
