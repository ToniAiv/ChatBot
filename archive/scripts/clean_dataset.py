"""
Dataset Cleaner — Expanded Intent Dataset
==========================================
Κανονικοποιεί τα labels του CSV εκπαίδευσης και βγάζει αναφορά με τα
προβλήματα που βρέθηκαν. ΔΕΝ πειράζει το αρχικό αρχείο.

Τι διορθώνει:
  1. Labels τυλιγμένα σε εισαγωγικά:  ' "βεβαίωση_κατοικίας"' → 'βεβαίωση_κατοικίας'
  2. Leading / trailing whitespace, tabs, backslashes
  3. Κενά / NaN intents  → πετιούνται
  4. Παύλες → underscores  (ώστε 'νέα-ανακοινώσεις' == 'νέα_ανακοινώσεις')
  5. Διπλά κενά μέσα στο text, duplicate (text, intent) ζεύγη

Χρήση:
    python3 clean_dataset.py --input Expanded_Intent_Dataset_2.csv
    python3 clean_dataset.py --input Expanded_Intent_Dataset_2.csv \
        --out Expanded_Intent_Dataset_clean.csv --min-samples 30
"""

import argparse
import csv
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# Κάτω από τόσα δείγματα, ένα intent θεωρείται υπο-εκπροσωπούμενο
DEFAULT_MIN_SAMPLES = 30


# ══════════════════════════════════════════════════════════════
# Canonicalisation
# ══════════════════════════════════════════════════════════════
def strip_accents(text: str) -> str:
    """Αφαιρεί τόνους — ίδια λογική με το connector.normalize()."""
    text = unicodedata.normalize("NFD", str(text))
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def canon_label(raw) -> str:
    """
    Το ΜΟΝΟ σημείο αλήθειας για το πώς γράφεται ένα intent.
    Πρέπει να χρησιμοποιείται πανομοιότυπα σε training, inference και mapping.
    """
    if raw is None:
        return ""
    s = str(raw)
    if s.strip().lower() in {"nan", "none", ""}:
        return ""
    s = s.replace("\t", " ").replace("\\", " ")
    s = s.strip().strip('"').strip("'").strip()   # ξετύλιγμα εισαγωγικών
    s = re.sub(r"[-\s]+", "_", s)                 # παύλες & κενά → underscore
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()


def canon_text(raw) -> str:
    return re.sub(r"\s+", " ", str(raw or "")).strip()


# ══════════════════════════════════════════════════════════════
# Cleaning
# ══════════════════════════════════════════════════════════════
def clean(rows: list) -> tuple:
    """Επιστρέφει (clean_rows, issues) όπου issues είναι dict με λίστες."""
    issues = defaultdict(list)
    seen: set = set()
    out: list = []

    for lineno, r in enumerate(rows, start=2):     # +2 = header + 1-indexed
        raw_text, raw_label = r.get("text"), r.get("intent")
        text, label = canon_text(raw_text), canon_label(raw_label)

        if not text:
            issues["empty_text"].append((lineno, raw_label))
            continue
        if not label:
            issues["empty_label"].append((lineno, raw_text))
            continue

        if str(raw_label).strip() != str(raw_label):
            issues["whitespace"].append((lineno, repr(raw_label)))
        if str(raw_label).strip().startswith('"') or str(raw_label).strip().endswith('"'):
            issues["quoted"].append((lineno, repr(raw_label)))
        if "\t" in str(raw_label) or "\\" in str(raw_label):
            issues["control_chars"].append((lineno, repr(raw_label)))

        key = (text.lower(), label)
        if key in seen:
            issues["duplicate"].append((lineno, text[:60]))
            continue
        seen.add(key)

        out.append({"text": text, "intent": label, "_line": lineno})

    # Ίδιο κείμενο με ΔΙΑΦΟΡΕΤΙΚΟ intent — αυτά μπερδεύουν ενεργά το μοντέλο
    by_text = defaultdict(list)
    for r in out:
        by_text[r["text"].lower()].append((r["intent"], r["_line"]))
    for text, entries in by_text.items():
        if len({lab for lab, _ in entries}) > 1:
            issues["conflicting"].append((text[:60], sorted(entries)))

    # Labels που διαφέρουν ΜΟΝΟ σε τόνους/ορθογραφία. Το μοντέλο εκπαιδεύεται
    # πάνω σε accent-stripped labels, οπότε αυτά συγχωνεύονται σιωπηλά — ο
    # άνθρωπος πρέπει να διαλέξει ποια γραφή είναι η σωστή.
    counts = Counter(r["intent"] for r in out)
    by_stripped = defaultdict(list)
    for label, n in counts.items():
        by_stripped[strip_accents(label)].append((label, n))
    for variants in by_stripped.values():
        if len(variants) > 1:
            issues["accent_collision"].append(sorted(variants, key=lambda x: -x[1]))

    return out, issues


# ══════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════
def report(rows_in: list, rows_out: list, issues: dict, min_samples: int) -> None:
    counts = Counter(r["intent"] for r in rows_out)

    print("═" * 66)
    print("  ΑΝΑΦΟΡΑ ΚΑΘΑΡΙΣΜΟΥ DATASET")
    print("═" * 66)
    print(f"  Γραμμές εισόδου      : {len(rows_in)}")
    print(f"  Γραμμές εξόδου       : {len(rows_out)}")
    print(f"  Μοναδικά intents     : {len(counts)}")

    raw_labels = {str(r.get('intent')) for r in rows_in if r.get('intent')}
    print(f"  Labels ΠΡΙΝ canon    : {len(raw_labels)}   ← το «φαινομενικό» πλήθος")
    print(f"  Labels ΜΕΤΑ canon    : {len(counts)}   ← το πραγματικό πλήθος")

    print(f"\n── Προβλήματα ─────────────────────────────────────────────────")
    labels_map = {
        "quoted":        "labels τυλιγμένα σε εισαγωγικά",
        "whitespace":    "labels με leading/trailing κενά",
        "control_chars": "labels με tab ή backslash",
        "empty_label":   "γραμμές με κενό intent (πετάχτηκαν)",
        "empty_text":    "γραμμές με κενό text (πετάχτηκαν)",
        "duplicate":     "ακριβή διπλότυπα (text, intent)",
        "conflicting":   "ΙΔΙΟ text με ΔΙΑΦΟΡΕΤΙΚΟ intent",
        "accent_collision": "labels που διαφέρουν μόνο σε τόνους",
    }
    for key, desc in labels_map.items():
        n = len(issues.get(key, []))
        mark = "  " if n == 0 else "⚠ "
        print(f"  {mark}{n:5}  {desc}")

    if issues.get("accent_collision"):
        print(f"\n── ⚠ Ορθογραφικά διπλότυπα labels ─────────────────────────────")
        print(f"  (το μοντέλο τα βλέπει ως ΕΝΑ — διάλεξε τη σωστή γραφή)")
        for variants in issues["accent_collision"]:
            keep, *drop = variants
            print(f"  κράτα:  {keep[0]}  ({keep[1]} δείγματα)")
            for d in drop:
                print(f"  σβήσε:  {d[0]}  ({d[1]} δείγματα)")

    if issues.get("conflicting"):
        print(f"\n── ⚠ Αντικρουόμενα δείγματα (μπερδεύουν το μοντέλο) ──────────")
        for text, entries in issues["conflicting"][:15]:
            print(f'  "{text}"')
            for label, line in entries:
                print(f"      γρ.{line:<6} → {label}")
        if len(issues["conflicting"]) > 15:
            print(f"  … και άλλα {len(issues['conflicting']) - 15}")

    weak = [(i, c) for i, c in counts.items() if c < min_samples]
    print(f"\n── Ισορροπία κλάσεων ──────────────────────────────────────────")
    print(f"  Μέγιστο  : {max(counts.values())} δείγματα")
    print(f"  Ελάχιστο : {min(counts.values())} δείγματα")
    print(f"  Διάμεσος : {sorted(counts.values())[len(counts) // 2]} δείγματα")
    if weak:
        print(f"\n  ⚠ {len(weak)} intents με < {min_samples} δείγματα "
              f"(χαμηλή αναμενόμενη ακρίβεια):")
        for intent, c in sorted(weak, key=lambda x: x[1]):
            print(f"      {c:4}  {intent}")
    print("═" * 66)


# ══════════════════════════════════════════════════════════════
# Export ανά γραμμή — για να μοιραστεί με την υπόλοιπη ομάδα
# ══════════════════════════════════════════════════════════════
ISSUE_DESC = {
    "quoted":        "label τυλιγμένο σε εισαγωγικά (κενό μετά το κόμμα)",
    "whitespace":    "label με leading/trailing κενό",
    "control_chars": "label με tab ή backslash",
    "empty_label":   "κενό intent — η γραμμή πετάχτηκε",
    "empty_text":    "κενό text — η γραμμή πετάχτηκε",
    "duplicate":     "ακριβές διπλότυπο (text, intent)",
}


def write_issues_csv(issues: dict, path: str) -> None:
    rows = []

    for key, desc in ISSUE_DESC.items():
        for lineno, detail in issues.get(key, []):
            rows.append({
                "γραμμή": lineno,
                "πρόβλημα": key,
                "περιγραφή": desc,
                "λεπτομέρεια": detail,
                "πρόταση": "διαγραφή" if key.startswith("empty") or key == "duplicate"
                           else "αυτόματη διόρθωση",
            })

    for text, entries in issues.get("conflicting", []):
        labels = sorted({lab for lab, _ in entries})
        for label, lineno in entries:
            rows.append({
                "γραμμή": lineno,
                "πρόβλημα": "conflicting",
                "περιγραφή": "ίδιο text σε πολλαπλά intents",
                "λεπτομέρεια": f'"{text}" → {label}',
                "πρόταση": f"κράτα ΕΝΑ από: {', '.join(labels)}",
            })

    for variants in issues.get("accent_collision", []):
        keep, *drop = variants
        for label, n in drop:
            rows.append({
                "γραμμή": "",
                "πρόβλημα": "accent_collision",
                "περιγραφή": "label διαφέρει μόνο σε τόνο",
                "λεπτομέρεια": f"{label} ({n} δείγματα)",
                "πρόταση": f"μετονόμασε σε: {keep[0]}",
            })

    rows.sort(key=lambda r: (r["γραμμή"] == "", r["γραμμή"]))

    fields = ["γραμμή", "πρόβλημα", "περιγραφή", "λεπτομέρεια", "πρόταση"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)


# ══════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description="Καθαρισμός intent dataset")
    ap.add_argument("--input", required=True, help="CSV εισόδου")
    ap.add_argument("--out", default=None,
                    help="CSV εξόδου (αν λείπει → μόνο αναφορά, δεν γράφεται τίποτα)")
    ap.add_argument("--labels-out", default=None,
                    help="Γράψε τη λίστα των canonical intents σε αρχείο κειμένου")
    ap.add_argument("--issues-out", default=None,
                    help="CSV με κάθε πρόβλημα και τον αριθμό γραμμής του")
    ap.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"Δεν βρέθηκε: {src}")

    with src.open(encoding="utf-8") as f:
        rows_in = list(csv.DictReader(f))

    rows_out, issues = clean(rows_in)
    report(rows_in, rows_out, issues, args.min_samples)

    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["text", "intent"],
                extrasaction="ignore",      # πετάει το εσωτερικό _line
                quoting=csv.QUOTE_ALL,
            )
            w.writeheader()
            w.writerows(rows_out)
        print(f"\n💾 Καθαρό dataset → {args.out}")

    if args.issues_out:
        write_issues_csv(issues, args.issues_out)
        print(f"💾 Αναφορά προβλημάτων ανά γραμμή → {args.issues_out}")

    if args.labels_out:
        labels = sorted({r["intent"] for r in rows_out})
        Path(args.labels_out).write_text("\n".join(labels) + "\n", encoding="utf-8")
        print(f"💾 {len(labels)} canonical intents → {args.labels_out}")


if __name__ == "__main__":
    main()
