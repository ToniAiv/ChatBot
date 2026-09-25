"""
Προβολή των ενοτήτων που εξήχθησαν από τις σελίδες υπηρεσιών.

Για έλεγχο με το μάτι: βάλε μια λέξη, δες τι βγήκε, και σύγκρινέ το
με τη σελίδα του δήμου (ο σύνδεσμος τυπώνεται).

Χρήση:
    python3 scripts/show_sections.py κατοικίας
    python3 scripts/show_sections.py "θέση στάθμευσης"
    python3 scripts/show_sections.py --missing      # όσες ΔΕΝ έχουν δικαιολογητικά
"""
import json
import re
import sys
import unicodedata

from paths import DATA

S = json.load(open(DATA / "service_sections.json", encoding="utf-8"))
DOCS = re.compile(r"^(τι χρειάζεται|τι θα χρειαστείτε|απαιτούμενα δικαιολογητικά|δικαιολογητικά)", re.I)


def norm(s):
    s = unicodedata.normalize("NFD", str(s))
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()


def show(url, rec):
    print("═" * 72)
    print(f"  {rec['title']}")
    print(f"  {url}")
    print("═" * 72)
    for head, text in rec["sections"].items():
        mark = "  ← ΔΙΚΑΙΟΛΟΓΗΤΙΚΑ" if DOCS.match(head) else ""
        print(f"\n  ▸ {head}{mark}")
        for line in text.split("\n"):
            print(f"      {line}")
    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    if sys.argv[1] == "--missing":
        miss = [(u, r) for u, r in S.items() if not any(DOCS.match(h) for h in r["sections"])]
        print(f"\n  {len(miss)} υπηρεσίες ΧΩΡΙΣ ενότητα δικαιολογητικών:\n")
        for u, r in sorted(miss, key=lambda x: x[1]["title"]):
            print(f"    · {r['title'][:70]}")
        return
    q = norm(" ".join(sys.argv[1:]))
    hits = [(u, r) for u, r in S.items() if q in norm(r["title"])]
    if not hits:
        print(f"  Καμία υπηρεσία με «{' '.join(sys.argv[1:])}» στον τίτλο."); return
    for u, r in hits[:5]:
        show(u, r)
    if len(hits) > 5:
        print(f"  … και άλλες {len(hits)-5}. Γίνε πιο συγκεκριμένος.")


if __name__ == "__main__":
    main()
