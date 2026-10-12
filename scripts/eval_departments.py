"""
Αναγνώριση τμήματος σε ερωτήσεις επικοινωνίας (app/departments.py).

Για κάθε ερώτηση του datasets/department_eval.csv βλέπουμε ΠΟΥ καταλήγει η
απάντηση: σε τμήμα (κωδικός), σε υπηρεσία («service»: σύνδεσμος ή τα
στοιχεία της ίδιας της υπηρεσίας), ή πουθενά («none»: πρόταση/άρνηση).
Το σετ έχει και παγίδες όπου το σωστό ΔΕΝ είναι τμήμα, π.χ. «πού βρίσκεται
το δημαρχείο» ή «τηλέφωνο για τη βεβαίωση κατοικίας».

    python3 scripts/eval_departments.py            # πριν/μετά
"""
import csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import departments
from paths import DATASETS
from regression import Bot

EVAL = DATASETS / "department_eval.csv"


def where(bot, q):
    it, cf, out = bot.ask(q, translate=False)
    if bot.last_dept:
        return bot.last_dept, cf
    if out == "phone":
        return (bot.ns.get(it) or {}).get("code", "?"), cf
    if out == "smalltalk":
        return "KENTRO", cf          # το «ποιος είσαι» δίνει τηλέφωνο του δήμου
    return ("service" if out == "link" else "none"), cf


def evaluate(bot, verbose=False):
    rows = list(csv.DictReader(open(EVAL, encoding="utf-8")))
    ok = 0
    for r in rows:
        got, cf = where(bot, r["text"])
        good = got in r["expect"].split("|")
        ok += good
        if verbose:
            print(f"  {'✓' if good else '✗'} {got:18} {cf:4.2f}  {r['text'][:58]:60} (αναμενόμενο {r['expect']})")
    return ok / len(rows), len(rows)


if __name__ == "__main__":
    bot = Bot()
    departments.ENABLED = False
    before, n = evaluate(bot)
    departments.ENABLED = True
    after, _ = evaluate(bot, verbose=True)
    print(f"\n  σωστός προορισμός: χωρίς αναγνώριση τμήματος {before:.0%} → με {after:.0%}  (n={n})")
