"""Χτίζει τον πίνακα intent → υπηρεσία που χρησιμοποιεί ο connector.

Σχεδιαστική αρχή: ο πίνακας ΔΕΝ πρέπει ποτέ να είναι χειρότερος από τον
TF-IDF. Γι' αυτό κάθε γραμμή έχει status:

  verified  — άνθρωπος το επιβεβαίωσε              → χρησιμοποιείται
  auto      — η τρέχουσα επιλογή του TF-IDF,
              στέρεη αλλά ανεπιβεβαίωτη            → χρησιμοποιείται
  review    — ύποπτη· χρειάζεται ανθρώπινο μάτι    → ΕΦΕΔΡΕΙΑ στον TF-IDF

Έτσι σήμερα η συμπεριφορά είναι πανομοιότυπη με πριν, και βελτιώνεται
σταδιακά όσο γυρίζουν γραμμές σε verified.
"""
import csv, json, re, sys, unicodedata
from pathlib import Path

import joblib
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import connector as C
from paths import ROOT, MAPPINGS, KB

MIN_SOLID_SCORE  = 0.45
MIN_SOLID_MARGIN = 0.10


def key(s):
    """Κλειδί σύγκρισης labels: τα χειροφτιαγμένα αρχεία έχουν τόνους,
    το μοντέλο όχι. Χωρίς αυτό, καμία από τις 27 επιβεβαιωμένες
    αντιστοιχίσεις δεν κουμπώνει."""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
    return re.sub(r"[^a-zα-ω0-9]", "", s)


def norm(t):
    t = unicodedata.normalize("NFD", str(t))
    t = "".join(c for c in t if unicodedata.category(c) != "Mn").lower()
    return re.sub(r"[^\w\s]", " ", re.sub(r"\s+", " ", t)).strip()


def titles_match(a, b):
    na, nb = norm(a), norm(b)
    if not na or not nb: return False
    if na == nb or na in nb or nb in na: return True
    wa = {w for w in na.split() if len(w) >= 4}; wb = {w for w in nb.split() if len(w) >= 4}
    return bool(wa and wb) and len(wa & wb) / min(len(wa), len(wb)) >= 0.7


def main():
    kb = json.load(open(KB, encoding="utf-8"))
    vec, mat, valid = C.build_tfidf_index(kb)
    classes = list(joblib.load(C.MODEL_DIR / "label_encoder.joblib").classes_)

    # Τα 27 χειροφτιαγμένα είναι ground truth — μπαίνουν ως verified.
    human = {}
    for r in csv.DictReader((MAPPINGS / "intent_map.csv").open(encoding="utf-8-sig")):
        # ΠΡΩΤΑ ακριβής σύγκριση. Η ασαφής μόνο ως έσχατη λύση: το
        # «Γενική αίτηση για θέματα Κ.Ο.Κ» ταίριαζε με το «Γενική Αίτηση
        # για θέματα Επιδομάτων» επειδή μετά την αφαίρεση στίξης
        # μοιράζονται και τις τρεις λέξεις ≥4 γραμμάτων — και έτσι
        # αλλοιώθηκαν 4 ανθρώπινες αντιστοιχίσεις χωρίς προειδοποίηση.
        rec = next((k for k in valid if norm(k["title"]) == norm(r["service_title"])), None)
        if rec is None:
            fuzzy = [k for k in valid if titles_match(k["title"], r["service_title"])]
            if len(fuzzy) == 1:
                rec = fuzzy[0]
            elif len(fuzzy) > 1:
                print(f"  ⚠ διφορούμενο: «{r['service_title'][:50]}» → {len(fuzzy)} υποψήφιες, παραλείπεται")
        if rec:
            human[key(r["intent"])] = rec

    rows, counts = [], {"verified": 0, "auto": 0, "review": 0}
    for intent in classes:
        cands = C.tfidf_search(intent, vec, mat, valid)
        s1, s2 = cands[0][0], cands[1][0]
        rec = cands[0][1]

        if key(intent) in human:
            status, rec = "verified", human[key(intent)]
        elif s1 >= MIN_SOLID_SCORE and (s1 - s2) >= MIN_SOLID_MARGIN:
            status = "auto"
        else:
            status = "review"

        counts[status] += 1
        rows.append({
            "intent": intent,
            "status": status,
            "title": rec["title"] if status != "review" else "",
            "url": rec["url"] if status != "review" else "",
            "tfidf_score": f"{s1:.3f}",
            "tfidf_proposal": cands[0][1]["title"],
            "tfidf_alt": cands[1][1]["title"],
            "note": "",
        })

    out = MAPPINGS / "intent_to_service.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        # review πρώτα — εκεί είναι η δουλειά
        order = {"review": 0, "auto": 1, "verified": 2}
        w.writerows(sorted(rows, key=lambda r: (order[r["status"]], float(r["tfidf_score"]))))

    print(f"\n  verified : {counts['verified']:3}   (τα 27 χειροφτιαγμένα)")
    print(f"  auto     : {counts['auto']:3}   (στέρεος TF-IDF, ανεπιβεβαίωτο)")
    print(f"  review   : {counts['review']:3}   ← εδώ είναι η δουλειά, πέφτουν σε εφεδρεία")
    print(f"\n  💾 {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
