"""Μετονομασία των 29 greeklish labels σε ελληνικά — μία φορά.

ΓΙΑΤΙ: ο TF-IDF του connector.py είναι char n-gram πάνω σε ελληνικούς
τίτλους KB. Ένα greeklish label δεν μοιράζεται κανέναν χαρακτήρα μαζί
τους, άρα score 0.000 και "δεν παρέχεται από τον Δήμο" — ακόμα κι όταν
το BERT έχει αναγνωρίσει σωστά με 0.97 βεβαιότητα.

ΔΕΝ ΧΡΕΙΑΖΕΤΑΙ ΕΠΑΝΕΚΠΑΙΔΕΥΣΗ: το μοντέλο βγάζει δείκτη κλάσης· το
όνομα είναι ετικέτα από πάνω. Αλλάζουμε την ετικέτα, τα βάρη μένουν.

Σύμβαση ονομάτων: άτονα, πεζά, κάτω παύλες — όπως τα άλλα 188 labels.
"""
import csv, glob, json, re, shutil, sys, unicodedata
from pathlib import Path

import joblib

from paths import ROOT, DATASETS, MAPPINGS, MODEL_218

RENAME = {
    "adeia_diakopis_kykloforias":             "αδεια_διακοπης_κυκλοφοριας",
    "adeia_dieleusis_vareon_oximaton":        "αδεια_διελευσης_βαρεων_οχηματων",
    "adeia_mousikon_idrymaton":               "αδεια_μουσικων_οργανων",
    "adeia_pezodromou":                       "αδεια_πεζοδρομου",
    "adeiodotisi_athlitikon_egkatastaseon":   "αδειοδοτηση_αθλητικων_εγκαταστασεων",
    "aitima_odokatharismou":                  "αιτημα_οδοκαθαρισμου",
    "aitisi_ergotherapeias":                  "αιτηση_εργοθεραπειας",
    "apomakrynsi_egkataleleimmenon_oximaton": "απομακρυνση_εγκαταλελειμμενων_οχηματων",
    "dikaiologitika_anelkystiron":            "δικαιολογητικα_ανελκυστηρων",
    "dilosi_diazygiou":                       "δηλωση_διαζυγιου",
    "dilosi_onomatodosias":                   "δηλωση_ονοματοδοσιας",
    "dilosi_symfonou_symviosis":              "δηλωση_συμφωνου_συμβιωσης",
    "dilosi_vaptisis":                        "δηλωση_βαπτισης",
    "ektelesi_ergon_odopoiias":               "εκτελεση_εργων_οδοποιιας",
    "enstasi_paranomis_stathmefsis":          "ενσταση_παρανομης_σταθμευσης",
    "enstasi_prostimou_katharaiotitas":       "ενσταση_προστιμου_καθαριοτητας",
    "epistrofi_pinakidon":                    "επιστροφη_πινακιδων",
    "kataggelia_egkataleleimmenon_ktirion":   "καταγγελια_εγκαταλελειμμενων_κτιριων",
    "kataggelia_empodion_odostromatos":       "καταγγελια_εμποδιων_οδοστρωματος",
    "kataggelia_laikon_agoron_emporiou":      "καταγγελια_λαικων_αγορων_εμποριου",
    "kataggelia_paraviasis_kok":              "καταγγελια_παραβιασης_κοκ",
    "katharaiotita_koinochristou_chorou":     "καθαριοτητα_κοινοχρηστου_χωρου",
    "metakinisi_topothetisi_kadon":           "μετακινηση_τοποθετηση_καδων",
    "monades_proscholikis_agogis":            "μοναδες_προσχολικης_αγωγης",
    "plirofories_laikon_agoron":              "πληροφοριες_λαικων_αγορων",
    "sima_stathmefsis_katoikou":              "σημα_σταθμευσης_κατοικου",
    "telesi_politikou_gamou":                 "τελεση_πολιτικου_γαμου",
    "tropopoiisi_tetragonikon_koinochristou": "τροποποιηση_τετραγωνικων_κοινοχρηστου",
    "vlavi_foteinon_simatodoton":             "βλαβη_φωτεινων_σηματοδοτων",
}


def key(s):
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
    return re.sub(r"[^a-zα-ω0-9]", "", s)


def main(apply: bool) -> None:
    enc = joblib.load(MODEL_218 / "label_encoder.joblib")
    classes = list(enc.classes_)

    missing = [g for g in RENAME if g not in classes]
    if missing:
        sys.exit(f"❌ Δεν υπάρχουν στο μοντέλο: {missing}")

    new_classes = [RENAME.get(c, c) for c in classes]

    # Σύγκρουση: το νέο όνομα να μη συμπίπτει με υπάρχον άλλο label.
    seen = {}
    for old, new in zip(classes, new_classes):
        k = key(new)
        if k in seen:
            sys.exit(f"❌ Σύγκρουση: '{old}' και '{seen[k]}' δίνουν και τα δύο '{new}'")
        seen[k] = old
    print(f"✓ Καμία σύγκρουση — {len(new_classes)} μοναδικά labels")

    if not apply:
        print("\n(dry-run — τίποτα δεν γράφτηκε. Ξανατρέξε με --apply)")
        for o, n in list(RENAME.items())[:5]:
            print(f"    {o}\n      → {n}")
        return

    # ── 1. Μοντέλο ────────────────────────────────────────────
    bak = MODEL_218 / "label_encoder.greeklish.joblib"
    if not bak.exists():
        shutil.copy2(MODEL_218 / "label_encoder.joblib", bak)
        print(f"✓ αντίγραφο ασφαλείας: {bak.name}")

    enc.classes_ = __import__("numpy").array(new_classes, dtype=object)
    joblib.dump(enc, MODEL_218 / "label_encoder.joblib")

    # Ρητή, ανεξάρτητη πηγή αλήθειας — δεν εξαρτάται από sklearn internals.
    (MODEL_218 / "labels.json").write_text(
        json.dumps(new_classes, ensure_ascii=False, indent=2), encoding="utf-8")
    print("✓ label_encoder.joblib + labels.json")

    cfg_path = MODEL_218 / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if "id2label" in cfg:
        cfg["id2label"] = {str(i): n for i, n in enumerate(new_classes)}
        cfg["label2id"] = {n: i for i, n in enumerate(new_classes)}
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        print("✓ config.json (id2label/label2id)")

    # ── 2. Δεδομένα ───────────────────────────────────────────
    targets = (
        glob.glob(str(DATASETS / "*.csv"))
        + glob.glob(str(DATASETS / "dataset_new" / "*.csv"))
        + glob.glob(str(DATASETS / "tests" / "*.csv"))
        + glob.glob(str(MAPPINGS / "*.csv"))
    )
    total = 0
    for path in sorted(targets):
        p = Path(path)
        rows = list(csv.DictReader(p.open(encoding="utf-8-sig")))
        if not rows:
            continue
        cols = [c for c in rows[0] if "intent" in c.lower()]
        if not cols:
            continue
        n = 0
        for r in rows:
            for c in cols:
                if r.get(c) in RENAME:
                    r[c] = RENAME[r[c]]; n += 1
        if n:
            with p.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), quoting=csv.QUOTE_MINIMAL)
                w.writeheader(); w.writerows(rows)
            print(f"    {n:5}  {p.relative_to(ROOT)}")
            total += n
    print(f"✓ {total} τιμές intent μετονομάστηκαν σε αρχεία δεδομένων")


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
