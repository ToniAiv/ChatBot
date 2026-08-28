"""
Intent → URL Mapping Builder
=============================
Παράγει ΠΡΟΤΕΙΝΟΜΕΝΟ mapping από κάθε intent στην αντίστοιχη σελίδα υπηρεσίας,
για ανθρώπινο έλεγχο. ΔΕΝ είναι τελικό — η στήλη `status` δείχνει πού χρειάζεται
προσοχή.

Γιατί υπάρχει:
  Το taxonomy είναι κλειστό και μικρό (~101 intents / ~167 URLs). Αν το mapping
  είναι σωστό, τότε σωστό intent ⇒ 100% σωστό URL, ντετερμινιστικά — χωρίς να
  εξαρτόμαστε από retrieval στο runtime.

Δεύτερη χρήση — gap analysis:
  * intents ΧΩΡΙΣ αντίστοιχο URL  → υποψήφια προς αφαίρεση (ο Δήμος δεν την παρέχει)
  * URLs ΧΩΡΙΣ αντίστοιχο intent  → υπηρεσίες που λείπουν από το dataset

Ο συνδυασμός δίνει σήματα από δύο ανεξάρτητες πηγές:
  1. Sentence embeddings (σημασιολογική ομοιότητα)
  2. TF-IDF char n-grams (ανθεκτικό σε καταλήξεις/κλίση ελληνικών)

Χρήση:
    python3 build_intent_map.py \
        --dataset Expanded_Intent_Dataset_2.csv \
        --kb data/heraklion_eservices.json \
        --out intent_map_draft.csv
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from clean_dataset import canon_label, canon_text, strip_accents
from paths import ROOT, DATA, DATASETS, MODELS, MAPPINGS, MODEL_218, KB, KB_RICH

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Τίτλοι-σκουπίδια που δεν αντιστοιχούν σε πραγματική υπηρεσία
EXCLUDED_TITLES = {"Ο ΛΟΓΑΡΙΑΣΜΟΣ ΜΟΥ", ""}

# Πόσοι χαρακτήρες body λαμβάνονται υπόψη ανά υπηρεσία
BODY_CHARS = 400

# Κατώφλια για τον χαρακτηρισμό της πρότασης
AUTO_SCORE = 0.60     # πάνω από αυτό + καθαρή διαφορά → μάλλον σωστό
AUTO_MARGIN = 0.12    # διαφορά top1 από top2
WEAK_SCORE = 0.35     # κάτω από αυτό → μάλλον δεν υπάρχει αντίστοιχη υπηρεσία


def norm(text: str) -> str:
    """Ίδια κανονικοποίηση με το connector/eval."""
    return strip_accents(str(text)).lower().strip()


# ══════════════════════════════════════════════════════════════
# Φόρτωση
# ══════════════════════════════════════════════════════════════
def load_intents(path: Path) -> dict:
    """Επιστρέφει {intent: [δείγματα ερωτήσεων]}."""
    by_intent = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            label = canon_label(r.get("intent"))
            text = canon_text(r.get("text"))
            if label and text:
                by_intent[label].append(text)
    return dict(by_intent)


def load_kb(path: Path) -> list:
    with path.open(encoding="utf-8") as f:
        kb = json.load(f)
    return [
        r for r in kb
        if r.get("title") not in EXCLUDED_TITLES and r.get("language", "el") == "el"
    ]


# ══════════════════════════════════════════════════════════════
# Scoring
# ══════════════════════════════════════════════════════════════
def build_queries(by_intent: dict) -> tuple:
    """
    Το query κάθε intent είναι ΜΟΝΟ το label του.

    Πρώτη εκδοχή ένωνε label + 8 δείγματα ερωτήσεων. Αποτύχε: τα δείγματα είναι
    σε μεγάλο βαθμό boilerplate («τι δικαιολογητικά χρειάζομαι;», «πόσο
    κοστίζει;») κοινό σε όλα τα intents, οπότε κυριαρχούσε στο embedding και
    έσβηνε το διακριτικό σήμα. Αποτέλεσμα: γενικές διοικητικές σελίδες
    (π.χ. «Δημοτική Ενημερότητα») γίνονταν μαγνήτης για τα πάντα.
    """
    intents = sorted(by_intent)
    queries = [norm(i.replace("_", " ")) for i in intents]
    return intents, queries


def score_embeddings(queries: list, titles: list, bodies: list) -> np.ndarray:
    """
    Το intent συγκρίνεται ξεχωριστά με τον τίτλο ΚΑΙ με το σώμα κειμένου, και
    κρατάμε το μέγιστο: ταίριασμα σε οποιοδήποτε από τα δύο είναι ένδειξη.
    Η ένωσή τους σε ένα κείμενο θα αραίωνε πάλι το σήμα του τίτλου.
    """
    from sentence_transformers import SentenceTransformer

    print(f"🔄 Φόρτωση {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)
    print(f"   Encoding {len(queries)} intents + {len(titles)} υπηρεσίες...")

    q = model.encode(queries, show_progress_bar=False, normalize_embeddings=True)
    t = model.encode(titles, show_progress_bar=False, normalize_embeddings=True)
    sim = cosine_similarity(q, t)

    if any(bodies):
        b = model.encode(
            [x[:BODY_CHARS] or " " for x in bodies],
            show_progress_bar=False, normalize_embeddings=True,
        )
        sim = np.maximum(sim, cosine_similarity(q, b))
    return sim


def score_classifier(intents: list, titles: list, bodies: list) -> np.ndarray:
    """
    Αντιστρέφει το πρόβλημα: αντί να ψάχνουμε ποια υπηρεσία μοιάζει με το intent,
    ρωτάμε τον ΙΔΙΟ τον ταξινομητή «σε ποιο intent ανήκει αυτή η σελίδα;».

    Πλεονέκτημα έναντι των generic embeddings: το μοντέλο είναι fine-tuned πάνω
    σε ελληνική διοικητική ορολογία δήμου, οπότε ξέρει ότι «ΤΑΠ», «ΔΕΥΑΗ» και
    «ληξιαρχική πράξη» είναι διακριτές έννοιες — κάτι που το multilingual
    MiniLM αγνοεί.

    Επιστρέφει πίνακα (n_intents × n_services) με πιθανότητες.
    """
    import joblib
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = MODEL_218
    if not model_dir.exists():
        print("  ⚠  Δεν βρέθηκε dimos-intent-model — παραλείπω το σήμα classifier")
        return np.zeros((len(intents), len(titles)))

    print("🔄 Φόρτωση intent classifier...")
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    classes = list(joblib.load(model_dir / "label_encoder.joblib").classes_)

    # Το Greek BERT είναι uncased: χωρίς norm() τα tokens γίνονται [UNK]
    docs = [norm(f"{t} {b[:BODY_CHARS]}") for t, b in zip(titles, bodies)]

    probs = []
    for start in range(0, len(docs), 16):
        enc = tok(docs[start:start + 16], return_tensors="pt",
                  truncation=True, max_length=128, padding=True)
        with torch.no_grad():
            logits = model(**enc).logits
        probs.append(torch.softmax(logits, dim=-1).numpy())
    probs = np.vstack(probs)                      # (n_services, n_classes)

    # Αναδιάταξη στηλών ώστε να ταιριάζουν με τη σειρά των intents μας
    col = {c: i for i, c in enumerate(classes)}
    out = np.zeros((len(intents), len(titles)))
    for i, intent in enumerate(intents):
        j = col.get(norm(intent))
        if j is not None:
            out[i] = probs[:, j]
    return out


def score_tfidf(queries: list, docs: list) -> np.ndarray:
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    matrix = vec.fit_transform([norm(d) for d in docs])
    return cosine_similarity(vec.transform(queries), matrix)


def classify(top1: float, top2: float) -> str:
    if top1 < WEAK_SCORE:
        return "ΚΑΜΙΑ_ΥΠΗΡΕΣΙΑ"
    if top1 >= AUTO_SCORE and (top1 - top2) >= AUTO_MARGIN:
        return "ok"
    return "ΕΛΕΓΞΕ"


# ══════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description="Παραγωγή draft intent→URL mapping")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--kb", default=str(KB))
    ap.add_argument("--out", default=str(MAPPINGS / "intent_map_draft.csv"))
    ap.add_argument("--gaps-out", default=str(MAPPINGS / "intent_map_gaps.txt"))
    ap.add_argument("--tfidf-weight", type=float, default=0.4,
                    help="Βάρος TF-IDF στο τελικό score (υπόλοιπο = embeddings)")
    ap.add_argument("--clf-weight", type=float, default=0.5,
                    help="Βάρος του intent classifier ως bonus (0 = απενεργοποίηση)")
    args = ap.parse_args()

    ds_path, kb_path = Path(args.dataset), Path(args.kb)
    for p in (ds_path, kb_path):
        if not p.exists():
            sys.exit(f"Δεν βρέθηκε: {p}")

    by_intent = load_intents(ds_path)
    kb = load_kb(kb_path)
    titles = [r["title"] for r in kb]
    print(f"📋 {len(by_intent)} intents  ×  {len(kb)} υπηρεσίες\n")

    intents, queries = build_queries(by_intent)
    bodies = [r.get("body", "") for r in kb]
    if not any(bodies):
        print("  ⚠  Το KB δεν έχει πεδίο `body` — τρέξε πρώτα enrich_kb.py\n")

    emb = score_embeddings(queries, titles, bodies)
    tfidf = score_tfidf(
        queries, [f"{t} {b[:BODY_CHARS]}" for t, b in zip(titles, bodies)]
    )
    clf = score_classifier(intents, titles, bodies)

    w = args.tfidf_weight
    combined = (1 - w) * emb + w * tfidf
    # Ο classifier είναι το ισχυρότερο σήμα όπου είναι σίγουρος, αλλά σιωπά
    # (τιμές ~0) στις περισσότερες σελίδες — γι' αυτό μπαίνει ως bonus πάνω
    # στο βασικό score αντί να αντικαθιστά τα άλλα δύο.
    combined = combined + args.clf_weight * clf

    # ── Draft mapping ────────────────────────────────────────────────────────
    rows = []
    claimed = set()
    for i, intent in enumerate(intents):
        order = np.argsort(combined[i])[::-1][:3]
        s1, s2 = combined[i][order[0]], combined[i][order[1]]
        best = kb[order[0]]
        status = classify(s1, s2)
        if status != "ΚΑΜΙΑ_ΥΠΗΡΕΣΙΑ":
            claimed.add(best["url"])

        rows.append({
            "intent": intent,
            "δείγματα": len(by_intent[intent]),
            "status": status,
            "score": f"{s1:.3f}",
            "url": best["url"] if status != "ΚΑΜΙΑ_ΥΠΗΡΕΣΙΑ" else "",
            "τίτλος": best["title"],
            "εναλλακτική_2": kb[order[1]]["title"],
            "εναλλακτική_3": kb[order[2]]["title"],
        })

    fields = ["intent", "δείγματα", "status", "score", "url", "τίτλος",
              "εναλλακτική_2", "εναλλακτική_3"]
    order_rank = {"ΚΑΜΙΑ_ΥΠΗΡΕΣΙΑ": 0, "ΕΛΕΓΞΕ": 1, "ok": 2}
    rows.sort(key=lambda r: (order_rank[r["status"]], -float(r["score"])))

    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        wcsv.writeheader()
        wcsv.writerows(rows)

    # ── Gap analysis ─────────────────────────────────────────────────────────
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in order_rank}
    orphan_urls = [r for r in kb if r["url"] not in claimed]

    lines = []
    lines.append("ΥΠΗΡΕΣΙΕΣ ΧΩΡΙΣ INTENT — υποψήφιες προς ΠΡΟΣΘΗΚΗ στο dataset")
    lines.append("=" * 66)
    for r in sorted(orphan_urls, key=lambda x: x.get("category", "")):
        cat = r.get("category") or "—"
        lines.append(f"[{cat}] {r['title']}")
        lines.append(f"    {r['url']}")
    lines.append("")
    lines.append("INTENTS ΧΩΡΙΣ ΥΠΗΡΕΣΙΑ — υποψήφια προς ΑΦΑΙΡΕΣΗ από το dataset")
    lines.append("=" * 66)
    for r in rows:
        if r["status"] == "ΚΑΜΙΑ_ΥΠΗΡΕΣΙΑ":
            lines.append(f"{r['intent']}  ({r['δείγματα']} δείγματα, "
                         f"καλύτερο score {r['score']})")
            lines.append(f"    πλησιέστερο: {r['τίτλος']}")
    Path(args.gaps_out).write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 66)
    print("  DRAFT INTENT → URL MAPPING")
    print("═" * 66)
    print(f"  ✅ ok               : {counts['ok']:3}  υψηλή εμπιστοσύνη")
    print(f"  🔍 ΕΛΕΓΞΕ           : {counts['ΕΛΕΓΞΕ']:3}  θέλουν ανθρώπινη επιβεβαίωση")
    print(f"  ❌ ΚΑΜΙΑ_ΥΠΗΡΕΣΙΑ   : {counts['ΚΑΜΙΑ_ΥΠΗΡΕΣΙΑ']:3}  πιθανώς δεν παρέχεται")
    print(f"\n  🔗 Υπηρεσίες χωρίς intent : {len(orphan_urls)} / {len(kb)}")
    print(f"\n  💾 {args.out}")
    print(f"  💾 {args.gaps_out}")
    print("═" * 66)
    print("\n  ⚠  ΤΙΠΟΤΑ ΑΠΟ ΑΥΤΑ ΔΕΝ ΕΙΝΑΙ ΤΕΛΙΚΟ — είναι προτάσεις για έλεγχο.")


if __name__ == "__main__":
    main()
