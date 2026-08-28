"""
Benchmark: TF-IDF char ngram vs Sentence Embeddings
====================================================
Συγκρίνει τις δύο προσεγγίσεις για τα problematic intents.

Χρήση:
    pip install sentence-transformers
    python3 benchmark_embeddings.py --json data/heraklion_eservices.json
"""

import argparse
import json
import unicodedata

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text))
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower().strip()


EXCLUDED = {"Ο ΛΟΓΑΡΙΑΣΜΟΣ ΜΟΥ", ""}

# Intents που ξέρουμε ότι έχουν προβλήματα με TF-IDF
# format: (intent, λέξη-κλειδί που πρέπει να υπάρχει στο σωστό τίτλο)
PROBLEMATIC = [
    ("επεκταση_δικτυου_υδρευσης",       "δευαη"),
    ("αγορα_οικογενειακου_ταφου",        "ταφ"),
    ("αιτηση_κιτρινης_διαγραμμισης",     "διαγραμμ"),
    ("ρυθμιση_οφειλων",                  "οφειλ"),
    ("αδεια_κολυμβητικης_δεξαμενης",     "κολυμβ"),
    ("ενοικιαση_ταφου",                  "ταφ"),
    ("παραταση_μουσικης",                "μουσ"),
    ("συμμετοχη_εμποροπανηγυρη",         "εμπορ"),
    ("στεγαστρο_οασα",                   "στεγαστρ"),
    ("μειωση_τελων_λογω_αναπηριας",      "αναπηρ"),
]

# Και μερικά που δουλεύουν καλά — για να επαληθεύσουμε ότι δεν χαλάμε τίποτα
WORKING = [
    ("αδεσποτα_ζωα",                    "αδεσποτ"),
    ("πιστοποιητικο_γεννησης",           "γεννησ"),
    ("δημοτικη_ενημεροτητα",             "ενημεροτητ"),
    ("ασφαλτοστρωση_οδου",               "ασφαλτ"),
    ("βεβαιωση_κατοικιας",               "κατοικ"),
    ("ληξιαρχικη_πραξη_θανατου",         "θανατ"),
]


def run_tfidf(valid_kb, test_cases):
    titles_norm = [normalize(r["title"]) for r in valid_kb]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    matrix = vec.fit_transform(titles_norm)

    results = []
    for intent, keyword in test_cases:
        q = vec.transform([normalize(intent.replace("_", " "))])
        scores = cosine_similarity(q, matrix)[0]
        top3 = np.argsort(scores)[::-1][:3]
        hits = [(scores[i], valid_kb[i]["title"]) for i in top3]
        correct = any(keyword in normalize(t) for _, t in hits[:1])
        results.append((intent, keyword, hits, correct))
    return results


def run_embeddings(valid_kb, test_cases):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ sentence-transformers δεν είναι εγκατεστημένο.")
        print("   pip install sentence-transformers")
        return None

    print("🔄 Φόρτωση paraphrase-multilingual-MiniLM-L12-v2...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print("✅ Μοντέλο έτοιμο.")

    titles = [r["title"] for r in valid_kb]
    print(f"   Encoding {len(titles)} τίτλους...")
    title_embeddings = model.encode(titles, show_progress_bar=False)

    results = []
    for intent, keyword in test_cases:
        query = intent.replace("_", " ")
        q_emb = model.encode([query])
        scores = cosine_similarity(q_emb, title_embeddings)[0]
        top3 = np.argsort(scores)[::-1][:3]
        hits = [(scores[i], valid_kb[i]["title"]) for i in top3]
        correct = any(keyword in normalize(t) for _, t in hits[:1])
        results.append((intent, keyword, hits, correct))
    return results


def print_comparison(label, results):
    correct = sum(1 for _, _, _, c in results if c)
    total   = len(results)
    print(f"\n{'═'*65}")
    print(f"  {label}  —  {correct}/{total} σωστά")
    print(f"{'═'*65}")
    for intent, keyword, hits, correct in results:
        mark = "✅" if correct else "❌"
        print(f"\n  {mark} {intent}")
        for score, title in hits[:3]:
            dot = "●" if keyword in normalize(title) else " "
            print(f"      {dot} {score:.3f}  {title[:55]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="data/heraklion_eservices.json")
    args = parser.parse_args()

    with open(args.json, encoding="utf-8") as f:
        kb = json.load(f)
    valid_kb = [r for r in kb if r["title"] not in EXCLUDED]

    all_cases = PROBLEMATIC + WORKING
    print(f"📊 Benchmark: {len(PROBLEMATIC)} problematic + {len(WORKING)} working intents\n")

    # TF-IDF
    tfidf_results = run_tfidf(valid_kb, all_cases)
    print_comparison("TF-IDF char ngram (3-5) — ΤΩΡΙΝΟ", tfidf_results)

    # Sentence Embeddings
    emb_results = run_embeddings(valid_kb, all_cases)
    if emb_results:
        print_comparison("Sentence Embeddings (multilingual-MiniLM)", emb_results)

        # Σύγκριση
        tfidf_score = sum(1 for _, _, _, c in tfidf_results if c)
        emb_score   = sum(1 for _, _, _, c in emb_results   if c)
        print(f"\n{'─'*65}")
        print(f"  ΣΥΓΚΡΙΣΗ ΣΥΝΟΛΙΚΑ")
        print(f"  TF-IDF            : {tfidf_score}/{len(all_cases)}")
        print(f"  Sentence Embeddings: {emb_score}/{len(all_cases)}")
        diff = emb_score - tfidf_score
        if diff > 0:
            print(f"  → Βελτίωση: +{diff} σωστά matches")
        elif diff == 0:
            print(f"  → Ισοβαθμία")
        else:
            print(f"  → Χειρότερο: {diff}")
        print(f"{'─'*65}")


if __name__ == "__main__":
    main()
