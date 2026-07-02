"""
Connector Script — Δήμος Ηρακλείου e-Services  v5.0
====================================================
Pipeline:
    Ερώτηση πολίτη
        → Greek BERT    (intent detection)
        → TF-IDF search (intent label → top-K τίτλοι JSON)
        → Llama 3.1     (intent + candidates → απάντηση)

Χρήση:
    python3 connector.py                  # interactive mode
    python3 connector.py --query "..."    # single query
"""

import argparse
import difflib
import json
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import requests
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Paths ─────────────────────────────────────────────────────
MODEL_DIR    = Path("dimos-intent-model")
JSON_PATH    = Path("data/heraklion_eservices.json")
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"

# ── Ρυθμίσεις ─────────────────────────────────────────────────
MIN_BERT_CONFIDENCE = 0.20   # κάτω από αυτό → άγνωστη ερώτηση
MIN_TFIDF_SCORE     = 0.25   # κάτω από αυτό → out-of-scope
TOP_K               = 5      # πόσα candidates στέλνουμε στο Llama

# Τίτλοι που αποκλείονται από την αναζήτηση (template strings)
EXCLUDED_TITLES = {"Ο ΛΟΓΑΡΙΑΣΜΟΣ ΜΟΥ", ""}


# ══════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════
def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text))
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower().strip()


# ══════════════════════════════════════════════════════════════
# Φόρτωση BERT
# ══════════════════════════════════════════════════════════════
def load_bert():
    print("🔄 Φόρτωση Greek BERT μοντέλου...")
    tokenizer     = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model         = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    print("✅ Μοντέλο έτοιμο.")
    return tokenizer, model, label_encoder


# ══════════════════════════════════════════════════════════════
# TF-IDF Index
# ══════════════════════════════════════════════════════════════
def build_tfidf_index(knowledge_base: list):
    """Φτιάχνει TF-IDF index από τους τίτλους του JSON."""
    # Φιλτράρισμα εγγραφών με template titles
    valid_kb = [r for r in knowledge_base if r["title"] not in EXCLUDED_TITLES]

    titles_norm = [normalize(r["title"]) for r in valid_kb]

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    tfidf_matrix = vectorizer.fit_transform(titles_norm)

    print(f"✅ TF-IDF index: {len(valid_kb)} υπηρεσίες.")
    return vectorizer, tfidf_matrix, valid_kb


def tfidf_search(intent: str, vectorizer, tfidf_matrix, valid_kb, top_k: int = TOP_K):
    """Βρίσκει τα top-K πιο σχετικά records με βάση cosine similarity."""
    query = normalize(intent.replace("_", " "))
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix)[0]

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = [(scores[i], valid_kb[i]) for i in top_indices]
    return results  # list of (score, record)


# ══════════════════════════════════════════════════════════════
# Intent Detection
# ══════════════════════════════════════════════════════════════
def detect_intent(query, tokenizer, model, label_encoder):
    inputs = tokenizer(
        normalize(query),
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    top_indices = torch.argsort(probs, descending=True)[:5]
    return [(label_encoder.classes_[i.item()], probs[i].item()) for i in top_indices]


# ══════════════════════════════════════════════════════════════
# Llama
# ══════════════════════════════════════════════════════════════
_URL_RE = re.compile(r"https?://[^\s<>\"')]+")


def _sanitize_urls(response: str, allowed_urls: list) -> str:
    """
    Αντικαθιστά κάθε URL που δεν υπάρχει στο allowed-set με το πιο κοντινό match.
    Αν δεν υπάρχει κοντινό (ratio < 0.6), το αφαιρεί εντελώς.
    """
    allowed_set = set(allowed_urls)

    def _fix(match: re.Match) -> str:
        url = match.group(0).rstrip(".,);:]")
        trailing = match.group(0)[len(url):]
        if url in allowed_set:
            return url + trailing
        # Βρες πιο κοντινό
        close = difflib.get_close_matches(url, allowed_urls, n=1, cutoff=0.6)
        if close:
            return close[0] + trailing
        return ""  # drop hallucinated URL

    return _URL_RE.sub(_fix, response)


def ask_llama(query: str, intent: str, candidates: list) -> str:
    records = [r for _, r in candidates]
    services_enum = "\n".join(
        f"{i+1}. {r['title']}"
        for i, r in enumerate(records)
    )
    n = len(records)

    prompt = f"""Ένας πολίτης του Δήμου Ηρακλείου ρώτησε: "{query}"

Το σύστημα αναγνώρισε ότι αναζητά: {intent.replace('_', ' ')}

Διαθέσιμες υπηρεσίες:
{services_enum}

Επίλεξε ποια υπηρεσία ταιριάζει καλύτερα (αριθμός 1-{n}), ή 0 αν καμία δεν ταιριάζει.
Απάντησε ΜΟΝΟ με valid JSON στη μορφή:
{{"choice": <αριθμός>, "answer": "<2-3 προτάσεις στα ελληνικά προς τον πολίτη, ΧΩΡΙΣ URLs ή links>"}}

JSON:"""

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1, "num_predict": 300},
            },
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json()["response"].strip()

        try:
            parsed = json.loads(raw)
            choice = int(parsed.get("choice", 0))
            answer = str(parsed.get("answer", "")).strip()
        except (json.JSONDecodeError, ValueError, TypeError):
            # fallback: καθάρισε hallucinated URLs και δώσε ό,τι έβγαλε
            allowed = [r["url"] for r in records]
            return _sanitize_urls(raw, allowed)

        # Strip τυχόν URLs που ξέφυγαν στο answer
        answer = _URL_RE.sub("", answer).strip()

        if 1 <= choice <= n:
            chosen = records[choice - 1]
            return f"{answer}\n\n{chosen['url']}"
        else:
            return (
                "Η υπηρεσία δεν φαίνεται να παρέχεται ηλεκτρονικά από τον Δήμο. "
                "Παρακαλώ επικοινωνήστε στο 2813 409185."
            )
    except requests.exceptions.ConnectionError:
        return "❌ Σφάλμα: Το Ollama δεν τρέχει. Ξεκίνα το με: ollama serve"
    except Exception as e:
        return f"❌ Σφάλμα: {e}"


# ══════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════
def run(query, tokenizer, model, label_encoder, vectorizer, tfidf_matrix, valid_kb):
    print(f"\n{'─'*55}")
    print(f"  Ερώτηση: {query}")
    print(f"{'─'*55}")

    # Step 1: BERT intent
    intents = detect_intent(query, tokenizer, model, label_encoder)
    top_intent, top_score = intents[0]

    print(f"\n🧠 Intent Detection:")
    for intent, score in intents[:3]:
        bar = "█" * int(score * 20)
        print(f"   {intent:<45} {score:.3f} {bar}")

    if top_score < MIN_BERT_CONFIDENCE:
        print(f"\n⚠️  Άγνωστη ερώτηση (score: {top_score:.2f}). Παρακαλώ επικοινωνήστε με τον Δήμο στο 2813 409185.")
        return

    # Step 2: TF-IDF search
    candidates = tfidf_search(top_intent, vectorizer, tfidf_matrix, valid_kb)
    best_score = candidates[0][0]

    print(f"\n🔍 TF-IDF Candidates (top score: {best_score:.3f}):")
    for score, r in candidates[:3]:
        print(f"   {score:.3f}  {r['title'][:55]}")

    # Out-of-scope check
    if best_score < MIN_TFIDF_SCORE:
        print(f"\n⚠️  Η υπηρεσία δεν φαίνεται να παρέχεται από τον Δήμο (score: {best_score:.3f}).")
        print(f"   Παρακαλώ επικοινωνήστε με τον Δήμο στο 2813 409185.")
        return

    # Step 3: Llama
    print(f"\n⚡ Απάντηση:")
    print(ask_llama(query, top_intent, candidates))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default=None, help="Ερώτηση (αν δεν δοθεί → interactive)")
    args = parser.parse_args()

    # Φόρτωση
    tokenizer, model, label_encoder = load_bert()

    with open(JSON_PATH, encoding="utf-8") as f:
        knowledge_base = json.load(f)

    vectorizer, tfidf_matrix, valid_kb = build_tfidf_index(knowledge_base)
    print()

    if args.query:
        run(args.query, tokenizer, model, label_encoder, vectorizer, tfidf_matrix, valid_kb)
    else:
        print("💬 Interactive mode — γράψε την ερώτησή σου (ή 'exit')\n")
        while True:
            try:
                query = input("Πολίτης: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nΈξοδος.")
                break
            if not query or query.lower() in {"exit", "quit", "q"}:
                break
            run(query, tokenizer, model, label_encoder, vectorizer, tfidf_matrix, valid_kb)


if __name__ == "__main__":
    main()
