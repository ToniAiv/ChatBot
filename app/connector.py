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
import csv
import json
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import requests
import torch

from language import (
    ENGLISH_SUPPORT,
    TranslationUnavailable,
    detect_language,
    translate_to_greek,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Paths ─────────────────────────────────────────────────────
# Αγκυρωμένα στη ρίζα του project, όχι στο cwd — το παλιό connector.py
# έτρεχε μόνο αν τον καλούσες από τον φάκελο New-praktiki/.
ROOT         = Path(__file__).resolve().parent.parent
MODEL_DIR    = ROOT / "models" / "intent-model-218"   # 218 κλάσεις (Αυγ 2026)
JSON_PATH    = ROOT / "data" / "heraklion_eservices.json"
TABLE_PATH   = ROOT / "mappings" / "intent_to_service.csv"
DEPTS_PATH   = ROOT / "mappings" / "departments.csv"
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"

# ── Ρυθμίσεις ─────────────────────────────────────────────────
MIN_BERT_CONFIDENCE = 0.80   # κάτω από αυτό → άγνωστη ερώτηση
# Το 0.80 δεν είναι αυθαίρετο: μετρήθηκε σε 1.907 άγνωστες προτάσεις
# (tests/RESULTS.txt). Απαντά στο 80% με 95% ακρίβεια. Στο παλιό 0.20
# το σύστημα απαντούσε σχεδόν πάντα, με 89% ακρίβεια — δηλαδή μία στις
# εννιά απαντήσεις λάθος, χωρίς καμία ένδειξη προς τον πολίτη.
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


# ══════════════════════════════════════════════════════════════
# Σημείωση για τα labels
# ══════════════════════════════════════════════════════════════
# Το query του TF-IDF είναι το ίδιο το intent label. Ο vectorizer είναι
# char_wb n-gram πάνω σε ΕΛΛΗΝΙΚΟΥΣ τίτλους KB, οπότε ένα label με
# λατινικούς χαρακτήρες δίνει score 0.000 — και το σύστημα απαντά "δεν
# παρέχεται από τον Δήμο" ακόμα κι όταν το BERT έχει αναγνωρίσει σωστά.
# Τον Αύγ 2026 ήταν έτσι 29 από τα 218 labels (13% της ταξινομίας,
# σιωπηλά νεκρό). Μετονομάστηκαν σε ελληνικά με το
# scripts/rename_greeklish_labels.py — αν προστεθεί ποτέ νέο label,
# πρέπει να είναι ελληνικό, άτονο, πεζό, με κάτω παύλες.


def load_service_table(valid_kb: list) -> dict:
    """
    Διαβάζει τον ελεγμένο πίνακα intent → υπηρεσία.

    Μόνο οι γραμμές με status verified/auto μπαίνουν. Οι "review" μένουν
    ΕΞΩ σκόπιμα: πέφτουν στον TF-IDF, δηλαδή στην παλιά συμπεριφορά.
    Έτσι ο πίνακας δεν μπορεί να κάνει το σύστημα χειρότερο απ' ό,τι ήταν
    — μόνο καλύτερο, όσο γυρίζουν γραμμές σε verified.
    """
    if not TABLE_PATH.exists():
        print("⚠️  Δεν βρέθηκε πίνακας υπηρεσιών — μόνο TF-IDF.")
        return {}

    by_url = {r["url"]: r for r in valid_kb}
    table, skipped = {}, 0
    with TABLE_PATH.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("status") not in ("verified", "auto"):
                continue
            rec = by_url.get(row.get("url", ""))
            if rec is None:      # ο πίνακας δείχνει σε εγγραφή που δεν υπάρχει πια
                skipped += 1
                continue
            table[row["intent"]] = rec

    msg = f"✅ Πίνακας υπηρεσιών: {len(table)} intents"
    if skipped:
        msg += f"  ({skipped} γραμμές δείχνουν σε άγνωστο URL — αγνοήθηκαν)"
    print(msg)
    return table


def load_departments() -> dict:
    """Τηλέφωνα τμημάτων από τον επίσημο κατάλογο του δήμου
    (heraklion.gr/municipality/contacts/thlefwna-yphresiwn.html)."""
    if not DEPTS_PATH.exists():
        return {}
    with DEPTS_PATH.open(encoding="utf-8-sig") as f:
        return {r["code"]: r for r in csv.DictReader(f)}


def load_no_service(depts: dict) -> dict:
    """
    intent → τμήμα, για τα intents που ΔΕΝ έχουν ηλεκτρονική υπηρεσία.

    218 intents έναντι 164 υπηρεσιών: δεκάδες αιτήματα πολιτών απλώς δεν
    γίνονται ηλεκτρονικά. Χωρίς αυτόν τον έλεγχο έπεφταν στην εφεδρεία,
    όπου ο TF-IDF έδινε την πλησιέστερη άσχετη υπηρεσία και το Llama
    εφηύρε αιτιολόγηση — μετρημένο στο 10,3% των ερωτήσεων. Το χειρότερο
    είδος λάθους: σίγουρο, εύγλωττο και λανθασμένο.
    """
    if not TABLE_PATH.exists():
        return {}
    out = {}
    with TABLE_PATH.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "no_service":
                out[row["intent"]] = depts.get(row.get("department", ""), {})
    print(f"✅ Χωρίς ηλεκτρονική υπηρεσία: {len(out)} intents → τηλέφωνο τμήματος")
    return out


def department_answer(dept: dict, lang: str = "el") -> str:
    """Απάντηση για αίτημα που δεν εξυπηρετείται ηλεκτρονικά."""
    m = MSG[lang]
    if not dept:
        return m["outofscope"]
    lines = [m["no_online"], "", f"{m['dept']}: {dept['name']}", f"{m['tel']}: {dept['tel']}"]
    if dept.get("email"):
        lines.append(f"{m['email']}: {dept['email']}")
    return "\n".join(lines)


def find_candidates(intent, vectorizer, tfidf_matrix, valid_kb, service_table=None):
    """
    Επιστρέφει (candidates, source). Ο πίνακας έχει προτεραιότητα· ο
    TF-IDF είναι η εφεδρεία για ό,τι δεν έχει ελεγχθεί ακόμα.

    Ένα hit στον πίνακα επιστρέφεται με score 1.0 ώστε να περνάει το
    MIN_TFIDF_SCORE: η αντιστοίχιση είναι ρητή, δεν είναι εικασία ομοιότητας.
    """
    if service_table:
        rec = service_table.get(intent)
        if rec is not None:
            return [(1.0, rec)], "table"
    return tfidf_search(intent, vectorizer, tfidf_matrix, valid_kb), "tfidf"


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

# ══════════════════════════════════════════════════════════════
# Μηνύματα προς τον πολίτη, ελληνικά και αγγλικά
# ══════════════════════════════════════════════════════════════
# Ο τίτλος της υπηρεσίας και το όνομα του τμήματος μένουν ΕΛΛΗΝΙΚΑ και
# στις αγγλικές απαντήσεις: είναι η επίσημη ονομασία που θα δει ο
# πολίτης στη σελίδα και στο έντυπο. Μεταφρασμένη θα ήταν αγνώριστη.
PHONE = "2813 409185"

MSG = {
    "el": {
        "unknown":   f"Δεν κατάλαβα την ερώτηση. Παρακαλώ επικοινωνήστε με τον Δήμο στο {PHONE}.",
        "outofscope": f"Η υπηρεσία δεν φαίνεται να παρέχεται από τον Δήμο. Παρακαλώ επικοινωνήστε στο {PHONE}.",
        "service":   "Η αρμόδια υπηρεσία του Δήμου είναι:",
        "no_online": "Το αίτημα αυτό δεν γίνεται ηλεκτρονικά.",
        "dept":      "Αρμόδιο", "tel": "Τηλέφωνο", "email": "Email",
        "greek_page": "",
        "tr_fail":   f"Η μετάφραση δεν είναι διαθέσιμη αυτή τη στιγμή. Παρακαλώ επικοινωνήστε στο {PHONE}.",
    },
    "en": {
        "unknown":   f"I did not understand your question. Please contact the Municipality at {PHONE}.",
        "outofscope": f"This does not appear to be a service the Municipality provides. Please call {PHONE}.",
        "service":   "The service you need is:",
        "no_online": "This request cannot be submitted online.",
        "dept":      "Department", "tel": "Phone", "email": "Email",
        "greek_page": "(the Municipality's page is in Greek)",
        "tr_fail":   f"Translation is unavailable right now. Please contact the Municipality at {PHONE}.",
    },
}


def resolve_query(query: str):
    """
    Επιστρέφει (ελληνικό_κείμενο, γλώσσα_πολίτη).

    Το BERT δέχεται ΠΑΝΤΑ ελληνικά. Αν ο πολίτης έγραψε αγγλικά,
    μεταφράζεται πρώτα — μία κλήση LLM, ~1,8s, μόνο σε αυτή την
    περίπτωση. Η ελληνική διαδρομή δεν πληρώνει τίποτα.
    """
    if not ENGLISH_SUPPORT:
        return query, "el"
    lang = detect_language(query)
    if lang == "el":
        return query, "el"
    return translate_to_greek(query), "en"


def compose_answer(query: str, intent: str, candidates: list, source: str,
                   lang: str = "el") -> str:
    """
    Παράγει την τελική απάντηση προς τον πολίτη.

    Όταν η υπηρεσία ήρθε από τον ελεγμένο πίνακα, το LLM ΔΕΝ καλείται:
    δεν έχει τίποτα να αποφασίσει — η αντιστοίχιση είναι ρητή — και θα
    πλήρωνε 6-14 δευτερόλεπτα μόνο για να διατυπώσει δυο προτάσεις,
    με το ρίσκο να πει κάτι ανακριβές για υπηρεσία που ήδη βρήκαμε
    σωστά. Μετρημένο: BERT 76-1050ms, πίνακας 0.0ms, Llama 6.1-13.9s.

    Το LLM κρατιέται για τη διαδρομή εφεδρείας, όπου όντως επιλέγει
    ανάμεσα σε πολλές υποψήφιες υπηρεσίες.
    """
    m = MSG[lang]
    if source == "table" and candidates:
        rec = candidates[0][1]
        note = f"\n{m['greek_page']}" if m["greek_page"] else ""
        return f"{m['service']}\n«{rec['title']}»{note}\n\n{rec['url']}"
    return ask_llama(query, intent, candidates)


def run(query, tokenizer, model, label_encoder, vectorizer, tfidf_matrix, valid_kb,
        service_table=None, no_service=None):
    print(f"\n{'─'*55}")
    print(f"  Ερώτηση: {query}")
    print(f"{'─'*55}")

    # Step 0: γλώσσα. Το BERT δέχεται πάντα ελληνικά.
    try:
        greek_query, lang = resolve_query(query)
    except TranslationUnavailable as exc:
        print(f"\n⚠️  Μετάφραση απέτυχε: {exc}")
        print(MSG["en"]["tr_fail"])
        return
    if lang == "en":
        print(f"  🌐 EN → EL: {greek_query}")

    # Step 1: BERT intent
    intents = detect_intent(greek_query, tokenizer, model, label_encoder)
    top_intent, top_score = intents[0]

    print(f"\n🧠 Intent Detection:")
    for intent, score in intents[:3]:
        bar = "█" * int(score * 20)
        print(f"   {intent:<45} {score:.3f} {bar}")

    if top_score < MIN_BERT_CONFIDENCE:
        print(f"\n⚠️  (score: {top_score:.2f})")
        print(MSG[lang]["unknown"])
        return

    # Step 2: TF-IDF search
    if no_service and top_intent in no_service:
        print(f"\n📞 Δεν εξυπηρετείται ηλεκτρονικά:")
        print(department_answer(no_service[top_intent], lang))
        return

    candidates, source = find_candidates(
        top_intent, vectorizer, tfidf_matrix, valid_kb, service_table)
    best_score = candidates[0][0]

    label = "Πίνακας (ελεγμένο)" if source == "table" else f"TF-IDF (top {best_score:.3f})"
    print(f"\n🔍 {label}:")
    for score, r in candidates[:3]:
        print(f"   {score:.3f}  {r['title'][:55]}")

    # Out-of-scope check
    if best_score < MIN_TFIDF_SCORE:
        print(f"\n⚠️  (score: {best_score:.3f})")
        print(MSG[lang]["outofscope"])
        return

    # Step 3: Llama
    print(f"\n⚡ Απάντηση:")
    print(compose_answer(greek_query, top_intent, candidates, source, lang))
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
    service_table = load_service_table(valid_kb)
    departments = load_departments()
    no_service = load_no_service(departments)
    print()

    if args.query:
        run(args.query, tokenizer, model, label_encoder, vectorizer, tfidf_matrix,
            valid_kb, service_table, no_service)
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
            run(query, tokenizer, model, label_encoder, vectorizer, tfidf_matrix,
                valid_kb, service_table, no_service)


if __name__ == "__main__":
    main()
