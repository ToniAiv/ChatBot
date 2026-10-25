"""
LLM-assisted Intent → URL Mapping
==================================
Δεύτερο πέρασμα πάνω από το build_intent_map.py: για κάθε intent δίνει στο LLM
τους top-N υποψήφιους ΜΕ το περιεχόμενο της σελίδας και το βάζει να διαλέξει.

Γιατί βοηθά:
    Το TF-IDF συγκρίνει τριγράμματα χαρακτήρων, οπότε «εκταφή» ταιριάζει με
    «επιστροφή πινακίδων» λόγω κοινών συλλαβών. Τα embeddings πιάνουν γενική
    ομοιότητα αλλά όχι ελληνική διοικητική ορολογία. Το LLM ξέρει ότι η εκταφή
    αφορά κοιμητήρια.

Το `agreement` δείχνει πού συμφωνούν LLM και similarity — εκεί η εμπιστοσύνη
είναι υψηλή. Ο ανθρώπινος έλεγχος χρειάζεται μόνο στις διαφωνίες.

Κάνει checkpoint σε κάθε intent, οπότε διακοπή δεν χάνει δουλειά (--resume).

Χρήση:
    ollama serve &
    python3 llm_map_review.py --dataset Expanded_Intent_Dataset_2.csv \
        --kb data/heraklion_eservices_enriched.json --out intent_map_llm.csv
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests

from build_intent_map import (
    BODY_CHARS, build_queries, load_intents, load_kb,
    score_classifier, score_embeddings, score_tfidf,
)

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:14b"
TOP_N = 8                 # πόσοι υποψήφιοι πάνε στο LLM
BODY_SNIPPET = 300        # χαρακτήρες περιεχομένου ανά υποψήφιο
REQUEST_TIMEOUT = 180


PROMPT = """Είσαι υπάλληλος του Δήμου Ηρακλείου και γνωρίζεις τις υπηρεσίες του.

Ένας πολίτης θέλει: «{intent}»

Παρακάτω είναι {n} υπηρεσίες του Δήμου. Διάλεξε ΠΟΙΑ αντιστοιχεί σε αυτό που
θέλει ο πολίτης.

{candidates}

Κανόνες:
- Διάλεξε τον αριθμό της υπηρεσίας που ταιριάζει ΑΚΡΙΒΩΣ στο αίτημα.
- Αν καμία δεν ταιριάζει πραγματικά, απάντησε 0. Μην διαλέγεις κάτι
  «σχετικό» — καλύτερα 0 παρά λάθος υπηρεσία.
- Πρόσεξε τα παρόμοια: «ταφή» ≠ «οικογενειακή βοήθεια», «ύδρευση» ≠ «φωτισμός».

Απάντησε ΜΟΝΟ με JSON:
{{"choice": <αριθμός 0-{n}>, "confidence": <"high"|"medium"|"low">, "reason": "<το πολύ 8 λέξεις>"}}"""


def build_candidates(kb: list, order: np.ndarray) -> str:
    lines = []
    for rank, idx in enumerate(order, start=1):
        r = kb[idx]
        body = (r.get("body") or "")[:BODY_SNIPPET].strip()
        offline = "" if r.get("online", True) else "  [ΜΟΝΟ με φυσική παρουσία]"
        lines.append(f"{rank}. {r['title']}{offline}\n   {body}")
    return "\n\n".join(lines)


def ask_llm(model: str, intent: str, candidates: str, n: int) -> dict:
    prompt = PROMPT.format(
        intent=intent.replace("_", " "), n=n, candidates=candidates
    )
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 400},
        },
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    raw = resp.json()["response"].strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Τα ελληνικά κοστίζουν πολλά tokens, οπότε η αιτιολογία μπορεί να κοπεί
        # στη μέση και να αχρηστεύσει το JSON. Το `choice` όμως έρχεται πρώτο,
        # άρα σχεδόν πάντα σώζεται — μην πετάς μια σωστή απάντηση.
        m = re.search(r'"choice"\s*:\s*(\d+)', raw)
        if not m:
            return {"choice": 0, "confidence": "low",
                    "reason": f"μη έγκυρο JSON: {raw[:80]}"}
        conf = re.search(r'"confidence"\s*:\s*"(\w+)"', raw)
        return {
            "choice": int(m.group(1)),
            "confidence": conf.group(1).lower() if conf else "low",
            "reason": "(η απάντηση κόπηκε — ανακτήθηκε το choice)",
        }

    try:
        choice = int(parsed.get("choice", 0))
    except (TypeError, ValueError):
        choice = 0

    return {
        "choice": choice if 0 <= choice <= n else 0,
        "confidence": str(parsed.get("confidence", "")).lower(),
        "reason": str(parsed.get("reason", ""))[:160],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM review του intent→URL mapping")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--kb", default="data/heraklion_eservices_enriched.json")
    ap.add_argument("--out", default="intent_map_llm.csv")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--top-n", type=int, default=TOP_N)
    ap.add_argument("--limit", type=int, default=None, help="Μόνο τα πρώτα N intents")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
    except requests.exceptions.RequestException:
        sys.exit("❌ Το Ollama δεν τρέχει. Ξεκίνα το με:  ollama serve")

    by_intent = load_intents(Path(args.dataset))
    kb = load_kb(Path(args.kb))
    titles = [r["title"] for r in kb]
    bodies = [r.get("body", "") for r in kb]

    if not any(bodies):
        sys.exit("❌ Το KB δεν έχει `body` — τρέξε πρώτα enrich_kb.py")

    intents, queries = build_queries(by_intent)
    if args.limit:
        intents, queries = intents[:args.limit], queries[:args.limit]

    print(f"📋 {len(intents)} intents × {len(kb)} υπηρεσίες  |  μοντέλο: {args.model}\n")

    # Ίδια σήματα με το build_intent_map — μόνο για να διαλέξουμε υποψήφιους
    emb = score_embeddings(queries, titles, bodies)
    tfidf = score_tfidf(queries, [f"{t} {b[:BODY_CHARS]}" for t, b in zip(titles, bodies)])
    clf = score_classifier(intents, titles, bodies)
    combined = 0.6 * emb + 0.4 * tfidf + 0.5 * clf

    ckpt = Path(args.out).with_suffix(".checkpoint.json")
    done: dict = {}
    if args.resume and ckpt.exists():
        done = json.loads(ckpt.read_text(encoding="utf-8"))
        print(f"[resume] {len(done)} intents ήδη ολοκληρωμένα\n")

    rows, agree, t0 = [], 0, time.monotonic()

    for i, intent in enumerate(intents, start=1):
        order = np.argsort(combined[i - 1])[::-1][:args.top_n]
        sim_best = kb[order[0]]

        if intent in done:
            result = done[intent]
        else:
            try:
                result = ask_llm(
                    args.model, intent, build_candidates(kb, order), len(order)
                )
            except requests.exceptions.RequestException as e:
                result = {"choice": 0, "confidence": "low", "reason": f"σφάλμα: {e}"}
            done[intent] = result
            ckpt.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")

        choice = result["choice"]
        picked = kb[order[choice - 1]] if 1 <= choice <= len(order) else None
        same = picked is not None and picked["url"] == sim_best["url"]
        agree += int(same)

        rows.append({
            "intent": intent,
            "δείγματα": len(by_intent[intent]),
            "συμφωνία": "ΝΑΙ" if same else ("—" if picked is None else "ΟΧΙ"),
            "llm_confidence": result["confidence"],
            "url": picked["url"] if picked else "",
            "τίτλος_llm": picked["title"] if picked else "(καμία)",
            "τίτλος_similarity": sim_best["title"],
            "ηλεκτρονικά": ("ναι" if picked.get("online", True) else "ΟΧΙ") if picked else "",
            "αιτιολογία": result["reason"],
        })

        elapsed = time.monotonic() - t0
        eta = elapsed / i * (len(intents) - i)
        mark = "=" if same else ("∅" if picked is None else "≠")
        print(f"  [{i:3}/{len(intents)}] {mark} {intent[:34]:<36} "
              f"→ {(picked['title'] if picked else '(καμία)')[:34]:<36} "
              f"ETA {eta/60:.1f}′")

    fields = ["intent", "δείγματα", "συμφωνία", "llm_confidence", "url",
              "τίτλος_llm", "τίτλος_similarity", "ηλεκτρονικά", "αιτιολογία"]
    order_rank = {"ΟΧΙ": 0, "—": 1, "ΝΑΙ": 2}
    rows.sort(key=lambda r: (order_rank[r["συμφωνία"]], r["intent"]))

    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)

    none_picked = sum(1 for r in rows if r["συμφωνία"] == "—")
    print("\n" + "═" * 66)
    print(f"  ✅ Συμφωνία LLM & similarity : {agree:3} / {len(rows)}"
          f"   ← υψηλή εμπιστοσύνη")
    print(f"  🔍 Διαφωνία                  : {len(rows) - agree - none_picked:3}"
          f"   ← θέλουν ανθρώπινο έλεγχο")
    print(f"  ∅  Το LLM δεν βρήκε καμία    : {none_picked:3}"
          f"   ← πιθανώς δεν παρέχεται")
    print(f"\n  💾 {args.out}")
    print("═" * 66)


if __name__ == "__main__":
    main()
