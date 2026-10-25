"""
Intent Evaluation Harness
==========================
Μετράει το BERT intent classifier και βγάζει error analysis.

Τι δίνει:
  1. Top-1 / Top-3 / Top-5 accuracy + macro-F1
  2. Per-class accuracy — ταξινομημένο από το ΧΕΙΡΟΤΕΡΟ
  3. Confusion pairs — ποια intents μπερδεύονται μεταξύ τους
  4. Threshold sweep — τι κερδίζεις/χάνεις σε κάθε τιμή MIN_BERT_CONFIDENCE

Χρήση:
    # Diagnostic πάνω στο dataset εκπαίδευσης (βλ. προειδοποίηση contamination)
    python3 eval_intent.py --dataset Expanded_Intent_Dataset_2.csv

    # Γρήγορο τρέξιμο σε δείγμα
    python3 eval_intent.py --dataset Expanded_Intent_Dataset_2.csv --limit 1000

    # Σε ξεχωριστό gold set (ΤΟ ΣΩΣΤΟ — δες gold_set.csv)
    python3 eval_intent.py --dataset gold_set.csv --held-out
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from clean_dataset import canon_label, canon_text, strip_accents
from paths import ROOT, DATA, DATASETS, MODELS, MAPPINGS, MODEL_218, KB, KB_RICH

MODEL_DIR = MODEL_218
BATCH_SIZE = 32
THRESHOLDS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


# ══════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════
def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def predict_batched(texts: list, tokenizer, model, device, batch_size=BATCH_SIZE):
    """Επιστρέφει (top5_indices, top5_probs) ως numpy arrays σχήματος (N, 5)."""
    all_idx, all_prob = [], []
    total = len(texts)

    for start in range(0, total, batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=128, padding=True,
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        k = min(5, probs.shape[1])
        top = torch.topk(probs, k=k, dim=-1)

        all_idx.append(top.indices.cpu().numpy())
        all_prob.append(top.values.cpu().numpy())

        done = min(start + batch_size, total)
        print(f"\r  inference {done}/{total}", end="", flush=True)

    print()
    return np.vstack(all_idx), np.vstack(all_prob)


# ══════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════
def macro_f1(gold: list, pred: list) -> float:
    labels = set(gold) | set(pred)
    scores = []
    for lab in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(gold, pred) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(gold, pred) if g == lab and p != lab)
        if tp == 0:
            scores.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        scores.append(2 * prec * rec / (prec + rec))
    return sum(scores) / len(scores) if scores else 0.0


def report_overall(gold, top5_labels, top5_probs) -> None:
    n = len(gold)
    pred1 = [row[0] for row in top5_labels]
    top1 = sum(1 for g, row in zip(gold, top5_labels) if g == row[0])
    top3 = sum(1 for g, row in zip(gold, top5_labels) if g in row[:3])
    top5 = sum(1 for g, row in zip(gold, top5_labels) if g in row[:5])

    print("\n" + "═" * 66)
    print("  ΣΥΝΟΛΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ")
    print("═" * 66)
    print(f"  Δείγματα        : {n}")
    print(f"  Top-1 accuracy  : {top1 / n:.4f}   ({top1}/{n})")
    print(f"  Top-3 accuracy  : {top3 / n:.4f}   ({top3}/{n})")
    print(f"  Top-5 accuracy  : {top5 / n:.4f}   ({top5}/{n})")
    print(f"  Macro F1        : {macro_f1(gold, pred1):.4f}")
    print(f"  Μέσο confidence : {top5_probs[:, 0].mean():.4f}")


def report_per_class(gold, top5_labels, worst_n: int) -> None:
    per = defaultdict(lambda: [0, 0])   # label → [correct, total]
    for g, row in zip(gold, top5_labels):
        per[g][1] += 1
        if g == row[0]:
            per[g][0] += 1

    ranked = sorted(per.items(), key=lambda kv: (kv[1][0] / kv[1][1], -kv[1][1]))

    print("\n" + "─" * 66)
    print(f"  ΧΕΙΡΟΤΕΡΑ {worst_n} INTENTS  (προτεραιότητα διόρθωσης)")
    print("─" * 66)
    for label, (ok, tot) in ranked[:worst_n]:
        acc = ok / tot
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {acc:.2f} {bar}  {ok:3}/{tot:3}  {label}")

    perfect = sum(1 for _, (ok, tot) in per.items() if ok == tot)
    print(f"\n  {perfect}/{len(per)} intents με 100% accuracy")


def report_confusions(gold, top5_labels, top_n: int) -> None:
    conf = Counter()
    for g, row in zip(gold, top5_labels):
        if g != row[0]:
            conf[(g, row[0])] += 1

    if not conf:
        print("\n  Κανένα λάθος — δεν υπάρχουν confusions.")
        return

    print("\n" + "─" * 66)
    print(f"  ΤΟΠ {top_n} ΣΥΓΧΥΣΕΙΣ  (σωστό → προβλεπόμενο)")
    print("─" * 66)
    for (g, p), c in conf.most_common(top_n):
        print(f"  {c:4}×  {g}")
        print(f"        → {p}")


def report_thresholds(gold, top5_labels, top5_probs) -> None:
    """
    Δείχνει το trade-off: όσο ανεβάζεις το threshold, τόσο λιγότερες
    ερωτήσεις απαντάς, αλλά με μεγαλύτερη ακρίβεια σε όσες απαντάς.
    """
    n = len(gold)
    print("\n" + "─" * 66)
    print("  THRESHOLD SWEEP  (MIN_BERT_CONFIDENCE)")
    print("─" * 66)
    print(f"  {'thresh':>7} {'coverage':>10} {'acc@answered':>14} {'λάθος απαντ.':>14}")
    print("  " + "-" * 50)

    for t in THRESHOLDS:
        answered = [(g, row[0]) for g, row, p in zip(gold, top5_labels, top5_probs[:, 0]) if p >= t]
        if not answered:
            print(f"  {t:>7.2f} {0.0:>9.1%} {'—':>14} {'—':>14}")
            continue
        ok = sum(1 for g, p in answered if g == p)
        cov = len(answered) / n
        acc = ok / len(answered)
        wrong = len(answered) - ok
        print(f"  {t:>7.2f} {cov:>9.1%} {acc:>13.1%} {wrong:>10} / {n}")

    print("\n  coverage      = % ερωτήσεων που το σύστημα απαντά (δεν κάνει abstain)")
    print("  acc@answered  = ακρίβεια ΜΟΝΟ σε όσες απάντησε")
    print("  Για δημόσια υπηρεσία, μια λάθος απάντηση κοστίζει περισσότερο")
    print("  από ένα «δεν ξέρω» — άρα προτίμησε υψηλότερο threshold.")


# ══════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description="Αξιολόγηση BERT intent classifier")
    ap.add_argument("--dataset", required=True, help="CSV με στήλες text,intent")
    ap.add_argument("--limit", type=int, default=None, help="Χρήση μόνο N δειγμάτων")
    ap.add_argument("--worst", type=int, default=20, help="Πόσα χειρότερα intents")
    ap.add_argument("--confusions", type=int, default=15)
    ap.add_argument("--held-out", action="store_true",
                    help="Δήλωσε ότι το dataset ΔΕΝ χρησιμοποιήθηκε στο training")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-dir", default=None)
    args = ap.parse_args()

    global MODEL_DIR
    if args.model_dir:
        MODEL_DIR = Path(args.model_dir)

    path = Path(args.dataset)
    if not path.exists():
        sys.exit(f"Δεν βρέθηκε: {path}")

    with path.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)]

    # ΚΡΙΣΙΜΟ: το nlpaueb/bert-base-greek-uncased-v1 είναι uncased, αλλά το
    # αποθηκευμένο tokenizer_config.json έχει do_lower_case=False και
    # strip_accents=None. Χωρίς χειροκίνητο normalize, το 60-70% των tokens
    # γίνεται [UNK] και η ακρίβεια πέφτει στο ~4%. Ίδια λογική με
    # connector.normalize() — πρέπει να μένουν συγχρονισμένα.
    samples = []
    for r in rows:
        text, label = canon_text(r.get("text")), canon_label(r.get("intent"))
        if text and label:
            samples.append((strip_accents(text).lower(), strip_accents(label).lower()))

    if args.limit and args.limit < len(samples):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(samples), size=args.limit, replace=False)
        samples = [samples[i] for i in idx]

    print(f"📊 Αξιολόγηση {len(samples)} δειγμάτων από {path.name}")

    if not args.held_out:
        print()
        print("  ⚠  ΠΡΟΣΟΧΗ — CONTAMINATION")
        print("     Το μοντέλο εκπαιδεύτηκε πάνω σε αυτό το dataset (80/20 split),")
        print("     άρα τα νούμερα είναι ΑΙΣΙΟΔΟΞΑ και ΔΕΝ είναι δημοσιεύσιμα.")
        print("     Χρήσιμο μόνο ως diagnostic: ό,τι αποτυγχάνει ΕΔΩ είναι")
        print("     πραγματικό πρόβλημα, όχι θόρυβος γενίκευσης.")

    device = pick_device()
    print(f"\n🔄 Φόρτωση μοντέλου ({device})...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval().to(device)
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    classes = list(label_encoder.classes_)
    print(f"✅ Έτοιμο — {len(classes)} κλάσεις.\n")

    known = set(classes)
    unknown = {lab for _, lab in samples if lab not in known}
    if unknown:
        print(f"  ⚠  {len(unknown)} intents στο dataset που το μοντέλο δεν ξέρει:")
        for u in sorted(unknown):
            print(f"       {u}")
        print("     (μετρώνται ως λάθος — το μοντέλο δεν μπορεί να τα προβλέψει)\n")

    texts = [t for t, _ in samples]
    gold = [lab for _, lab in samples]

    top5_idx, top5_probs = predict_batched(texts, tokenizer, model, device)
    top5_labels = [[classes[i] for i in row] for row in top5_idx]

    report_overall(gold, top5_labels, top5_probs)
    report_per_class(gold, top5_labels, args.worst)
    report_confusions(gold, top5_labels, args.confusions)
    report_thresholds(gold, top5_labels, top5_probs)
    print("═" * 66)


if __name__ == "__main__":
    main()
