"""Προσυμπλήρωση των «review» γραμμών του πίνακα με προτάσεις LLM.

Ο σκοπός είναι να ΕΠΙΤΑΧΥΝΕΙ τον ανθρώπινο έλεγχο, όχι να τον
αντικαταστήσει: οι προτάσεις μπαίνουν με status `llm_proposed`, που ο
connector ΔΕΝ χρησιμοποιεί. Μέχρι να τις δει άνθρωπος και να τις γυρίσει
σε `verified`, το σύστημα συνεχίζει με την εφεδρεία του TF-IDF.

ΜΑΘΗΜΑ ΑΠΟ ΤΗΝ ΠΡΩΤΗ ΑΠΟΠΕΙΡΑ: δόθηκαν και οι 164 τίτλοι μαζί. Το
μοντέλο κατέρρευσε — απάντησε «καμία υπηρεσία» και στις 82, με
πανομοιότυπη διατύπωση. Με λίστα ~23 υποψηφίων απαντά κανονικά.

Η λίστα χτίζεται από τρία σήματα (char n-gram, word n-gram, κοινή λέξη
≥5 γραμμάτων) ώστε να μην κληρονομεί την αποτυχία του TF-IDF που έστειλε
αυτές τις γραμμές σε review. Μετρημένο recall στις 26 επιβεβαιωμένες
γραμμές: 26/26.

Χρήση:
    ollama serve &
    python3 scripts/llm_prefill_table.py            # συνεχίζει από checkpoint
    python3 scripts/llm_prefill_table.py --restart
"""
import argparse, csv, json, re, sys, time
from pathlib import Path

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from paths import KB, MAPPINGS, ROOT

MODEL = "qwen2.5:14b"
URL = "http://localhost:11434/api/generate"
TABLE = MAPPINGS / "intent_to_service.csv"
CKPT = MAPPINGS / ".llm_prefill.checkpoint.json"


ACRONYMS = {
    "μφπαδ": "μοναδες φροντιδας προσχολικης αγωγης διαπαιδαγωγησης παιδικος σταθμος",
    "κοκ":   "κωδικας οδικης κυκλοφοριας παραβαση",
    "ταπ":   "τελος ακινητης περιουσιας",
    "δευαη": "δημοτικη επιχειρηση υδρευσης αποχετευσης νερο λογαριασμος",
    "αμεα":  "ατομα με αναπηρια αναπηρικο",
    "καπη":  "κεντρο ανοικτης προστασιας ηλικιωμενων",
    "πεκ":   "πιστοποιητικο ελεγχου κατασκευης",
    "οασα":  "σταση λεωφορειου στεγαστρο",
    "συποθα": "συμβουλιο πολεοδομικων θεματων αμφισβητησεων",
}


def nrm(t: str) -> str:
    import unicodedata as u
    t = u.normalize("NFD", str(t))
    return "".join(c for c in t if u.category(c) != "Mn").lower().strip()


class Shortlister:
    """Υποψήφιες υπηρεσίες από τρία σήματα — όχι μόνο από τον TF-IDF."""

    def __init__(self, titles):
        self.titles = [nrm(t) for t in titles]
        self.v_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.m_char = self.v_char.fit_transform(self.titles)
        self.v_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
        self.m_word = self.v_word.fit_transform(self.titles)

    def __call__(self, intent: str) -> list:
        q = nrm(intent.replace("_", " "))
        q = " ".join([q] + [ACRONYMS[a] for a in ACRONYMS if a in q.split()])
        idx = set()
        for v, m, k in ((self.v_char, self.m_char, 15), (self.v_word, self.m_word, 12)):
            s = cosine_similarity(v.transform([q]), m)[0]
            idx.update(np.argsort(s)[::-1][:k].tolist())
        words = {w for w in q.split() if len(w) >= 5}
        for i, t in enumerate(self.titles):
            if words & {w for w in t.split() if len(w) >= 5}:
                idx.add(i)
        return sorted(idx)


def readable(intent: str) -> str:
    return intent.replace("_", " ")


def build_prompt(intent: str, titles: list, cand: list) -> str:
    listing = "\n".join(f"{i}. {titles[i]}" for i in cand)
    return f"""Είσαι υπάλληλος του Δήμου Ηρακλείου και ξέρεις τις ηλεκτρονικές υπηρεσίες του.

Ένας πολίτης έχει αίτημα της κατηγορίας: «{readable(intent)}»

Παρακάτω είναι ΟΛΕΣ οι διαθέσιμες ηλεκτρονικές υπηρεσίες του Δήμου:
{listing}

Ποια υπηρεσία αντιστοιχεί σε αυτή την κατηγορία αιτήματος;

Κανόνες:
- Διάλεξε τον αριθμό ΜΟΝΟ αν η υπηρεσία αφορά πράγματι αυτό το αίτημα.
- Αν καμία υπηρεσία δεν ταιριάζει, γράψε -1. Είναι απολύτως αποδεκτό:
  ο Δήμος δεν έχει ηλεκτρονική υπηρεσία για κάθε αίτημα.
- ΜΗΝ διαλέξεις υπηρεσία που απλώς μοιάζει λεκτικά. Π.χ. «αντίγραφα
  οικοδομικών αδειών» ΔΕΝ είναι το ίδιο με «έκδοση οικοδομικής άδειας».
- confidence: high μόνο αν είσαι σίγουρος, αλλιώς low.

Απάντησε ΜΟΝΟ με JSON σε αυτή τη μορφή:
{{"choice": <αριθμός ή -1>, "confidence": "high|low", "why": "<μία σύντομη πρόταση>"}}"""


def ask(prompt: str) -> dict:
    r = requests.post(URL, json={
        "model": MODEL, "prompt": prompt, "stream": False,
        "format": "json", "options": {"temperature": 0},
    }, timeout=300)
    r.raise_for_status()
    raw = r.json()["response"].strip()
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        d = json.loads(m.group(0)) if m else {}
    return d if isinstance(d, dict) else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()

    kb = json.load(open(KB, encoding="utf-8"))
    valid = [r for r in kb if r["title"] not in ("Ο ΛΟΓΑΡΙΑΣΜΟΣ ΜΟΥ", "")]
    titles = [r["title"] for r in valid]

    shortlist = Shortlister(titles)

    rows = list(csv.DictReader(TABLE.open(encoding="utf-8-sig")))
    REDO = ("review", "llm_proposed", "llm_no_service")
    todo = [r for r in rows if r["status"] in REDO]
    print(f"{len(todo)} γραμμές προς προσυμπλήρωση, {MODEL}\n")

    done = {}
    if CKPT.exists() and not args.restart:
        done = json.loads(CKPT.read_text(encoding="utf-8"))
        print(f"[resume] {len(done)} ήδη έτοιμες\n")

    t0 = time.monotonic()
    for i, row in enumerate(todo, 1):
        intent = row["intent"]
        if intent in done:
            continue
        try:
            d = ask(build_prompt(intent, titles, shortlist(intent)))
        except Exception as exc:
            print(f"  ✗ {intent}: {exc}")
            break

        c = d.get("choice", -1)
        allowed = set(shortlist(intent))
        c = c if isinstance(c, int) and c in allowed else -1
        done[intent] = {
            "choice": c,
            "title": titles[c] if c >= 0 else "",
            "url": valid[c]["url"] if c >= 0 else "",
            "confidence": str(d.get("confidence", "low")).lower(),
            "why": str(d.get("why", ""))[:160],
        }
        CKPT.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")

        mark = "—" if c < 0 else ("✓" if done[intent]["confidence"] == "high" else "?")
        eta = (time.monotonic() - t0) / max(1, len(done)) * (len(todo) - len(done)) / 60
        print(f"  [{i:3}/{len(todo)}] {mark} {intent[:44]:<46} ETA {eta:4.1f}′")

    # ── γράψιμο πίσω στον πίνακα ──
    filled = noservice = 0
    for row in rows:
        d = done.get(row["intent"])
        if not d or row["status"] not in REDO:
            continue
        if d["choice"] < 0:
            row["status"] = "llm_no_service"
            row["note"] = f"LLM: καμία αντίστοιχη υπηρεσία — {d['why']}"
            noservice += 1
        else:
            row["status"] = "llm_proposed"
            row["title"], row["url"] = d["title"], d["url"]
            row["note"] = f"LLM {d['confidence']}: {d['why']}"
            filled += 1

    with TABLE.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        order = {"llm_proposed": 0, "llm_no_service": 1, "review": 2, "auto": 3, "verified": 4}
        w.writerows(sorted(rows, key=lambda r: (order.get(r["status"], 9), r["intent"])))

    high = sum(1 for d in done.values() if d["choice"] >= 0 and d["confidence"] == "high")
    print(f"\n  προτάθηκε υπηρεσία : {filled}  (από αυτές high confidence: {high})")
    print(f"  «καμία υπηρεσία»   : {noservice}")
    print(f"\n  💾 {TABLE.relative_to(ROOT)}")
    print("\n  ΠΡΟΣΟΧΗ: status `llm_proposed` ΔΕΝ χρησιμοποιείται από τον connector.")
    print("  Γύρισέ το σε `verified` αφού το ελέγξει άνθρωπος.")


if __name__ == "__main__":
    main()
