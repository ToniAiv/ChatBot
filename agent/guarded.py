"""
Welcome agent «με πάτωμα»: το σημερινό bot αποφασίζει, το LLM βοηθά μόνο εκεί
που το BERT δεν είναι σίγουρο, και ό,τι κάνει το LLM ελέγχεται από κώδικα.
Αποτελέσματα και προβλέψεις: reports/guarded_agent.md.

  Validator 1 (κώδικας): το BERT τρέχει πάντα στο αρχικό μήνυμα.
    Α  σίγουρο            → η απάντηση του σημερινού bot, χωρίς LLM
    Β  «Μήπως εννοείτε;»  → το LLM διαλέγει μία από τις επιλογές ή καμία
    Γ  τίποτα             → το LLM ξαναγράφει το αίτημα· κρατιέται μόνο αν το BERT γίνει σίγουρο
  Σύνθετο μήνυμα: το LLM το χωρίζει· κρατιούνται μόνο κομμάτια με σίγουρο BERT.
  Validator 2 (κώδικας): το LLM δεν γράφει ποτέ σύνδεσμο ή τηλέφωνο. Όπου γράφει
    δικό του κείμενο (χαιρετισμοί, εκτός θέματος), απορρίπτεται αν έχει.

    PYTHONPATH=scripts:app python agent/guarded.py qwen3:14b
"""
from __future__ import annotations

import csv, json, re, sys, time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "agent"), str(ROOT / "scripts"), str(ROOT / "app")]
import bakeoff as B                      # βαθμολόγηση (ίδια με τη σύγκριση)
import connector as C
import greeklish
import gui
import usage_log

usage_log.ENABLED = False                # ερωτήσεις δοκιμής, όχι πολιτών
gui._load_pipeline()
bot = B.bot

ASK_PICK = """Είσαι ο βοηθός δρομολόγησης του Δήμου Ηρακλείου. Ο πολίτης έγραψε ένα μήνυμα και η αναζήτηση βρήκε μερικές πιθανές υπηρεσίες.
Διάλεξε τον αριθμό της υπηρεσίας που ταιριάζει ΣΑΦΩΣ στο μήνυμα. Αν ταιριάζουν περισσότερες από μία, αν δεν ταιριάζει καμία, ή αν το μήνυμα δεν αρκεί για να ξεχωρίσεις, απάντησε 0.
Στο "evidence" αντέγραψε τις λέξεις του μηνύματος που δείχνουν ότι εννοεί αυτή την υπηρεσία και όχι τις άλλες."""

ASK_REWRITE = """Ο πολίτης έγραψε στον εικονικό βοηθό του Δήμου Ηρακλείου, αλλά η αναζήτηση δεν βρήκε υπηρεσία. Κατάταξε το μήνυμα:
- "request": ζητά υπηρεσία, πιστοποιητικό, άδεια, πληρωμή, πληροφορία, ή αναφέρει πρόβλημα που μπορεί να αφορά τον Δήμο. Στο "text" ξαναγράψε το αίτημα σε μία απλή, σωστή ελληνική πρόταση, όπως θα το έγραφε η σελίδα του Δήμου (διόρθωσε greeklish και ορθογραφία, μετάφρασε αν είναι σε άλλη γλώσσα). Μην προσθέτεις κάτι που δεν είπε.
- "smalltalk": χαιρετισμός, ευχαριστίες, ερώτηση για τον βοηθό. Στο "text" γράψε σύντομη ευγενική απάντηση.
- "out_of_scope": θέμα άλλου φορέα (καιρός, εφορία, διαβατήρια, φαρμακεία, πάροχοι ρεύματος κ.λπ.). Στο "text" γράψε σύντομα και ευγενικά ότι δεν είναι θέμα του Δήμου.
Στο "text" ΠΟΤΕ τηλέφωνα, συνδέσμους ή email."""

ASK_SPLIT = """Χώρισε το μήνυμα του πολίτη σε ξεχωριστά αιτήματα προς τον Δήμο Ηρακλείου.
Χώρισε ΜΟΝΟ αν ζητά δύο ή περισσότερα διαφορετικά πράγματα (π.χ. δύο διαφορετικές υπηρεσίες ή δύο διαφορετικά προβλήματα). Ερωτήσεις για το ίδιο θέμα (π.χ. δικαιολογητικά και τηλέφωνο της ίδιας υπηρεσίας) είναι ΕΝΑ αίτημα.
Γράψε κάθε αίτημα ως αυτοτελή πρόταση, με τις λέξεις του πολίτη."""

MAYBE_COMPOSITE = re.compile(r"\b(και|επισης|αλλα|ακομα)\b")


def llm(model: str, system: str, user: str, schema: dict) -> dict:
    r = requests.post(B.OLLAMA, json={
        "model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "format": schema, "stream": False, "think": False,
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 300}}, timeout=180)
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])


def floor(text: str) -> dict:
    """Το σημερινό bot, χωρίς μνήμη συζήτησης."""
    return gui.Api().chat(text)


def zone_of(text: str, fl: dict) -> tuple:
    """Validator 1: ίδιες αποφάσεις με το gui.Api.chat. Επιστρέφει (ζώνη, προορισμός)."""
    if C.smalltalk_answer(text, bot.vec, bot.mat, bot.valid, bot.table, bot.ns):
        return "smalltalk", None
    gq, _lang = C.resolve_query(text)
    intents = C.detect_intent(gq, bot.tok, bot.mdl, bot.le)
    code = C.department_override(gq, intents)
    if code:
        return "A", ("dept", code)
    if C.destination_confidence(intents, bot.table, bot.ns) >= C.MIN_BERT_CONFIDENCE:
        return "A", C.destination_key(intents[0][0], bot.table, bot.ns) or ("intent", intents[0][0])
    return ("B" if fl["options"] else "Γ"), None


def _stems(text: str) -> set:
    text = greeklish.transliterate(text)         # «metadimotefsi» = «μεταδημότευση», ίδια και στις δύο πλευρές
    return {w[:5] for w in re.findall(r"\w+", C.normalize(text)) if len(w) >= 5}


def pick_ok(message: str, evidence: str, opts: list) -> bool:
    """Validator 2 της ζώνης Β: οι λέξεις που επικαλείται το LLM υπάρχουν στο
    μήνυμα και δεν είναι κοινές σε όλες τις επιλογές («μεταδημότευση» → όχι)."""
    ev, msg = _stems(evidence), _stems(message)
    if not ev or not ev <= msg:
        return False
    shared = set.intersection(*(_stems(o["title"]) for o in opts))
    return bool(ev - shared)


def validator2(text: str) -> bool:
    """Κείμενο του LLM: χωρίς συνδέσμους / τηλέφωνα / email, στα ελληνικά."""
    return bool(text.strip()) and not B.facts(text) and B.greek_ratio(text) >= 0.6


def answer(text: str, model: str) -> dict:
    fl = floor(text)
    zone, dest = zone_of(text, fl)
    out = {"zone": zone, "actions": [], "llm_calls": 0, "text": fl["text"], "options": fl["options"],
           "floor_text": fl["text"]}

    def ask(system, user, schema):
        out["llm_calls"] += 1
        return llm(model, system, user, schema)

    if zone == "smalltalk":
        return out
    try:
        # Σύνθετο μήνυμα: η απάντηση για όλο το μήνυμα μένει πρώτη, και
        # προστίθενται μόνο κομμάτια με σίγουρο BERT σε άλλο προορισμό.
        if MAYBE_COMPOSITE.search(C.normalize(text)) and len(text.split()) >= 6:
            parts = ask(ASK_SPLIT, text, {"type": "object", "required": ["requests"], "properties": {
                "requests": {"type": "array", "items": {"type": "string"}, "maxItems": 3}}}).get("requests") or []
            out["actions"].append(f"split:{len(parts)}")
            if len(parts) >= 2:
                found = {dest: fl["text"]} if zone == "A" else {}
                for p in parts:
                    pf = floor(p)
                    pz, pd = zone_of(p, pf)
                    if pz == "A" and pd not in found:
                        found[pd] = pf["text"]
                if len(found) >= 2 or (zone != "A" and found):
                    out["actions"].append(f"composite:{len(found)}")
                    out.update(text="\n\n".join(found.values()), options=[])
                    return out
        if zone == "B":
            opts = fl["options"]
            listing = "\n".join(f"{i}. {o['title']}" for i, o in enumerate(opts, 1))
            r = ask(ASK_PICK, f"Μήνυμα: {text}\n\nΕπιλογές:\n{listing}", {
                "type": "object", "required": ["choice", "evidence"],
                "properties": {"choice": {"type": "integer", "enum": list(range(len(opts) + 1))},
                               "evidence": {"type": "string"}}})
            c, ev = r.get("choice"), str(r.get("evidence", ""))
            out["actions"].append(f"pick:{c} «{ev[:40]}»")
            if isinstance(c, int) and 1 <= c <= len(opts) and not pick_ok(text, ev, opts):
                out["actions"].append("validator2:απόρριψη")
            elif isinstance(c, int) and 1 <= c <= len(opts):
                chosen = opts[c - 1]
                out.update(text=gui.Api().choose(chosen["intent"])["text"],
                           options=[o for o in opts if o is not chosen])   # τα υπόλοιπα μένουν κουμπιά
        elif zone == "Γ":
            r = ask(ASK_REWRITE, text, {"type": "object", "required": ["kind", "text"], "properties": {
                "kind": {"type": "string", "enum": ["request", "smalltalk", "out_of_scope"]},
                "text": {"type": "string"}}})
            kind, new = r.get("kind"), str(r.get("text", "")).strip()
            out["actions"].append(f"{kind}:{new[:80]}")
            if kind == "request" and new:
                nf = floor(new)
                if zone_of(new, nf)[0] == "A":
                    out["actions"].append("rewrite→A")
                    out.update(text=nf["text"], options=nf["options"])
            elif kind in ("smalltalk", "out_of_scope"):
                if validator2(new):
                    tail = f" Για θέματα του Δήμου: {C.PHONE}." if kind == "out_of_scope" else ""
                    out.update(text=new + tail, options=[])
                else:
                    out["actions"].append("validator2:απόρριψη")
    except Exception as e:                     # οτιδήποτε στραβό → το σημερινό bot
        out["actions"].append(f"σφάλμα:{str(e)[:80]}")
        out.update(text=fl["text"], options=fl["options"])
    return out


# ── αξιολόγηση ───────────────────────────────────────────────
def question_sets() -> dict:
    sets = {"bakeoff": [{"id": r["id"], "type": r["type"], "question": r["text"], "expect": r["expect"]}
                        for r in csv.DictReader(open(ROOT / "agent/bakeoff_questions.csv", encoding="utf-8"))]}
    sets["greeklish"] = [{"id": f"g{i:02d}", "type": "service", "question": r["text"], "expect": r["intent"]}
                         for i, r in enumerate(csv.DictReader(open(ROOT / "datasets/greeklish_eval.csv",
                                                                   encoding="utf-8")), 1)]
    return sets


def main():
    model = (sys.argv[1:] or ["qwen3:14b"])[0]
    path = ROOT / "reports/guarded_answers.csv"
    fields = ["set", "id", "type", "question", "expect", "zone", "actions", "llm_calls", "seconds",
              "floor_correct", "correct", "floor_answer", "answer"]
    summary = {"model": model}
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for name, qs in question_sets().items():
            agg = {"n_service": 0, "floor_correct": 0, "correct": 0, "n_other": 0, "floor_other": 0,
                   "other": 0, "wins": [], "losses": [], "llm_used": 0, "zones": {}, "seconds_llm": [],
                   "seconds_no_llm": []}
            print(f"\n=== {name}", flush=True)
            for q in qs:
                t0 = time.monotonic()
                res = answer(q["question"], model)
                sec = round(time.monotonic() - t0, 1)
                fake = {"tool_text": "", "tool_calls": [{}], "seconds": sec}   # το BERT τρέχει πάντα
                ok_floor = B.score(q, {**fake, "answer": res["floor_text"]})["correct"]
                ok = B.score(q, {**fake, "answer": res["text"]})["correct"]
                svc = q["type"] in ("service", "department", "clarify")
                agg["n_service" if svc else "n_other"] += 1
                agg["floor_correct" if svc else "floor_other"] += ok_floor
                agg["correct" if svc else "other"] += ok
                if ok and not ok_floor:
                    agg["wins"].append(q["id"])
                if ok_floor and not ok:
                    agg["losses"].append(q["id"])
                agg["zones"][res["zone"]] = agg["zones"].get(res["zone"], 0) + 1
                agg["llm_used"] += bool(res["llm_calls"])
                agg["seconds_llm" if res["llm_calls"] else "seconds_no_llm"].append(sec)
                mark = "+" if ok and not ok_floor else ("-" if ok_floor and not ok else ("✓" if ok else "✗"))
                print(f"  {mark} {q['id']} {res['zone']:9} {sec:5.1f}s {' | '.join(res['actions'])[:90]}"
                      f"  {q['question'][:45]}", flush=True)
                w.writerow({"set": name, "id": q["id"], "type": q["type"], "question": q["question"],
                            "expect": q["expect"], "zone": res["zone"], "actions": " | ".join(res["actions"]),
                            "llm_calls": res["llm_calls"], "seconds": sec, "floor_correct": ok_floor,
                            "correct": ok, "floor_answer": res["floor_text"], "answer": res["text"]})
                f.flush()
            for k in ("seconds_llm", "seconds_no_llm"):
                v = agg[k]
                agg[k] = round(sum(v) / len(v), 1) if v else None
            summary[name] = agg
            print(json.dumps(agg, ensure_ascii=False), flush=True)
    json.dump(summary, open(ROOT / "reports/guarded_summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\nαποθηκεύτηκαν: reports/guarded_answers.csv, reports/guarded_summary.json")


if __name__ == "__main__":
    main()
