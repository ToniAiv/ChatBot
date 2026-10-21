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

    PYTHONPATH=scripts:app python agent/guarded.py qwen3:14b             (σύγκριση + greeklish, v1)
    PYTHONPATH=scripts:app python agent/guarded.py qwen3:14b --holdout   (νέο holdout: v1 / v2 / v2+thinking)
    PYTHONPATH=scripts:app python agent/guarded.py qwen3:14b --prompt-language   (v1: ελληνικές / αγγλικές οδηγίες)
    PYTHONPATH=scripts:app python agent/guarded.py <μοντέλο> --single           (v1 με άλλο μοντέλο, π.χ. Krikri)
    PYTHONPATH=scripts:app python agent/guarded.py --safe                       (v1.1 σε όλα τα σετ + holdout2 με v1 και v1.1)
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

ASK_PICK_EN = """You are the routing assistant of the Municipality of Heraklion (Greece). A citizen wrote a message and the search found a few possible services.
Choose the number of the service that CLEARLY matches the message. If more than one matches, if none matches, or if the message is not enough to tell them apart, answer 0.
In "evidence", copy the words of the message that show the citizen means this service and not the others."""

ASK_REWRITE_EN = """A citizen wrote to the virtual assistant of the Municipality of Heraklion (Greece), but the search did not find a service. Classify the message:
- "request": asks for a service, certificate, permit, payment or information, or reports a problem that may concern the Municipality. In "text", rewrite the request as one simple, correct Greek sentence, the way the Municipality's website would phrase it (fix greeklish and spelling, translate if it is in another language). Do not add anything the citizen did not say.
- "smalltalk": greeting, thanks, question about the assistant. In "text", write a short polite reply in Greek.
- "out_of_scope": a matter of another authority (weather, tax office, passports, pharmacies, electricity providers etc.). In "text", write briefly and politely, in Greek, that it is not a matter for the Municipality.
NEVER put phone numbers, links or emails in "text"."""

ASK_SPLIT_EN = """Split the citizen's message into separate requests to the Municipality of Heraklion (Greece).
Split ONLY if it asks for two or more different things (e.g. two different services or two different problems). Questions about the same topic (e.g. documents and phone number of the same service) are ONE request.
Write each request as a self-contained sentence, using the citizen's words."""

# v1.1 («safe»): οι δύο αλλαγές του reports/guarded_agent.md μετά τη σύγκριση με το Krikri
ASK_PICK_SAFE = ASK_PICK.replace(
    'Στο "evidence" αντέγραψε τις λέξεις του μηνύματος που δείχνουν ότι εννοεί αυτή την υπηρεσία και όχι τις άλλες.',
    'Στο "evidence" γράψε 1 έως 3 λέξεις του μηνύματος που ξεχωρίζουν αυτή την υπηρεσία από τις άλλες (όχι όλο το μήνυμα).')
assert ASK_PICK_SAFE != ASK_PICK

TEMPLATES = {   # ζώνη Γ: το LLM αποφασίζει μόνο το είδος· το κείμενο είναι έτοιμο
    "el": {"smalltalk": "Είμαι ο εικονικός βοηθός του Δήμου Ηρακλείου. Σας βοηθάω να βρείτε υπηρεσίες του Δήμου: "
                        "πιστοποιητικά, βεβαιώσεις, άδειες, αιτήματα για καθαριότητα, δρόμους και φωτισμό, πληρωμές. "
                        "Γράψτε μου τι χρειάζεστε.",
           "out_of_scope": "Δεν βρήκα υπηρεσία του Δήμου Ηρακλείου γι' αυτό το θέμα· ίσως αφορά άλλον φορέα. "
                           f"Αν πιστεύετε ότι αφορά τον Δήμο, καλέστε στο {C.PHONE}."},
    "en": {"smalltalk": "I am the virtual assistant of the Municipality of Heraklion. I help you find Municipality "
                        "services: certificates, permits, requests about cleaning, roads and lighting, payments. "
                        "Tell me what you need.",
           "out_of_scope": "I could not find a Municipality of Heraklion service for this; it may concern another "
                           f"authority. If you think it concerns the Municipality, please call {C.PHONE}."},
}

PROMPTS = {
    "el": {"pick": ASK_PICK, "rewrite": ASK_REWRITE, "split": ASK_SPLIT, "message": "Μήνυμα", "options": "Επιλογές"},
    "en": {"pick": ASK_PICK_EN, "rewrite": ASK_REWRITE_EN, "split": ASK_SPLIT_EN, "message": "Message", "options": "Options"},
}

MAYBE_COMPOSITE = re.compile(r"\b(και|επισης|αλλα|ακομα)\b")


_THINKING: dict = {}


def supports_thinking(model: str) -> bool:
    if model not in _THINKING:
        show = requests.post(B.OLLAMA.replace("/api/chat", "/api/show"), json={"model": model}, timeout=30).json()
        _THINKING[model] = "thinking" in (show.get("capabilities") or [])
    return _THINKING[model]


def llm(model: str, system: str, user: str, schema: dict, think: bool = False, num_ctx: int = 4096) -> dict:
    payload = {
        "model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "format": schema, "stream": False,
        "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": 2048 if think else 300}}
    if supports_thinking(model):          # στο Qwen ρητά false· αλλιώς «σκέφτεται» εξ ορισμού
        payload["think"] = think
    r = requests.post(B.OLLAMA, json=payload, timeout=600 if think else 180)
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
    """Κείμενο του LLM: χωρίς συνδέσμους / τηλέφωνα / email, στα ελληνικά. Και χωρίς
    σύντομους αριθμούς: το Krikri έγραψε «τηλ: 1135» για τα φαρμακεία."""
    return (bool(text.strip()) and not B.facts(text) and not re.search(r"\d{3,}", text)
            and B.greek_ratio(text) >= 0.6)


def evidence_specific(message: str, evidence: str) -> bool:
    """v1.1: έως 3 λέξεις περιεχομένου, και όχι (σχεδόν) όλο το μήνυμα."""
    words = [w for w in re.findall(r"\w+", evidence) if len(w) >= 4]
    ev, msg = _stems(evidence), _stems(message)
    return len(words) <= 3 and not (len(msg) >= 3 and len(ev) >= 0.8 * len(msg))


def answer(text: str, model: str, prompts: str = "el", safe: bool = False) -> dict:
    """safe=True: η v1.1 (έτοιμα κείμενα στη ζώνη Γ, αυστηρότερος έλεγχος επιλογής)."""
    P = PROMPTS[prompts]
    fl = floor(text)
    if safe:
        info = inspect(text, fl)
        zone, dest, lang = info["zone"], info["dest"], info["lang"]
    else:
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
            parts = ask(P["split"], text, {"type": "object", "required": ["requests"], "properties": {
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
            schema = {"type": "object", "required": ["choice", "evidence"],
                      "properties": {"choice": {"type": "integer", "enum": list(range(len(opts) + 1))},
                                     "evidence": {"type": "string"}}}

            def pick(order):
                listing = "\n".join(f"{i}. {o['title']}" for i, o in enumerate(order, 1))
                r = ask(ASK_PICK_SAFE if safe else P["pick"],
                        f"{P['message']}: {text}\n\n{P['options']}:\n{listing}", schema)
                c = r.get("choice")
                return (order[c - 1] if isinstance(c, int) and 1 <= c <= len(order) else None), c, str(r.get("evidence", ""))

            chosen, c, ev = pick(opts)
            out["actions"].append(f"pick:{c} «{ev[:40]}»")
            if safe and chosen:
                # Αυστηρότερος έλεγχος: σύντομη, συγκεκριμένη αιτιολόγηση και ίδια επιλογή
                # όταν οι επιλογές δοθούν με ανάποδη σειρά. Καταγράφονται χωριστά.
                ev_ok = pick_ok(text, ev, opts) and evidence_specific(text, ev)
                again, c2, _ = pick(list(reversed(opts)))
                same = bool(again) and again["intent"] == chosen["intent"]
                out["actions"] += [f"chosen:{chosen['intent']}", f"ev_ok:{int(ev_ok)}", f"order:{c2}:{int(same)}"]
                if ev_ok and same:
                    out.update(text=service_answer(chosen["intent"], lang),
                               options=[o for o in opts if o is not chosen])
                else:
                    out["actions"].append("validator2:απόρριψη")
            elif safe:
                pass
            elif isinstance(c, int) and 1 <= c <= len(opts) and not pick_ok(text, ev, opts):
                out["actions"].append("validator2:απόρριψη")
            elif isinstance(c, int) and 1 <= c <= len(opts):
                chosen = opts[c - 1]
                out.update(text=gui.Api().choose(chosen["intent"])["text"],
                           options=[o for o in opts if o is not chosen])   # τα υπόλοιπα μένουν κουμπιά
        elif zone == "Γ":
            r = ask(P["rewrite"], text, {"type": "object", "required": ["kind", "text"], "properties": {
                "kind": {"type": "string", "enum": ["request", "smalltalk", "out_of_scope"]},
                "text": {"type": "string"}}})
            kind, new = r.get("kind"), str(r.get("text", "")).strip()
            out["actions"].append(f"{kind}:{new[:80]}")
            if kind == "request" and new:
                nf = floor(new)
                if (inspect(new, nf)["zone"] if safe else zone_of(new, nf)[0]) == "A":
                    out["actions"].append("rewrite→A")
                    out.update(text=nf["text"], options=nf["options"])
            elif kind in ("smalltalk", "out_of_scope") and safe:
                out.update(text=TEMPLATES["en" if lang == "en" else "el"][kind], options=[])
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


# ── v2: οι αλλαγές του reports/guarded_agent.md («Πού είναι αδύναμες οι οδηγίες») ──
# περιγραφές στις επιλογές, ουδέτερη σειρά, παραδείγματα, λεξιλόγιο τίτλων στην
# αναδιατύπωση, ελληνική εκδοχή του μηνύματος, απάντηση στη γλώσσα του πολίτη,
# και προαιρετικά thinking.
ASK_PICK_V2 = """Είσαι ο βοηθός δρομολόγησης του Δήμου Ηρακλείου. Ο πολίτης έγραψε ένα μήνυμα και η αναζήτηση βρήκε μερικές πιθανές υπηρεσίες, η καθεμία με μια σύντομη περιγραφή. Η σειρά τους δεν σημαίνει τίποτα.
Διάλεξε τον αριθμό της υπηρεσίας που κάνει ΑΚΡΙΒΩΣ αυτό που ζητά ο πολίτης. Διάβασε τις περιγραφές: υπηρεσίες με παρόμοιο τίτλο μπορεί να κάνουν διαφορετικά πράγματα.
Απάντησε 0 αν ταιριάζουν περισσότερες από μία, αν δεν ταιριάζει καμία, ή αν το μήνυμα δεν αρκεί για να ξεχωρίσεις.
Στο "evidence" αντέγραψε τις λέξεις του μηνύματος που δείχνουν ότι εννοεί αυτή την υπηρεσία και όχι τις άλλες.

Παραδείγματα (φανταστικές υπηρεσίες, μόνο για τη μορφή):

Μήνυμα: το πεζοδρόμιο μπροστά στο σπίτι μου έχει σπάσει και σκοντάφτουν οι περαστικοί
Επιλογές:
1. Άδεια κατασκευής πεζοδρομίου — Για ιδιοκτήτες που θέλουν να κατασκευάσουν νέο πεζοδρόμιο.
2. Άδεια τραπεζοκαθισμάτων — Για καταστήματα που θέλουν τραπέζια στο πεζοδρόμιο.
3. Αναφορά φθοράς πεζοδρομίου — Δηλώστε σπασμένες πλάκες ή λακκούβες σε πεζοδρόμιο.
Απάντηση: {"choice": 3, "evidence": "έχει σπάσει"}

Μήνυμα: θέλω μια βεβαίωση
Επιλογές:
1. Βεβαίωση μη οφειλής — Βεβαιώνει ότι δεν υπάρχουν χρέη προς τον Δήμο.
2. Βεβαίωση μόνιμης κατοικίας — Βεβαιώνει τη διεύθυνση κατοικίας του δημότη.
3. Βεβαίωση χρήσης γης — Βεβαιώνει τις επιτρεπόμενες χρήσεις ενός ακινήτου.
Απάντηση: {"choice": 0, "evidence": ""}"""

ASK_REWRITE_V2 = ASK_REWRITE + """
Το "text" γράφεται ΠΑΝΤΑ στα ελληνικά, ακόμα κι αν ο πολίτης έγραψε σε άλλη γλώσσα.
Για "request": αν κάποιος από τους παρακάτω τίτλους υπηρεσιών ή τμημάτων του Δήμου ταιριάζει στο αίτημα, γράψε το αίτημα με τις λέξεις αυτού του τίτλου (π.χ. «registry office» → «ληξιαρχείο»).

Υπηρεσίες:
{services}

Τμήματα:
{departments}"""


def _vocabulary() -> str:
    services = sorted({v["title"] for v in C.SECTIONS.values() if v.get("title") and "Καλώς" not in v["title"]})
    depts = sorted({d.get("name", "") for d in bot.depts.values() if d.get("name")})
    return ASK_REWRITE_V2.format(services="\n".join(f"- {t}" for t in services),
                                 departments="\n".join(f"- {t}" for t in depts))


def describe(o: dict) -> str:
    """Μία γραμμή για την επιλογή: η «Περιγραφή» της σελίδας ή το τμήμα."""
    if o.get("kind") == "phone" or o["intent"] in bot.ns:
        d = bot.ns.get(o["intent"]) or {}
        return f"Τηλεφωνική επικοινωνία με το τμήμα «{d.get('name', '')}»."
    cand, _src = C.find_candidates(o["intent"], bot.vec, bot.mat, bot.valid, bot.table)
    url = cand[0][1]["url"] if cand else ""
    return B._trim((C.SECTIONS.get(url) or {}).get("sections", {}).get("Περιγραφή", ""), 220)


def inspect(text: str, fl: dict) -> dict:
    """Validator 1 (όπως το zone_of) + η ελληνική εκδοχή και η γλώσσα του bot."""
    if C.smalltalk_answer(text, bot.vec, bot.mat, bot.valid, bot.table, bot.ns):
        return {"zone": "smalltalk", "dest": None, "greek": text, "lang": "el"}
    gq, lang = C.resolve_query(text)
    intents = C.detect_intent(gq, bot.tok, bot.mdl, bot.le)
    code = C.department_override(gq, intents)
    if code:
        return {"zone": "A", "dest": ("dept", code), "greek": gq, "lang": lang}
    if C.destination_confidence(intents, bot.table, bot.ns) >= C.MIN_BERT_CONFIDENCE:
        dest = C.destination_key(intents[0][0], bot.table, bot.ns) or ("intent", intents[0][0])
        return {"zone": "A", "dest": dest, "greek": gq, "lang": lang}
    return {"zone": "B" if fl["options"] else "Γ", "dest": None, "greek": gq, "lang": lang}


def service_answer(intent: str, lang: str) -> str:
    """Η απάντηση του bot για μια υπηρεσία, στη γλώσσα του πολίτη (το choose() είναι μόνο ελληνικά)."""
    if intent in bot.ns:
        return C.department_answer(bot.ns[intent], lang)
    cand, src = C.find_candidates(intent, bot.vec, bot.mat, bot.valid, bot.table)
    return C.compose_answer("", intent, cand, src, lang) if cand else gui.Api().choose(intent)["text"]


def answer_v2(text: str, model: str, think: bool = False) -> dict:
    fl = floor(text)
    info = inspect(text, fl)
    zone, dest, greek, lang = info["zone"], info["dest"], info["greek"], info["lang"]
    out = {"zone": zone, "actions": [], "llm_calls": 0, "text": fl["text"], "options": fl["options"],
           "floor_text": fl["text"]}

    def ask(system, user, schema, use_think=think, num_ctx=8192):   # ίδιο num_ctx παντού: αλλιώς το Ollama ξαναφορτώνει το μοντέλο
        out["llm_calls"] += 1
        return llm(model, system, user, schema, use_think, num_ctx)

    if zone == "smalltalk":
        return out
    shown = text if C.normalize(greek) == C.normalize(text) else f"{text}\n(στα ελληνικά: {greek})"
    try:
        if MAYBE_COMPOSITE.search(C.normalize(greek)) and len(text.split()) >= 6:   # ίδιο με v1
            parts = ask(ASK_SPLIT, text, {"type": "object", "required": ["requests"], "properties": {
                "requests": {"type": "array", "items": {"type": "string"}, "maxItems": 3}}},
                use_think=False).get("requests") or []
            out["actions"].append(f"split:{len(parts)}")
            if len(parts) >= 2:
                found = {dest: fl["text"]} if zone == "A" else {}
                for p in parts:
                    pf = floor(p)
                    pi = inspect(p, pf)
                    if pi["zone"] == "A" and pi["dest"] not in found:
                        found[pi["dest"]] = pf["text"]
                if len(found) >= 2 or (zone != "A" and found):
                    out["actions"].append(f"composite:{len(found)}")
                    out.update(text="\n\n".join(found.values()), options=[])
                    return out
        if zone == "B":
            opts = sorted(fl["options"], key=lambda o: C.normalize(o["title"]))   # ουδέτερη σειρά
            listing = "\n".join(f"{i}. {o['title']} — {describe(o)}" for i, o in enumerate(opts, 1))
            r = ask(ASK_PICK_V2, f"Μήνυμα: {shown}\nΕπιλογές:\n{listing}", {
                "type": "object", "required": ["choice", "evidence"],
                "properties": {"choice": {"type": "integer", "enum": list(range(len(opts) + 1))},
                               "evidence": {"type": "string"}}})
            c, ev = r.get("choice"), str(r.get("evidence", ""))
            out["actions"].append(f"pick:{c} «{ev[:40]}»")
            if isinstance(c, int) and 1 <= c <= len(opts):
                if pick_ok(f"{text} {greek}", ev, opts):
                    chosen = opts[c - 1]
                    out.update(text=service_answer(chosen["intent"], lang),
                               options=[o for o in fl["options"] if o["intent"] != chosen["intent"]])
                else:
                    out["actions"].append("validator2:απόρριψη")
        elif zone == "Γ":
            r = ask(_vocabulary(), f"Μήνυμα: {shown}\n\n(Το text στα ελληνικά.)", {"type": "object", "required": ["kind", "text"],
                "properties": {"kind": {"type": "string", "enum": ["request", "smalltalk", "out_of_scope"]},
                               "text": {"type": "string"}}})
            kind, new = r.get("kind"), str(r.get("text", "")).strip()
            out["actions"].append(f"{kind}:{new[:80]}")
            if kind == "request" and new:
                nf = floor(new)
                if inspect(new, nf)["zone"] == "A":
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


def evaluate_holdout(model: str):
    """v1 / v2 / v2+thinking στο agent/holdout_questions.csv (γράφτηκε πριν τρέξει μοντέλο)."""
    qs = [{"id": r["id"], "type": r["type"], "question": r["text"], "expect": r["expect"], "lang": r["lang"]}
          for r in csv.DictReader(open(ROOT / "agent/holdout_questions.csv", encoding="utf-8"))]
    configs = [("v1", lambda t: answer(t, model)), ("v2", lambda t: answer_v2(t, model)),
               ("v2+think", lambda t: answer_v2(t, model, think=True))]
    fields = ["config", "id", "type", "lang", "question", "expect", "zone", "actions", "llm_calls", "seconds",
              "floor_correct", "correct", "confident_error", "floor_answer", "answer"]
    fake = {"tool_text": "", "tool_calls": [{}], "seconds": 0}
    summary = {"model": model, "first_button": {"n": 0, "correct": 0}}
    with open(ROOT / "reports/holdout_answers.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for name, fn in configs:
            agg = {"n_service": 0, "floor_correct": 0, "correct": 0, "n_other": 0, "floor_other": 0, "other": 0,
                   "wins": [], "losses": [], "confident_errors": [], "pick_right": 0, "pick_wrong": 0,
                   "rewrite_right": 0, "rewrite_wrong": 0, "llm_used": 0, "seconds_llm": []}
            print(f"\n=== {name}", flush=True)
            for q in qs:
                t0 = time.monotonic()
                res = fn(q["question"])
                sec = round(time.monotonic() - t0, 1)
                ok_floor = B.score(q, {**fake, "answer": res["floor_text"]})["correct"]
                ok = B.score(q, {**fake, "answer": res["text"]})["correct"]
                svc = q["type"] in ("service", "department", "clarify")
                changed = res["text"] != res["floor_text"]
                # βέβαιο λάθος που δεν υπήρχε: η απάντηση άλλαξε, είναι λάθος, και δίνει
                # σύνδεσμο/τηλέφωνο (ή, στα εκτός θέματος, οτιδήποτε λάθος)
                conf_err = changed and not ok and (bool(B.facts(res["text"])) if svc else True)
                agg["n_service" if svc else "n_other"] += 1
                agg["floor_correct" if svc else "floor_other"] += ok_floor
                agg["correct" if svc else "other"] += ok
                if ok and not ok_floor: agg["wins"].append(q["id"])
                if ok_floor and not ok: agg["losses"].append(q["id"])
                if conf_err: agg["confident_errors"].append(q["id"])
                acts = " | ".join(res["actions"])
                if changed and "pick:" in acts and "composite" not in acts:
                    agg["pick_right" if ok else "pick_wrong"] += 1
                if "rewrite→A" in acts:
                    agg["rewrite_right" if ok else "rewrite_wrong"] += 1
                if res["llm_calls"]:
                    agg["llm_used"] += 1; agg["seconds_llm"].append(sec)
                if name == "v1" and res["zone"] == "B" and svc and res["options"] is not None:
                    fl_opts = floor(q["question"])["options"]
                    if fl_opts:
                        first = service_answer(fl_opts[0]["intent"], "el")
                        summary["first_button"]["n"] += 1
                        summary["first_button"]["correct"] += B.score(q, {**fake, "answer": first})["correct"]
                mark = "+" if ok and not ok_floor else ("-" if ok_floor and not ok else ("!" if conf_err else ("✓" if ok else "✗")))
                print(f"  {mark} {q['id']} {res['zone']:9} {sec:5.1f}s {acts[:90]}  {q['question'][:40]}", flush=True)
                w.writerow({"config": name, "id": q["id"], "type": q["type"], "lang": q["lang"],
                            "question": q["question"], "expect": q["expect"], "zone": res["zone"], "actions": acts,
                            "llm_calls": res["llm_calls"], "seconds": sec, "floor_correct": ok_floor, "correct": ok,
                            "confident_error": conf_err, "floor_answer": res["floor_text"], "answer": res["text"]})
                f.flush()
            v = agg["seconds_llm"]; agg["seconds_llm"] = round(sum(v) / len(v), 1) if v else None
            summary[name] = agg
            print(json.dumps(agg, ensure_ascii=False), flush=True)
    json.dump(summary, open(ROOT / "reports/holdout_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary["first_button"]), "\nαποθηκεύτηκαν: reports/holdout_answers.csv, reports/holdout_summary.json")


def changed_kind(q: dict, text: str) -> str:
    """Αλλαγμένη απάντηση: «σωστό» (όλοι οι στόχοι), «μισό» (μέρος τους, καμία
    άλλη υπηρεσία) ή «λάθος» (τουλάχιστον μία υπηρεσία εκτός στόχων)."""
    if text in {t for d in TEMPLATES.values() for t in d.values()} and q["type"] in ("service", "department"):
        return "ουδέτερο"      # έτοιμο κείμενο: ούτε υπηρεσία ούτε λάθος πληροφορία
    if q["type"] not in ("service", "department"):
        return "σωστό" if B.score(q, {"tool_text": "", "tool_calls": [{}], "seconds": 0, "answer": text})["correct"] else "λάθος"
    targets = B.gold_targets(q["expect"], q["type"])
    hits = sum(B.hit(text, t) for t in targets)
    gold_titles = set()
    for g in [x for x in q["expect"].split("|") if x]:
        c = B.CLASSES.get(B.key(g))
        if c is not None:
            cd = B.card(c, 1.0)
            gold_titles.add(C.normalize(cd.get("title") or cd.get("name") or ""))
    extra = [t for t in re.findall(r"«([^»]+)»", text) if C.normalize(t) not in gold_titles]
    if targets and hits == len(targets):
        return "σωστό"
    return "μισό" if hits and not extra else "λάθος"


def evaluate_prompt_language(model: str):
    """v1 με ελληνικές και με αγγλικές οδηγίες, ανά ερώτηση δίπλα δίπλα, σε όλα τα σετ."""
    evaluate_configs([("el", model, "el"), ("en", model, "en")], "prompt_language")


def evaluate_configs(configs: list, out: str, only: tuple = ("bakeoff", "greeklish", "holdout")):
    """Η v1 για κάθε (ετικέτα, μοντέλο, γλώσσα οδηγιών[, safe]), ανά ερώτηση δίπλα δίπλα."""
    configs = [tuple(c) + (False,) * (4 - len(c)) for c in configs]
    labels = [c[0] for c in configs]
    sets = question_sets()
    for name in ("holdout", "holdout2"):
        sets[name] = [{"id": r["id"], "type": r["type"], "question": r["text"], "expect": r["expect"]}
                      for r in csv.DictReader(open(ROOT / f"agent/{name}_questions.csv", encoding="utf-8"))]
    sets = {k: v for k, v in sets.items() if k in only}
    fields = ["set", "prompts", "id", "type", "question", "expect", "zone", "actions", "llm_calls", "seconds",
              "floor_correct", "correct", "changed", "answer"]
    fake = {"tool_text": "", "tool_calls": [{}], "seconds": 0}
    summary = {"configs": configs}
    with open(ROOT / f"reports/{out}_answers.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for name, qs in sets.items():
            aggs = {k: {"n_service": 0, "floor_correct": 0, "correct": 0, "n_other": 0, "floor_other": 0, "other": 0,
                        "wins": [], "losses": [], "changed": {"σωστό": [], "μισό": [], "ουδέτερο": [], "λάθος": []},
                        "llm_used": 0, "seconds_llm": []} for k in labels}
            print(f"\n=== {name}", flush=True)
            for q in qs:
                marks = []
                for lang, model, prompts, safe in configs:
                    agg = aggs[lang]
                    t0 = time.monotonic()
                    res = answer(q["question"], model, prompts=prompts, safe=safe)
                    sec = round(time.monotonic() - t0, 1)
                    ok_floor = B.score(q, {**fake, "answer": res["floor_text"]})["correct"]
                    ok = B.score(q, {**fake, "answer": res["text"]})["correct"]
                    svc = q["type"] in ("service", "department", "clarify")
                    agg["n_service" if svc else "n_other"] += 1
                    agg["floor_correct" if svc else "floor_other"] += ok_floor
                    agg["correct" if svc else "other"] += ok
                    if ok and not ok_floor: agg["wins"].append(q["id"])
                    if ok_floor and not ok: agg["losses"].append(q["id"])
                    kind = changed_kind(q, res["text"]) if res["text"] != res["floor_text"] else ""
                    if kind: agg["changed"][kind].append(q["id"])
                    if res["llm_calls"]:
                        agg["llm_used"] += 1; agg["seconds_llm"].append(sec)
                    marks.append(f"{lang}:{kind or ('✓' if ok else '✗')}")
                    w.writerow({"set": name, "prompts": lang, "id": q["id"], "type": q["type"], "question": q["question"],
                                "expect": q["expect"], "zone": res["zone"], "actions": " | ".join(res["actions"]),
                                "llm_calls": res["llm_calls"], "seconds": sec, "floor_correct": ok_floor,
                                "correct": ok, "changed": kind, "answer": res["text"]})
                f.flush()
                print(f"  {q['id']:4} {res['zone']:9} {'  '.join(marks)}  {q['question'][:45]}", flush=True)
            for agg in aggs.values():
                v = agg["seconds_llm"]; agg["seconds_llm"] = round(sum(v) / len(v), 1) if v else None
            summary[name] = aggs
            print(json.dumps(aggs, ensure_ascii=False), flush=True)
    json.dump(summary, open(ROOT / f"reports/{out}_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nαποθηκεύτηκαν: reports/{out}_answers.csv, reports/{out}_summary.json")


def main():
    args = sys.argv[1:]
    model = ([a for a in args if not a.startswith("--")] or ["qwen3:14b"])[0]
    if "--holdout" in args:
        return evaluate_holdout(model)
    if "--prompt-language" in args:
        return evaluate_prompt_language(model)
    if "--safe" in args:                   # v1 / v1.1 για Qwen και Krikri
        krikri = "hf.co/ilsp/Llama-Krikri-8B-Instruct-GGUF:Q4_K_M"
        evaluate_configs([("qwen-v1.1", "qwen3:14b", "el", True), ("krikri-v1.1", krikri, "el", True)], "safe_old_sets")
        return evaluate_configs([("qwen-v1", "qwen3:14b", "el", False), ("qwen-v1.1", "qwen3:14b", "el", True),
                                 ("krikri-v1", krikri, "el", False), ("krikri-v1.1", krikri, "el", True)],
                                "holdout2", only=("holdout2",))
    if "--single" in args:                 # π.χ. άλλο μοντέλο με τη v1 και ελληνικές οδηγίες
        tag = re.sub(r"\W+", "_", model.split("/")[-1].split(":")[0].lower())
        return evaluate_configs([(tag, model, "el")], f"model_{tag}")
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
