"""
Η ροή του σχήματος για ένα μήνυμα. Τη σειρά την ορίζει ο κώδικας· οι agents
κρίνουν μόνο μέσα στο δικό τους βήμα.

    1. Welcome    παραλαβή: κουβέντα → έτοιμη απάντηση
    2. Security   κανόνες σε όλο το μήνυμα
    3. Welcome    χωρισμός σε αιτήματα
    4. Security   μοντέλο, ανά αίτημα· χειραγώγηση και προσβολές σταματούν εδώ.
                  Το «εκτός αρμοδιότητας» σταματά εδώ με scope=before· με scope=after
                  το αίτημα συνεχίζει με την ετικέτα του. Το «επείγον» σταματά εδώ με
                  emergency=block· με emergency=answer συνεχίζει, και η απάντηση
                  ξεκινά με το 112
    5. BERT       υποψήφιοι προορισμοί ανά αίτημα
    6. Validator  ταιριάζει κάποιος;
                    όχι, και ο Security είπε «εκτός αρμοδιότητας» → απάντηση «άλλος
                          φορέας», με τις επιλογές του BERT αν υπάρχουν
                    όχι → case 1: ο Welcome αναδιατυπώνει και το BERT ξανατρέχει
                          (χωρίς Security: το αίτημα έχει ήδη εγκριθεί)
                    πάλι όχι → case 2: ερώτηση προς τον πολίτη
    7. Welcome    σύνθεση: απαντήσεις για όσα λύθηκαν + ερώτηση για όσα όχι
    8. Validator  τελικός έλεγχος κάθε απάντησης

    from multiagent import pipeline
    reply = pipeline.run("θέλω βεβαίωση κατοικίας")     # reply.text, reply.options, reply.trace
"""
from __future__ import annotations

import time

import connector as C

from multiagent import classifier, config, security, validator, welcome
from multiagent.state import Reply, Segment, Session, Step


def _describe(cls) -> str:
    if cls.error:
        return f"η μετάφραση απέτυχε ({cls.error})"
    if not cls.intents:
        return "χωρίς αποτέλεσμα"
    top, score = cls.intents[0]
    if cls.confident:
        where = cls.candidates[0].title if cls.candidates else "χωρίς σελίδα ή τμήμα"
        return f"σίγουρο {cls.confidence:.2f}: {top} → «{where}»"
    if cls.candidates:
        return f"αβέβαιο {cls.confidence:.2f}: {top} ({score:.2f}), {len(cls.candidates)} επιλογές"
    return f"αβέβαιο {cls.confidence:.2f}: {top} ({score:.2f}), καμία επιλογή"


def _classify(text: str, pending, trace: list, note: str = ""):
    t0 = time.monotonic()
    cls = classifier.classify(text, pending)
    trace.append(Step("classifier", "bert", _describe(cls) + note, round(time.monotonic() - t0, 2)))
    return cls


def _how(cls, chosen) -> str:
    """Πώς βρέθηκε η απάντηση: η σίγουρη του BERT, επιλογή σε αβέβαιο BERT, ή άλλη
    από την πρώτη ενός σίγουρου BERT (μόνο με validator_mode=choose)."""
    if not cls.confident:
        return "pick"
    first = cls.candidates[0] if cls.candidates else None
    return "confident" if first and classifier.same_place(chosen, first) else "switch"


def _buttons(cls, settings) -> tuple:
    """(κουμπιά, κείμενο ή None) όταν δεν βρέθηκε απάντηση. Με validator_mode=choose,
    εκεί που το bot δεν θα έδειχνε επιλογές (σίγουρο BERT που απορρίφθηκε, ή αβέβαιο
    χωρίς επιλογές), κουμπιά γίνονται οι 3 πρώτες του BERT."""
    if settings.validator_mode == "choose" and not cls.error and cls.top and (cls.confident or not cls.candidates):
        return classifier.options(cls, cls.top[:3]), C.MSG[cls.lang]["suggest"]
    return classifier.options(cls), None


def _resolve(seg: Segment, pending, settings, trace: list) -> None:
    """Βήματα 5 και 6 για ένα αίτημα: BERT, Validator, case 1, case 2."""
    cls = _classify(seg.text, pending, trace)
    if not cls.confident and not cls.error and seg.alt:
        # Φλύαρο μήνυμα: δεύτερη ευκαιρία με το απόσπασμα του Welcome.
        alt = _classify(seg.alt, None, trace, " (απόσπασμα)")
        if alt.confident and alt.candidates:
            cls = alt
    seg.cls = cls

    said = seg.view or seg.text                     # οι λέξεις του πολίτη, ή η μεταγραφή τους (greeklish)
    chosen, step = validator.check_intent(said, cls, settings)
    trace.append(step)
    outside = seg.label == "out_of_scope"           # μόνο με scope=after φτάνει ως εδώ
    urgent = seg.label == "emergency"               # μόνο με emergency=answer φτάνει ως εδώ
    if chosen:
        seg.chosen, seg.how = chosen, _how(cls, chosen)
    elif outside or urgent:
        pass        # χωρίς case 1: δεν ψάχνουμε υπηρεσία με το ζόρι για ό,τι ο Security ξεχώρισε
    elif settings.reform and not cls.error:
        new, step = welcome.reform(seg.text, settings)            # case 1
        trace.append(step)
        if new:
            again = _classify(new, None, trace, " (αναδιατύπωση)")
            if again.confident and again.candidates:
                # Ο Validator κρίνει με τις λέξεις του πολίτη, όχι με την αναδιατύπωση.
                chosen, step = validator.check_intent(said, again, settings)
                trace.append(step)
                if chosen:
                    again.lang, again.aspect = cls.lang, cls.aspect
                    seg.cls, seg.chosen, seg.how, seg.reformed = again, chosen, "reform", new

    if seg.chosen:
        seg.status, seg.answer = "answered", classifier.answer(seg.chosen, seg.cls)
        if outside:
            trace.append(Step("security", "scope", "το «εκτός αρμοδιότητας» δεν ισχύει: BERT και Validator "
                                                   "βρήκαν υπηρεσία"))
        if urgent:
            trace.append(Step("security", "emergency", "112 και η υπηρεσία που βρήκαν BERT και Validator"))
    elif urgent:
        seg.status = "emergency"                    # μόνο το κείμενο για το 112
        trace.append(Step("security", "emergency", "μόνο 112: ούτε το BERT ούτε ο Validator βρήκαν υπηρεσία"))
    elif outside and not cls.error:
        seg.status, seg.options = "out_of_scope", _buttons(seg.cls, settings)[0]
        trace.append(Step("security", "scope", "εκτός αρμοδιότητας: ούτε το BERT ούτε ο Validator βρήκαν "
                                               "υπηρεσία"))
    else:                                                         # case 2
        seg.status = "error" if cls.error else "clarify"
        seg.options, text = _buttons(seg.cls, settings)
        seg.answer = text or classifier.clarify_text(seg.cls)


def run(message: str, session: Session | None = None, settings: config.Settings | None = None) -> Reply:
    settings, session = settings or config.load(), session or Session()
    t0, trace = time.monotonic(), []

    def done(text: str, options: list, lang: str, segments: list) -> Reply:
        return Reply(text, options, lang, segments, trace, round(time.monotonic() - t0, 2))

    text = (message or "").strip()
    if not text:
        return done(welcome.TEXT["el"]["empty"], [], "el", [])
    lang = welcome.language_of(text)

    # 1. Welcome: κουβέντα
    st = welcome.smalltalk(text)
    if st:
        trace.append(Step("welcome", "smalltalk", st["kind"]))
        options = [{"intent": o["intent"], "title": o["title"], "kind": o["kind"], "url": o.get("url", "")}
                   for o in st["options"]]
        return done(st["text"], options, st["lang"], [])

    # Greeklish: Security και Validator βλέπουν τη μεταγραφή σε ελληνικά (greeklish=greek)
    def view_of(s: str) -> str:
        return classifier.greek_view(s) if settings.greeklish == "greek" else s

    # 2. Security: κανόνες, στις λέξεις του πολίτη και στη μεταγραφή τους
    whole_view = view_of(text)
    broken = (security.rules(text) or (whole_view != text and security.rules(whole_view))) if settings.rules else None
    if broken:
        trace.append(Step("security", "rules", broken))
        segments = [Segment(text, label="manipulation", status="blocked")]
        return done(welcome.compose(segments, lang)[0], [], lang, segments)

    # 3. Welcome: χωρισμός
    parts, alt, step = welcome.split(text, settings)
    if step:
        trace.append(step)
    segments = [Segment(p) for p in parts]
    if len(segments) == 1:
        segments[0].alt = alt
    for seg in segments:
        seg.view = whole_view if seg.text == text else view_of(seg.text)
        if seg.view != seg.text:
            trace.append(Step("welcome", "greeklish", seg.view[:120]))

    # 4. Security: μοντέλο
    trace.append(security.check(segments, settings))
    for seg in segments:
        if security.blocks(seg.label, settings):
            seg.status = "blocked"
    allowed = [s for s in segments if s.status != "blocked"]

    # 5-6. BERT και Validator, ανά αίτημα
    pending = session.pending.take()
    for seg in allowed:
        _resolve(seg, pending if len(allowed) == 1 else None, settings, trace)

    # Αν ο χωρισμός δεν έδωσε ούτε μία απάντηση, δεν βοήθησε: το μήνυμα
    # κρίνεται ολόκληρο, όπως θα το έκρινε το σημερινό bot.
    if len(allowed) >= 2 and len(allowed) == len(segments) and not any(s.status == "answered" for s in allowed):
        trace.append(Step("welcome", "unsplit", "κανένα κομμάτι δεν λύθηκε, το μήνυμα κρίνεται ολόκληρο"))
        whole = Segment(text, view=whole_view)
        if any(s.label == "emergency" for s in allowed):
            whole.label = "emergency"
        elif all(s.label == "out_of_scope" for s in allowed):
            whole.label = "out_of_scope"
        _resolve(whole, pending, settings, trace)
        segments = allowed = [whole]

    # 8. Validator: τελικός έλεγχος (πριν από τη σύνθεση, γιατί αλλάζει το κείμενο)
    if settings.final:
        for seg in allowed:
            if seg.status == "answered":
                seg.final_ok, step = validator.check_final(seg.view or seg.text, seg.answer, settings)
                trace.append(step)

    # Ένα αίτημα που περιμένει διευκρίνιση ενώνεται με το επόμενο μήνυμα (context.py).
    waiting = [s for s in allowed if s.status == "clarify"]
    if len(waiting) == 1:
        session.pending.remember(waiting[0].cls.current, waiting[0].cls.aspect)

    # 7. Welcome: σύνθεση
    reply, options = welcome.compose(segments, lang)
    return done(reply or C.MSG[lang]["unknown"], options, lang, segments)
