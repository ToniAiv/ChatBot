"""Εισαγωγή ρεαλιστικών ελληνικών ορθογραφικών λαθών.

Η κατανομή δεν είναι αυθαίρετη: μετρήθηκε ότι 9/9 από τα λάθη που
σπάνε το μοντέλο είναι ΦΩΝΗΤΙΚΑ (ι/η/υ/ει/οι, ο/ω, ε/αι) — εκεί κάνουν
λάθος οι φυσικοί ομιλητές. Τα μηχανικά λάθη πληκτρολογίου είναι
σπανιότερα, αλλά μπαίνουν για να μην υπερπροσαρμοστεί η λύση.
"""
import random, re, unicodedata

PHON = [
    (r"οι", "ι"), (r"ει", "ι"), (r"υ", "ι"), (r"η", "ι"), (r"ι", "η"),
    (r"αι", "ε"), (r"ε", "αι"), (r"ω", "ο"), (r"ο", "ω"),
    (r"αυ", "αφ"), (r"ευ", "εφ"),
]

def _strip(s):
    s = unicodedata.normalize("NFD", str(s))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")

def phonetic(word, rng):
    w = _strip(word)
    cands = [(p, r) for p, r in PHON if re.search(p, w)]
    if not cands:
        return None
    p, r = rng.choice(cands)
    hits = [m.start() for m in re.finditer(p, w)]
    i = rng.choice(hits)
    return w[:i] + r + w[i + len(re.match(p, w[i:]).group(0)):]

def mechanical(word, rng):
    w = _strip(word)
    if len(w) < 5:
        return None
    kind = rng.choice(["drop", "swap", "double"])
    i = rng.randrange(1, len(w) - 1)
    if kind == "drop":
        return w[:i] + w[i+1:]
    if kind == "swap":
        return w[:i] + w[i+1] + w[i] + w[i+2:]
    return w[:i] + w[i] + w[i:]

def corrupt(text, rng, n_words=1, p_phonetic=0.7):
    """Χαλάει n_words λέξεις ≥5 γραμμάτων."""
    words = text.split()
    idx = [i for i, w in enumerate(words) if len(_strip(w)) >= 5]
    if not idx:
        return text
    rng.shuffle(idx)
    changed = 0
    for i in idx:
        fn = phonetic if rng.random() < p_phonetic else mechanical
        out = fn(words[i], rng)
        if out and out != _strip(words[i]):
            words[i] = out
            changed += 1
            if changed >= n_words:
                break
    return " ".join(words)
