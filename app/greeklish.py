"""
Greeklish → ελληνικά με μεταγραφή, όχι με μετάφραση.

ΓΙΑΤΙ: το greeklish περνούσε από το llama ως «αγγλικά» και το llama
παραισθανόταν. Το «pos plirono klisi parkarismatos» έγινε «Παρακαλώ να
μου δώσετε θέση στάθμευσης ΑμεΑ» και το bot απάντησε με 0,98, λάθος.
Επίσης κόστιζε ~1,5s.

ΜΕΘΟΔΟΣ: μεταγραφή γράμμα-γράμμα (th→θ, ps→ψ, 8→θ, 3→ξ, ou→ου). Η ασάφεια
i/h/y → ι/η/υ και o/w → ο/ω ΔΕΝ λύνεται εδώ, αλλά από τον διορθωτή: το
φωνητικό του κλειδί είναι φτιαγμένο ακριβώς γι' αυτή την ασάφεια.
«pistopoiitiko» → «πιστοποιιτικο» → «πιστοποιητικο».

ΑΝΙΧΝΕΥΣΗ: ένα λατινικό μήνυμα είναι greeklish αν, μετά τη μεταγραφή, οι
περισσότερες λέξεις του ΑΚΟΥΓΟΝΤΑΙ σαν πραγματικές ελληνικές λέξεις (ίδιο
φωνητικό κλειδί). Η διόρθωση με απόσταση επεξεργασίας δεν μετράει εδώ,
γιατί θα «έβρισκε» ελληνικές λέξεις και μέσα σε αγγλικό κείμενο.
"""
from __future__ import annotations

import re

import spellfix

ENABLED = True
MIN_KNOWN = 0.6          # ποσοστό λέξεων που πρέπει να «ακούγονται» ελληνικά

# Όχι «kh» → χ: στο συνηθισμένο greeklish το h είναι η, και το
# «lhxiarxikhs» γινόταν «ληχιαρχιχς».
_DIGRAPHS = [("th", "θ"), ("ps", "ψ"), ("ks", "ξ"), ("ch", "χ"), ("ou", "ου")]

# Συχνές αγγλικές λέξεις που δεν γράφονται ποτέ σε greeklish. Χωρίς αυτές,
# σύντομα αγγλικά περνούσαν για greeklish: στο «can I pay online?» το
# «καν» και το «παυ» ακούγονται ελληνικά. ΟΧΙ «i», «to», «me», «na»: αυτά
# είναι κοινά και στο greeklish («den anavei i lampa»).
_ENGLISH = {"the", "is", "are", "was", "can", "could", "you", "your", "my", "how",
            "what", "where", "when", "why", "which", "who", "does", "do", "did",
            "of", "for", "in", "on", "at", "with", "need", "want", "please", "and",
            "there", "have", "has", "it", "this", "that", "get", "pay", "bill",
            "street", "from", "will", "would", "an", "be", "i'm", "can't", "don't"}
# Σύντομες λέξεις: ο διορθωτής δεν αγγίζει λέξεις κάτω από 4 γράμματα,
# οπότε το «i lampa» έμενε «ι λαμπα» (0,38 αντί για σωστό). Μόνο
# μονοσήμαντες: το «ti» (τι / τη) μένει όπως είναι.
_SHORT = {"ι": "η", "μι": "μη", "πος": "πως", "στι": "στη", "στιν": "στην",
          "τιν": "την", "θν": "την", "κε": "και",
          # «sth / sthn / ths»: το h είναι η, όχι μέρος του th → θ
          "σθ": "στη", "σθν": "στην", "σθς": "στης", "θς": "της"}
_SINGLE = dict(zip("abcdefghijklmnopqrstuvwxyz",
                   "αβκδεφγηιζκλμνοπκρστυβωχυζ"))
_SINGLE.update({"8": "θ", "3": "ξ"})


def transliterate(text: str) -> str:
    out = []
    for token in re.split(r"(\s+)", str(text)):
        # Μόνο λέξεις που περιέχουν λατινικά. Αριθμοί όπως «20%» μένουν ίδιοι:
        # το 8 και το 3 γίνονται γράμματα μόνο μέσα σε λέξη («8elw», «peta3w»).
        if not re.search(r"[A-Za-z]", token):
            out.append(token)
            continue
        s, i, w = token.lower(), 0, []
        while i < len(s):
            pair = s[i:i + 2]
            hit = next((g for d, g in _DIGRAPHS if pair == d), None)
            if hit:
                w.append(hit); i += 2
                continue
            w.append(_SINGLE.get(s[i], s[i])); i += 1
        word = re.sub(r"σ(?=$|[^\wα-ω])", "ς", "".join(w))
        core = re.match(r"^([α-ως]+)(.*)$", word)
        if core and core.group(1) in _SHORT:
            word = _SHORT[core.group(1)] + core.group(2)
        out.append(word)
    return "".join(out)


def _sounds_greek(word: str) -> bool:
    s = spellfix._strip(word)
    return (s in spellfix._VOCAB or s in spellfix._GENERAL
            or spellfix.phonetic(s) in spellfix._INDEX)


def to_greek(text: str) -> str | None:
    """Ελληνικό κείμενο αν το μήνυμα είναι greeklish, αλλιώς None."""
    if not ENABLED:
        return None
    if set(re.findall(r"[a-z']+", str(text).lower())) & _ENGLISH:
        return None
    greek = transliterate(text)
    words = [w for w in re.findall(r"[α-ωά-ώϊϋΐΰς]+", greek) if len(w) >= 2]
    if not words:
        return None
    known = sum(_sounds_greek(w) for w in words) / len(words)
    return greek if known >= MIN_KNOWN else None
