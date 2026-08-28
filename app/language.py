"""
Υποστήριξη αγγλικών: ανίχνευση γλώσσας + μετάφραση προς τα ελληνικά.

ΓΙΑΤΙ ΜΕΤΑΦΡΑΣΗ ΚΑΙ ΟΧΙ ΠΟΛΥΓΛΩΣΣΟ ΜΟΝΤΕΛΟ
Το intent classifier είναι bert-base-greek-uncased, εκπαιδευμένο σε
21.802 ελληνικές προτάσεις. Σε αγγλικά δίνει 0,12-0,24 βεβαιότητα —
θόρυβο. Πολύγλωσσο μοντέλο θα απαιτούσε μεταφρασμένο dataset και
επανεκπαίδευση: εβδομάδες. Η μετάφραση της ερώτησης αφήνει την
ελληνική διαδρομή ανέπαφη, άρα όλες οι μετρήσεις της παραμένουν έγκυρες.

ΜΕΤΡΗΜΕΝΟ (20 αγγλικές ερωτήσεις με γνωστή σωστή απάντηση):
    qwen2.5:7b              2/20   (10%)
    llama3.1                12/20  (60%)
    llama3.1 + γλωσσάρι     14/20  (70%)
Η ελληνική διαδρομή στο ίδιο είδος ερωτήσεων: ~79%.

Το γλωσσάρι ήταν η καθοριστική διαφορά. Τα λάθη δεν ήταν γραμματικά
αλλά ΟΡΟΛΟΓΙΑΣ: «street market» → «λαϊσμός», «wheelchair» → «καρέκλα»,
«streetlights» → «φωτιστικές σημαίες». Αν προστεθούν νέες υπηρεσίες με
ιδιαίτερους όρους, ο πρώτος τόπος να κοιτάξεις είναι το GLOSSARY.

ΠΡΟΣΟΧΗ ΣΤΟ ΣΥΝΤΟΝΙΣΜΟ ΤΟΥ ΓΛΩΣΣΑΡΙΟΥ: κάθε αλλαγή στο prompt αλλάζει
ΟΛΕΣ τις μεταφράσεις, όχι μόνο τις στοχευμένες. Προσθέτοντας 7 όρους
για να διορθωθούν 3 αποτυχίες, το σκορ πήγε 16/20 → 15/20: κέρδισε τον
σηματοδότη, έχασε τα ποντίκια και τον πολιτικό γάμο. Με 20 δείγματα
αυτό είναι θόρυβος, όχι σήμα. Πριν ξανασυντονιστεί το γλωσσάρι χρειάζεται
αγγλικό test set ~200 προτάσεων, αλλιώς γίνεται overfitting σε λίγα
παραδείγματα.
"""
import re
import unicodedata

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
TRANSLATE_MODEL = "llama3.1"

# Διακόπτης: False → το bot συμπεριφέρεται ακριβώς όπως πριν την
# προσθήκη αγγλικών. Υπάρχει για να μπορεί να απενεργοποιηθεί χωρίς
# να αγγίξει κανείς την ελληνική διαδρομή.
ENGLISH_SUPPORT = True

GLOSSARY = """Use these official Greek municipal terms:
street market = λαϊκή αγορά | wheelchair = αναπηρικό αμαξίδιο
disabled parking = θέση στάθμευσης ΑμεΑ | streetlight = δημοτικός φωτισμός
pavement/sidewalk = πεζοδρόμιο | bin/skip = κάδος απορριμμάτων
bulky waste = ογκώδη αντικείμενα | stray dogs = αδέσποτα ζώα
rats/pest control = μυοκτονία | mosquitoes = απεντόμωση
civil marriage / town hall wedding = πολιτικός γάμος
nursery/kindergarten = παιδικός σταθμός | grave = τάφος | exhumation = εκταφή
birth registration copy = αντίγραφο ληξιαρχικής πράξης γέννησης
family status certificate = πιστοποιητικό οικογενειακής κατάστασης
fine/ticket = πρόστιμο | appeal/objection = ένσταση
water bill = λογαριασμός ύδρευσης | instalments = δόσεις
building permit = οικοδομική άδεια | town planning = πολεοδομία
abandoned vehicle = εγκαταλελειμμένο όχημα | pothole = λακκούβα
resident parking permit = σήμα στάθμευσης μονίμου κατοίκου
traffic light = φωτεινός σηματοδότης | street lighting out = βλάβη δημοτικού φωτισμού
pay online = ηλεκτρονική εξόφληση | road surface damage = βλάβη οδοστρώματος
tree pruning = κλάδεμα δένδρων | rubbish collection = αποκομιδή απορριμμάτων
certificate of residence = βεβαίωση μόνιμης κατοικίας"""

PROMPT = (
    "Translate this citizen's request from English into Greek, as a Greek "
    "citizen would phrase it to their municipality.\n\n" + GLOSSARY +
    "\n\nOutput ONLY the Greek translation, one sentence, nothing else.\n\n"
    "English: {q}"
)

_GREEK = re.compile(r"[Ͱ-Ͽἀ-῿]")
_LATIN = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    """
    'el' ή 'en', με βάση ποιο αλφάβητο κυριαρχεί.

    Δεν χρησιμοποιείται βιβλιοθήκη ανίχνευσης γλώσσας: το ερώτημα εδώ
    δεν είναι «ποια από 100 γλώσσες» αλλά «ελληνικό ή λατινικό
    αλφάβητο», που κρίνεται με μέτρημα χαρακτήρων και δεν αστοχεί.
    Greeklish («thelo pistopoiitiko») μετριέται ως αγγλικά — σωστά,
    γιατί ούτε αυτά τα καταλαβαίνει το ελληνικό μοντέλο.
    """
    text = unicodedata.normalize("NFC", str(text))
    gr, la = len(_GREEK.findall(text)), len(_LATIN.findall(text))
    return "en" if la > gr else "el"


class TranslationUnavailable(RuntimeError):
    """Το Ollama δεν απάντησε. Ο καλών αποφασίζει τι λέει στον πολίτη."""


def translate_to_greek(text: str) -> str:
    """Αγγλικά → ελληνικά. Κοστίζει ~1,8s· καλείται ΜΟΝΟ για αγγλικά."""
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": TRANSLATE_MODEL,
            "prompt": PROMPT.format(q=text),
            "stream": False,
            "options": {"temperature": 0},
        }, timeout=120)
        r.raise_for_status()
        out = r.json()["response"].strip().strip('"').split("\n")[0].strip()
    except Exception as exc:
        raise TranslationUnavailable(str(exc)) from exc

    if not out or not _GREEK.search(out):
        # Το μοντέλο επέστρεψε κάτι που δεν είναι ελληνικά — καλύτερα να
        # το πούμε παρά να τροφοδοτήσουμε σκουπίδια στον classifier.
        raise TranslationUnavailable(f"μη ελληνική έξοδος: {out[:60]!r}")
    return out
