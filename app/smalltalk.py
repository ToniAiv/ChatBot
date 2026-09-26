"""
Κουβέντα: χαιρετισμοί, «ευχαριστώ», «τι μπορείς να κάνεις», «ποιος είσαι».

ΓΙΑΤΙ: όλα έφταναν στο μοντέλο υπηρεσιών, που δεν έχει κλάση γι' αυτά.
Το «γεια σου» έβγαινε με 0,16 και ο πολίτης έπαιρνε «Δεν κατάλαβα την
ερώτηση»: κακή πρώτη εντύπωση, στο πρώτο κιόλας μήνυμα.

ΚΑΝΟΝΑΣ ΑΣΦΑΛΕΙΑΣ: πιάνει μόνο αν ΟΛΟ το μήνυμα είναι κουβέντα. Το
«καλημέρα, θέλω βεβαίωση κατοικίας» πηγαίνει κανονικά στο μοντέλο, αφού
ένα χαμένο αίτημα κοστίζει περισσότερο από έναν αναπάντητο χαιρετισμό.
Μετρημένο: 0 ενεργοποιήσεις στις ~24.000 προτάσεις dataset + test set.

Κανόνες και όχι μοντέλο: τα είδη είναι λίγα και κλειστά, και κάθε
απόφαση πρέπει να εξηγείται.
"""
from __future__ import annotations

import re
import unicodedata

ENABLED = True

# Παραδείγματα που δείχνονται ως κουμπιά. Μόνο verified γραμμές του πίνακα:
# το πρώτο πάτημα του πολίτη πρέπει να δώσει σίγουρα σωστή απάντηση.
EXAMPLES = ["βεβαιωση_κατοικιας", "ογκοδεματα",
            "ηλεκτρονικη_εξοφληση_λογαριασμου_δευαη"]

_POLITE = r"(?: (?:σας|σου|παιδια|φιλε|bot|μποτ|there|you|all))*"

# Κάθε μοτίβο πρέπει να ταιριάζει σε ΟΛΟ το κανονικοποιημένο μήνυμα.
_PATTERNS = {
    "greeting": [
        r"(?:γεια|γειασου|γειασας|καλημερα|καλησπερα|χαιρετε|χαιρετω|χαιρετισματα"
        r"|hi|hello|hey|yo|geia|gia|good (?:morning|afternoon|evening))" + _POLITE,
        r"(?:τι κανεις|τι κανετε|πως εισαι|πως ειστε|ολα καλα|how are you)",
    ],
    "help": [
        r"(?:τι (?:μπορεις|μπορειτε) να (?:κανεις|κανετε)|σε τι (?:μπορεις|μπορειτε) να (?:με )?βοηθη\w*"
        r"|(?:μπορεις|μπορειτε) να (?:με )?βοηθη\w*|τι (?:ξερεις|ξερετε)|πως (?:λειτουργεις|δουλευεις|δουλευει αυτο)"
        r"|τι κανει αυτο|βοηθεια|help|what can you do|how does this work|can you help(?: me)?)",
    ],
    "who": [
        r"(?:ποιος|τι) (?:εισαι|ειστε)",
        r"(?:εισαι|ειστε) (?:ανθρωπος|ρομποτ|μηχανη|bot|μποτ|πραγματικος)",
        r"μιλαω με (?:ανθρωπο|ρομποτ|μηχανη|bot|μποτ|υπολογιστη)",
        r"(?:θελω (?:να )?)?(?:μιλησω|μιλαω) (?:με )?(?:ανθρωπο|υπαλληλο|καποιον|πραγματικο ανθρωπο)",
        r"(?:who are you|are you (?:a )?(?:bot|human|robot|real person))",
    ],
    "thanks": [
        r"(?:(?:οκ|ok|οκει|εντα?ξει|τελεια|ωραια|super|σουπερ) )?(?:σε |σας )?"
        r"(?:ευχαριστω|ευχαριστουμε|ευχαριστω πολυ|thanks|thank you|thx|merci)"
        r"(?: (?:πολυ|πααρα πολυ|παρα πολυ|σας|σου|a lot|very much))*",
        r"(?:οκ|ok|οκει|εντα?ξει|τελεια|ωραια|super|σουπερ|μπραβο)",
    ],
    "bye": [
        r"(?:αντιο|αντιο σας|καλο βραδυ|καληνυχτα|καλη συνεχεια|τα λεμε|bye|goodbye|see you)" + _POLITE,
    ],
}
_COMPILED = {k: [re.compile(p) for p in ps] for k, ps in _PATTERNS.items()}

REPLY = {
    "el": {
        "greeting": "Γεια σας! Είμαι ο εικονικός βοηθός του Δήμου Ηρακλείου. "
                    "Γράψτε μου με δικά σας λόγια τι χρειάζεστε, για παράδειγμα:",
        "help": "Σας βοηθάω να βρείτε την υπηρεσία του Δήμου που χρειάζεστε: "
                "πιστοποιητικά, βεβαιώσεις, αιτήματα για καθαριότητα, δρόμους και "
                "φωτισμό, πληρωμές. Γράψτε τι θέλετε με δικά σας λόγια, για παράδειγμα:",
        "who": "Είμαι αυτόματος εικονικός βοηθός, όχι υπάλληλος. Αν θέλετε να "
               "μιλήσετε με άνθρωπο, καλέστε το τηλεφωνικό κέντρο του Δήμου στο {phone}.",
        "thanks": "Παρακαλώ! Αν χρειάζεστε κάτι άλλο, γράψτε μου.",
        "bye": "Καλή συνέχεια!",
    },
    "en": {
        "greeting": "Hello! I am the virtual assistant of the Municipality of Heraklion. "
                    "Tell me in your own words what you need, for example:",
        "help": "I help you find the Municipality service you need: certificates, "
                "requests about cleaning, roads and lighting, payments. Describe what "
                "you need in your own words, for example:",
        "who": "I am an automated virtual assistant, not a member of staff. To speak "
               "to a person, call the Municipality's switchboard at {phone}.",
        "thanks": "You're welcome! Let me know if you need anything else.",
        "bye": "Goodbye!",
    },
}
WITH_EXAMPLES = {"greeting", "help"}


def _normalize(text: str) -> str:
    s = unicodedata.normalize("NFD", str(text))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
    s = re.sub(r"[^\w\s]", " ", s)          # στίξη, emoji
    return re.sub(r"\s+", " ", s).strip()


def detect(text: str) -> str | None:
    """Το είδος της κουβέντας, ή None αν το μήνυμα είναι (και) αίτημα."""
    if not ENABLED:
        return None
    s = _normalize(text)
    if not s or len(s.split()) > 8:
        return None
    for kind, rxs in _COMPILED.items():
        if any(rx.fullmatch(s) for rx in rxs):
            return kind
    # «γεια σου, ευχαριστώ» ή «καλημέρα, τι μπορείς να κάνεις;»: κάθε
    # κομμάτι μόνο του είναι κουβέντα. Κρατάμε το είδος του τελευταίου.
    parts = [p for p in re.split(r"[,.!;;\n]+", str(text)) if p.strip()]
    if len(parts) > 1:
        kinds = [detect(p) for p in parts]
        if all(kinds):
            return kinds[-1]
    return None


def reply(kind: str, lang: str, phone: str) -> str:
    return REPLY[lang][kind].format(phone=phone)
