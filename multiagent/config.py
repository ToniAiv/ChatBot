"""
Ρυθμίσεις: ποιο μοντέλο και ποιος server για κάθε ρόλο.

Ο κώδικας των agents δεν ξέρει πού τρέχει το μοντέλο. Κάθε ρόλος παίρνει από
εδώ ένα όνομα μοντέλου και μια διεύθυνση Ollama, οπότε το ίδιο pipeline
τρέχει στο laptop ή σε άλλον server χωρίς αλλαγή στον κώδικα.

    MA_PROFILE=qwen4b                 έτοιμη ρύθμιση (PROFILES)
    MA_MODEL_SECURITY=nimble          άλλο μοντέλο για έναν ρόλο
    MA_URL_SECURITY=http://host:11434 άλλος server για έναν ρόλο
    MA_OLLAMA_URL=http://host:11434   άλλος server για όλους
    MA_RULES=0  MA_SPLIT=0  MA_REFORM=0  MA_FINAL=0   κλείνουν ένα βήμα
    MA_SCOPE=before | after           πότε μετρά το «εκτός αρμοδιότητας» του Security (Settings.scope)
    MA_DECISION_LANG=el | en          γλώσσα των ερωτήσεων προς decision model
    MA_EMERGENCY=block | answer       «επείγον»: μόνο το 112, ή το 112 και η απάντηση του Δήμου
    MA_SPLIT_LANGS=el | all           χωρισμός σε αιτήματα μόνο στα ελληνικά, ή και σε αγγλικά/greeklish
    MA_VALIDATOR_MODE=fits | choose   ο Validator κρίνει τους υποψήφιους του bot, ή διαλέγει στις 5 πρώτες του BERT
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

OLLAMA_URL = os.environ.get("MA_OLLAMA_URL", "http://localhost:11434").rstrip("/")

# Οι πέντε κλήσεις μοντέλου του σχήματος:
#   split      Welcome: χωρισμός του μηνύματος σε αιτήματα
#   security   Security: εκτός αρμοδιότητας, χειραγώγηση, προσβολές, επείγον
#   validator  Validator μετά το BERT: ταιριάζει η υπηρεσία στο αίτημα;
#   reform     Welcome, case 1: αναδιατύπωση του αιτήματος
#   final      Validator πριν τον πολίτη: απαντά η απάντηση στο αίτημα;
ROLES = ("split", "security", "validator", "reform", "final")

PROFILES = {
    # Κρίσεις σε decision model, κείμενο στο μικρό Qwen.
    "tev1": {"split": "qwen3:4b", "security": "tev1:4b", "validator": "tev1:4b",
             "reform": "qwen3:4b", "final": "tev1:4b"},
    "qwen4b": dict.fromkeys(ROLES, "qwen3:4b"),
    "qwen14b": dict.fromkeys(ROLES, "qwen3:14b"),
}

# Μοντέλα που απαντούν σε ερωτήσεις με πιθανότητα (/v1/systemone) και δεν
# γράφουν κείμενο. Όλα τα άλλα καλούνται ως LLM με έξοδο JSON (/api/chat).
DECISION_FAMILIES = ("tev1", "nimble", "clef")


def is_decision_model(model: str) -> bool:
    return model.split("/")[-1].split(":")[0].startswith(DECISION_FAMILIES)


def _flag(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() not in ("0", "false", "off", "no")


@dataclass
class Settings:
    profile: str = "tev1"
    models: dict = field(default_factory=dict)
    urls: dict = field(default_factory=dict)
    rules: bool = True          # Security: κανόνες στον κώδικα πριν από το μοντέλο
    split: bool = True          # Welcome: χωρισμός σε αιτήματα
    reform: bool = True         # case 1 του σχήματος
    final: bool = True          # τελικός Validator
    # «Εκτός αρμοδιότητας» από τον Security:
    #   before  κόβει το αίτημα πριν από το BERT, όπως στο σχήμα της 10/11
    #   after   το αίτημα περνά στο BERT· η ετικέτα μετρά μόνο αν ούτε το BERT ούτε
    #           ο Validator βρουν υπηρεσία
    # Προεπιλογή after με αγγλικές ερωτήσεις: η καλύτερη ρύθμιση στα δύο holdout,
    # 59 σωστές στις 93 έναντι 27 με before (reports/multiagent.md, 2η μέτρηση).
    scope: str = "after"
    decision_lang: str = "en"   # γλώσσα των ερωτήσεων προς decision model
    # «Επείγον» από τον Security:
    #   block   μόνο το κείμενο για το 112, χωρίς ταξινόμηση
    #   answer  το κείμενο για το 112 και, αν BERT και Validator βρουν υπηρεσία, η απάντησή της
    # Ο έλεγχος «μήπως έχει δύο αιτήματα» πριν από το μοντέλο του χωρισμού:
    #   el   μόνο ελληνικές λέξεις (και, επίσης…)· τα αγγλικά και τα greeklish δεν χωρίζονται
    #   all  και αγγλικές λέξεις και greeklish (and, also, kai, episis…)
    # Προεπιλογές answer και all από τους κανόνες της 3ης μέτρησης (holdout3): ίδιο
    # σκορ με block / el, 2 σύνθετα μισά σωστά επιπλέον (reports/multiagent.md).
    emergency: str = "answer"
    split_langs: str = "all"
    # Ο Validator μετά το BERT:
    #   fits    κρίνει τους υποψήφιους του bot: τη σίγουρη απάντηση («ταιριάζει;») ή τις
    #           επιλογές που θα έδειχνε το bot
    #   choose  διαλέγει πάντα ανάμεσα στις 5 πρώτες υπηρεσίες του BERT (ή «καμία»)
    validator_mode: str = "fits"
    threshold: float = 0.5      # πιθανότητα πάνω από την οποία ο Validator εγκρίνει
    timeout: int = 120          # δευτερόλεπτα ανά κλήση μοντέλου

    def model(self, role: str) -> str:
        return self.models[role]

    def url(self, role: str) -> str:
        return self.urls.get(role, OLLAMA_URL)


def load(profile: str | None = None, **overrides) -> Settings:
    profile = profile or os.environ.get("MA_PROFILE", "tev1")
    if profile not in PROFILES:
        raise ValueError(f"Άγνωστη ρύθμιση «{profile}». Υπάρχουν: {', '.join(PROFILES)}")
    models, urls = dict(PROFILES[profile]), {}
    for role in ROLES:
        if os.environ.get(f"MA_MODEL_{role.upper()}"):
            models[role] = os.environ[f"MA_MODEL_{role.upper()}"]
        if os.environ.get(f"MA_URL_{role.upper()}"):
            urls[role] = os.environ[f"MA_URL_{role.upper()}"].rstrip("/")
    s = Settings(profile=profile, models=models, urls=urls, rules=_flag("MA_RULES"),
                 split=_flag("MA_SPLIT"), reform=_flag("MA_REFORM"), final=_flag("MA_FINAL"))
    for name, key in (("MA_SCOPE", "scope"), ("MA_DECISION_LANG", "decision_lang"),
                      ("MA_EMERGENCY", "emergency"), ("MA_SPLIT_LANGS", "split_langs"),
                      ("MA_VALIDATOR_MODE", "validator_mode")):
        if os.environ.get(name):
            setattr(s, key, os.environ[name].strip().lower())
    for k, v in overrides.items():
        if k == "models":                       # αλλάζει μόνο τους ρόλους που δίνονται
            s.models.update(v)
        else:
            setattr(s, k, v)
    for name, key, allowed in (("MA_SCOPE", "scope", ("before", "after")),
                               ("MA_DECISION_LANG", "decision_lang", ("el", "en")),
                               ("MA_EMERGENCY", "emergency", ("block", "answer")),
                               ("MA_SPLIT_LANGS", "split_langs", ("el", "all")),
                               ("MA_VALIDATOR_MODE", "validator_mode", ("fits", "choose"))):
        if getattr(s, key) not in allowed:
            raise ValueError(f"{name}: «{getattr(s, key)}»· γίνεται {' ή '.join(allowed)}")
    return s
