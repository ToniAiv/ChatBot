# Multi-agent σύστημα

Το σχήμα της ομάδας (10/11/2026), δίπλα στο σημερινό bot. Το `app/` δεν αλλάζει.

    πολίτης → Welcome → Security → BERT → Validator → Welcome → Validator → πολίτης

Κάθε agent είναι μια κλάση που φτιάχνεται χωριστά από το δικό της αρχείο
`.yaml` στο `configs/` (Hydra, όπως στο repo Feathers του Νικόλα).

Μετρήσεις και ευρήματα: `reports/multiagent.md`.

## Εκκίνηση

Θέλει το Ollama ανοιχτό, με τα `tev1:4b` και `qwen3:4b` (`ollama pull`).

```bash
./run.sh agents                                  # συζήτηση στο τερματικό, με το ίχνος κάθε μηνύματος
./run.sh agents 'once="θέλω βεβαίωση κατοικίας και να δηλώσω μια λακκούβα"'
./run.sh agents experiment=tev1-en-clean         # μια ρύθμιση που έχουμε μετρήσει
./run.sh agents security.scope=before            # αλλαγή μίας τιμής
./run.sh agents model@security.model=qwen3_4b    # άλλο μοντέλο για τον Security
./run.sh agents print_config=true                # τύπωσε όλο το config στην αρχή
```

Το `./run.sh agents` τρέχει το `python -m multiagent.chat`. Ελληνικό κείμενο σε
όρισμα θέλει διπλά εισαγωγικά μέσα σε μονά (`'once="…"'`), αλλιώς το Hydra δεν
το δέχεται. Κάθε εκτέλεση κρατά το πλήρες config της στο
`outputs/chat/<ημερομηνία_ώρα>/.hydra/`.

Από Python:

```python
from multiagent import build
pipe = build.pipeline()                          # ή build.pipeline("tev1-en-clean")
reply = pipe.run("θέλω βεβαίωση κατοικίας")
reply.text, reply.options, reply.trace
```

## Τι κάνει κάθε agent

| Αρχείο | Κλάση | Δουλειά | Μοντέλο |
|---|---|---|---|
| `welcome.py` | `Welcome` | χαιρετισμοί· χωρίζει το μήνυμα σε αιτήματα· αναδιατυπώνει (case 1)· ενώνει τις απαντήσεις | LLM για χωρισμό και αναδιατύπωση |
| `security.py` | `Security` | κανόνες σε όλο το μήνυμα, μετά ετικέτα ανά αίτημα: αίτημα, εκτός αρμοδιότητας, χειραγώγηση, προσβολή, επείγον | decision model ή LLM |
| `classifier.py` | `Classifier` | το pipeline του σημερινού bot: υποψήφιοι προορισμοί και έτοιμα κείμενα | κανένα LLM |
| `validator.py` | `Validator` | μετά το BERT: ποια υπηρεσία ζητά ο πολίτης; πριν τον πολίτη: απαντά το κείμενο στο αίτημα; | decision model ή LLM |
| `pipeline.py` | `Pipeline` | η σειρά των βημάτων και οι δύο βρόχοι (case 1, case 2)· παίρνει έτοιμους τους agents | — |
| `llm.py` | `OllamaModel` | ένα μοντέλο του Ollama για έναν agent: όνομα, server, χρονικό όριο | — |

Τα κείμενα των απαντήσεων και τα στοιχεία επικοινωνίας έρχονται πάντα από
τους πίνακες του bot. Κανένα μοντέλο δεν γράφει τηλέφωνο ή σύνδεσμο.

## Ρυθμίσεις: τα αρχεία του `configs/`

```text
configs/
  multiagent.yaml          ποια κομμάτια φτιάχνονται (λίστα defaults) και κοινές τιμές
  hydra/default.yaml       φάκελος ανά εκτέλεση: outputs/chat/<ημερομηνία_ώρα>/
  model/                   ένα αρχείο ανά μοντέλο: tev1_4b, qwen3_4b, qwen3_14b
  welcome/default.yaml     Welcome
  security/default.yaml    Security
  classifier/bert.yaml     το BERT του bot
  validator/default.yaml   Validator
  pipeline/default.yaml    η σειρά των βημάτων
  experiment/              μία ρύθμιση ανά μέτρηση (tev1 … tev1-en-clean-hint)
```

Κάθε αρχείο έχει `_target_`, την κλάση που φτιάχνει. Το Hydra φτιάχνει πρώτα
τα μοντέλα, μετά τους agents που τα παίρνουν, και τέλος το `Pipeline` που
παίρνει έτοιμους τους agents. Ποιο μοντέλο παίρνει κάθε agent το λέει η λίστα
`defaults` του `multiagent.yaml` (π.χ. `model@security.model: tev1_4b`).

| Στο `.yaml` | Παλιά μεταβλητή | Τι κάνει |
|---|---|---|
| `model@<agent>.<μοντέλο>` | `MA_PROFILE`, `MA_MODEL_<ΡΟΛΟΣ>` | ποιο μοντέλο παίρνει κάθε agent (`configs/model/`) |
| `ollama_url`, `<agent>.model.url` | `MA_OLLAMA_URL`, `MA_URL_<ΡΟΛΟΣ>` | ο server του Ollama, για όλους ή για ένα μοντέλο |
| `security.rules` | `MA_RULES` | κανόνες του Security πριν από το μοντέλο |
| `welcome.split` | `MA_SPLIT` | χωρισμός σε αιτήματα |
| `welcome.reform` | `MA_REFORM` | αναδιατύπωση (case 1) |
| `validator.final` | `MA_FINAL` | τελικός Validator |
| `security.scope` | `MA_SCOPE` | `after` (προεπιλογή): το αίτημα περνά στο BERT και το «εκτός αρμοδιότητας» μετρά μόνο αν ο Validator δεν εγκρίνει υπηρεσία· `before`: κόβει πριν από το BERT, όπως στο σχήμα |
| `decision_lang` | `MA_DECISION_LANG` | `en` (προεπιλογή) ή `el`: γλώσσα των ερωτήσεων προς το decision model |
| `security.emergency` | `MA_EMERGENCY` | `answer` (προεπιλογή): το 112 και από κάτω η υπηρεσία του Δήμου, αν βρεθεί· `block`: μόνο το 112 |
| `welcome.split_langs` | `MA_SPLIT_LANGS` | `all` (προεπιλογή): χωρισμός και σε αγγλικά και greeklish· `el`: μόνο σε ελληνικά |
| `validator.mode` | `MA_VALIDATOR_MODE` | `choose` (προεπιλογή): διαλέγει στις 5 πρώτες του BERT· `fits`: κρίνει μόνο τους υποψήφιους του bot |
| `pipeline.greeklish` | `MA_GREEKLISH` | `greek` (προεπιλογή): Security και Validator βλέπουν τα greeklish σε ελληνικά· `raw`: όπως γράφτηκαν |
| `validator.text` | `MA_VALIDATOR_TEXT` | `plain`: τίτλος και αρχή της περιγραφής· `clean`: χωρίς τυποποιημένες φράσεις, θέματα για τα τμήματα |
| `validator.hint` | `MA_VALIDATOR_HINT` | ο Validator μαθαίνει την πρώτη επιλογή του BERT και τη βεβαιότητά της |
| `validator.threshold` | — | πιθανότητα από την οποία ο Validator εγκρίνει (0,5) |

Οι προεπιλογές βγήκαν από τους κανόνες απόφασης των μετρήσεων (`reports/multiagent.md`).
- Στο holdout4 (κυρίως ελληνικά και αγγλικά) το multi-agent βρήκε 35 σωστές στις 66, έναντι 30 του bot, με 6 σίγουρα λάθη έναντι 8.
- Στο holdout5 (κυρίως greeklish), με τη μεταγραφή των greeklish, βρήκε 26 στις 58, έναντι 32 του bot.
- Θέλει 6–12 s ανά μήνυμα.

### Ο παλιός τρόπος: μεταβλητές `MA_*`

Το `python -m multiagent.cli` και το `pipeline.run(...)` διαβάζουν ακόμα τις
μεταβλητές της δεύτερης στήλης (`config.py`). Φτιάχνουν τις ίδιες κλάσεις με τις
ίδιες τιμές (`from_settings`), άρα δίνουν την ίδια συμπεριφορά. Θα φύγουν όταν
το αποφασίσει η ομάδα.

## Μέτρηση

```bash
venv/bin/python -m multiagent.evaluate security     # ο Security μόνος του
venv/bin/python -m multiagent.evaluate validator    # ο Validator μόνος του
venv/bin/python -m multiagent.evaluate pipeline     # όλη η ροή, δίπλα στο σημερινό bot
venv/bin/python -m multiagent.evaluate summary      # πίνακες
venv/bin/python -m multiagent.evaluate metrics --configs tev1-en-choose-gr,<νέα>   # reports/metrics.txt: bot δίπλα στις ρυθμίσεις
venv/bin/python -m multiagent.evaluate pipeline --configs tev1-after --sets holdout,holdout2
```

Τα ονόματα του `--configs` είναι αρχεία του `configs/experiment/`, ίδια με τη
στήλη `config` των αρχείων αποτελεσμάτων. Κάθε αρχείο γράφει όλους τους
διακόπτες της μέτρησης, ώστε να μένει ίδιο κι αν αλλάξουν οι προεπιλογές.
Μια νέα ρύθμιση γράφεται πρώτα ως `configs/experiment/<όνομα>.yaml` και γίνεται
commit μαζί με τις προβλέψεις, πριν τρέξει μοντέλο.

Τα σετ του `sets/` γράφτηκαν πριν τρέξει μοντέλο και εξαιρούνται από την
εκπαίδευση (`scripts/prepare_dataset.py`).

## Τι δεν έχει γίνει ακόμα

- Δεν έχει συνδεθεί με το παράθυρο ή το web demo.
- Η μνήμη υπηρεσίας του bot (ερωτήσεις συνέχειας όπως «τι χαρτιά θέλει;») δεν περνά από εδώ.
- Ο Planner του σχήματος Planner–Observer δεν υπάρχει: τη σειρά την ορίζει ο κώδικας (`Pipeline`).
