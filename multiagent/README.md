# Multi-agent σύστημα

Το σχήμα της ομάδας (10/11/2026), δίπλα στο σημερινό bot. Το `app/` δεν αλλάζει.

    πολίτης → Welcome → Security → BERT → Validator → Welcome → Validator → πολίτης

Μετρήσεις και ευρήματα: `reports/multiagent.md`.

## Εκκίνηση

Θέλει το Ollama ανοιχτό, με τα `tev1:4b` και `qwen3:4b` (`ollama pull`).

```bash
./run.sh agents                              # συζήτηση στο τερματικό, με το ίχνος κάθε μηνύματος
./run.sh agents --once "θέλω βεβαίωση κατοικίας και να δηλώσω μια λακκούβα"
./run.sh agents --profile qwen4b             # άλλη ρύθμιση μοντέλων
```

Από Python:

```python
from multiagent import pipeline
reply = pipeline.run("θέλω βεβαίωση κατοικίας")
reply.text, reply.options, reply.trace
```

## Τι κάνει κάθε agent

| Αρχείο | Agent | Δουλειά | Μοντέλο |
|---|---|---|---|
| `welcome.py` | Welcome | χαιρετισμοί· χωρίζει το μήνυμα σε αιτήματα· αναδιατυπώνει (case 1)· ενώνει τις απαντήσεις | LLM για χωρισμό και αναδιατύπωση |
| `security.py` | Security | κανόνες σε όλο το μήνυμα, μετά ετικέτα ανά αίτημα: αίτημα, εκτός αρμοδιότητας, χειραγώγηση, προσβολή, επείγον | decision model ή LLM |
| `classifier.py` | BERT | το pipeline του σημερινού bot: υποψήφιοι προορισμοί και έτοιμα κείμενα | κανένα LLM |
| `validator.py` | Validator | μετά το BERT: ταιριάζει ο προορισμός στο αίτημα; πριν τον πολίτη: απαντά το κείμενο στο αίτημα; | decision model ή LLM |
| `pipeline.py` | — | η σειρά των βημάτων και οι δύο βρόχοι (case 1, case 2) | — |

Τα κείμενα των απαντήσεων και τα στοιχεία επικοινωνίας έρχονται πάντα από
τους πίνακες του bot. Κανένα μοντέλο δεν γράφει τηλέφωνο ή σύνδεσμο.

## Ρυθμίσεις (`config.py`)

| Μεταβλητή | Τι κάνει |
|---|---|
| `MA_PROFILE` | `tev1` (προεπιλογή: κρίσεις στο `tev1:4b`, κείμενο στο `qwen3:4b`), `qwen4b`, `qwen14b` |
| `MA_MODEL_<ΡΟΛΟΣ>` | άλλο μοντέλο για έναν ρόλο: `SPLIT`, `SECURITY`, `VALIDATOR`, `REFORM`, `FINAL` |
| `MA_URL_<ΡΟΛΟΣ>`, `MA_OLLAMA_URL` | άλλος server Ollama, για έναν ρόλο ή για όλους |
| `MA_RULES=0` | χωρίς τους κανόνες του Security |
| `MA_SPLIT=0` | χωρίς χωρισμό σε αιτήματα |
| `MA_REFORM=0` | χωρίς αναδιατύπωση (case 1) |
| `MA_FINAL=0` | χωρίς τον τελικό Validator |
| `MA_SCOPE` | `after` (προεπιλογή): το αίτημα περνά στο BERT και το «εκτός αρμοδιότητας» του Security μετρά μόνο αν ο Validator δεν εγκρίνει υπηρεσία· `before`: κόβει πριν από το BERT, όπως στο σχήμα |
| `MA_DECISION_LANG` | `en` (προεπιλογή) ή `el`: γλώσσα των ερωτήσεων προς το decision model (Security, Validator) |
| `MA_EMERGENCY` | `answer` (προεπιλογή): το «επείγον» δίνει το 112 και από κάτω την υπηρεσία του Δήμου, αν βρεθεί· `block`: μόνο το 112 |
| `MA_SPLIT_LANGS` | `all` (προεπιλογή): ο χωρισμός σε αιτήματα και σε αγγλικά και greeklish· `el`: μόνο σε ελληνικά |

Οι προεπιλογές βγήκαν από τους κανόνες απόφασης των μετρήσεων (`reports/multiagent.md`).
Στα τρία holdout το multi-agent βρίσκει όσες ή περισσότερες σωστές από το bot,
αλλά με ίδια ή περισσότερα σίγουρα λάθη, και θέλει 4–12 s ανά μήνυμα.

## Μέτρηση

```bash
venv/bin/python -m multiagent.evaluate security     # ο Security μόνος του
venv/bin/python -m multiagent.evaluate validator    # ο Validator μόνος του
venv/bin/python -m multiagent.evaluate pipeline     # όλη η ροή, δίπλα στο σημερινό bot
venv/bin/python -m multiagent.evaluate summary      # πίνακες
venv/bin/python -m multiagent.evaluate pipeline --configs tev1-after --sets holdout,holdout2
```

Οι ρυθμίσεις της μέτρησης (`tev1`, `tev1-after`, `tev1-en-after` κ.ά.) ορίζονται
στο `CONFIGS` του `evaluate.py`, με το `scope` και τη γλώσσα ρητά.

Τα σετ του `sets/` γράφτηκαν πριν τρέξει μοντέλο και εξαιρούνται από την
εκπαίδευση (`scripts/prepare_dataset.py`).

## Τι δεν έχει γίνει ακόμα

- Δεν έχει συνδεθεί με το παράθυρο ή το web demo.
- Η μνήμη υπηρεσίας του bot (ερωτήσεις συνέχειας όπως «τι χαρτιά θέλει;») δεν περνά από εδώ.
- Ο Planner του σχήματος δεν υπάρχει: τη σειρά την ορίζει ο κώδικας.
