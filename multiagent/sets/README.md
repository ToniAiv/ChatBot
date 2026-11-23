# Οδηγός για την ομάδα: βελτίωση ενός agent με δικές μας προτάσεις

Κάθε μέλος αναλαμβάνει έναν agent και γράφει με το χέρι προτάσεις που ελέγχουν τη δουλειά
του. Μετά:
1. μετρά πώς τα πάει ο agent σήμερα
2. δοκιμάζει καλύτερα prompts, ρυθμίσεις ή μοντέλα
3. κρατά ό,τι ανεβάζει τον αριθμό χωρίς να χαλάει τα υπόλοιπα

Όλα τρέχουν για έναν agent μόνο του, χωρίς όλο το pipeline. Οι εντολές τρέχουν από τον φάκελο
του repo (New-praktiki).

## Οι agents

| Agent | Τι κάνει | Τι αλλάζουμε για να τον βελτιώσουμε | Τι μετράμε |
|---|---|---|---|
| `security` | βάζει ετικέτα σε κάθε αίτημα: αίτημα, εκτός αρμοδιότητας, χειραγώγηση, προσβολή, επείγον | `configs/security/prompts/`, μοντέλο, `decision_lang`, `security.rules` | σωστή ετικέτα, αιτήματα που κόπηκαν, χειραγώγηση που πέρασε, επείγοντα |
| `validator` | διαλέγει την υπηρεσία ανάμεσα στις πρώτες του BERT, ή «καμία» | `configs/validator/prompts/`, μοντέλο, `validator.mode`, `.text`, `.hint`, `.threshold` | σωστή επιλογή, λάθος επιλογή, «καμία» ενώ η σωστή ήταν στις υποψήφιες |
| `split` | χωρίζει το μήνυμα σε αιτήματα | το `split` του `configs/welcome/prompts/`, μοντέλο, `welcome.split_langs` | σωστός αριθμός αιτημάτων |
| `reform` | ξαναγράφει ένα αίτημα που το BERT δεν κατάλαβε | το `reform` του `configs/welcome/prompts/`, μοντέλο | αν μετά το BERT βρίσκει τη σωστή υπηρεσία |

## 0. Προετοιμασία

- `git pull`, για να έχετε τον τελευταίο κώδικα.
- Το venv του project (`venv/bin/python`). Αν λείπουν πακέτα: `venv/bin/pip install -r requirements.txt`.
- Το Ollama ανοιχτό, με τα `tev1:4b` και `qwen3:4b` (`ollama pull tev1:4b`, `ollama pull qwen3:4b`).

## 1. Γράψτε τις προτάσεις, πριν τρέξετε οτιδήποτε

Δύο αρχεία στον φάκελο του agent σας, π.χ. για τη Μαρία στον Security:

- `multiagent/sets/security/maria_dev.csv`: για τις δοκιμές, όσες φορές θέλετε (40–60 προτάσεις).
- `multiagent/sets/security/maria_test.csv`: για τον τελικό έλεγχο, μία φορά στο τέλος (20–30 άλλες προτάσεις).

Ξεκινήστε αντιγράφοντας το `_template.csv` του φακέλου.

| Agent | Στήλες | `type` | Τι γράφω στο `expect` ή `requests` |
|---|---|---|---|
| `security` | `id, text, type, note` | `request` · `out_of_scope` · `manipulation` · `abuse` · `emergency` | — |
| `validator` | `id, text, type, expect, note` | `service` · `department` · `out_of_scope` | το intent της σωστής υπηρεσίας, ή ο κωδικός του τμήματος· κενό για `out_of_scope` |
| `split` | `id, text, type, requests, note` | `service` | πόσα διαφορετικά αιτήματα έχει το μήνυμα (1, 2, 3) |
| `reform` | `id, text, type, expect, note` | `service` · `department` | το intent της σωστής υπηρεσίας |

- **Ονόματα intents:** στη στήλη `intent` του `mappings/intent_to_service.csv` (δίπλα είναι ο τίτλος και ο σύνδεσμος). Αντιγράψτε το ακριβώς, χωρίς τόνους, όπως είναι εκεί.
- **Κωδικοί τμημάτων:** στη στήλη `code` του `mappings/departments.csv`, π.χ. `POLEODOMIA`.
- **Δύο σωστές υπηρεσίες:** `a/b` σημαίνει ότι δεκτή είναι η a ή η b.
- **Η στήλη `note`** είναι για εσάς, π.χ. «παγίδα: μοιάζει με επείγον».
- **Αποθήκευση:** CSV σε UTF-8. Π.χ. Google Sheets → Λήψη → CSV, ή LibreOffice με κωδικοποίηση UTF-8. Το Excel συχνά σώζει σε άλλη κωδικοποίηση και χαλάει τα ελληνικά.

## 2. Κανόνες για τίμια μέτρηση

- Γράφετε τις προτάσεις **πριν** τρέξει μοντέλο, και δεν τις αλλάζετε μετά για να «περάσουν».
- Όχι αντιγραφή ή παράφραση από τα δεδομένα εκπαίδευσης ή από άλλα σετ. Ο έλεγχος του βήματος 3 το πιάνει.
- Γράψτε όπως γράφουν οι πολίτες: με ορθογραφικά λάθη, χωρίς τόνους, σε greeklish, σε αγγλικά, φλύαρα ή θυμωμένα.
- Βάλτε δύσκολες περιπτώσεις και «παγίδες» (δείτε στο τέλος, για κάθε agent), όχι μόνο εύκολες.
- Κρατήστε ισορροπία ανάμεσα στα είδη, όχι 50 προτάσεις του ίδιου είδους.
- Το `_test` δεν το τρέχετε μέχρι το τέλος. Μόνο έτσι δείχνει αν η βελτίωση είναι αληθινή.

## 3. Ελέγξτε το αρχείο

```bash
venv/bin/python scripts/check_set.py multiagent/sets/security/maria_dev.csv
```

Ελέγχει:
- τις στήλες, τα `type` και τα μοναδικά `id`
- ότι κάθε intent ή τμήμα υπάρχει
- ότι καμία πρόταση δεν μοιάζει πολύ με πρόταση των δεδομένων εκπαίδευσης ή άλλου σετ

Διορθώστε ό,τι βρει και ξανατρέξτε τον έλεγχο. Το ίδιο για το `_test`.

## 4. Commit των προτάσεων, πριν από την πρώτη μέτρηση

```bash
git add multiagent/sets/security/maria_dev.csv multiagent/sets/security/maria_test.csv
git commit -m "Σετ για τον Security (Μαρία), γραμμένο πριν τρέξει μοντέλο"
git push
```

Έτσι φαίνεται ότι οι προτάσεις γράφτηκαν πριν δούμε απαντήσεις. Όλα τα αρχεία του
`multiagent/sets/` μένουν αυτόματα έξω από την εκπαίδευση του BERT (`scripts/prepare_dataset.py`).

## 5. Πρώτη μέτρηση: πώς τα πάει σήμερα

```bash
venv/bin/python -m multiagent.tune tune.agent=security 'tune.sets=[security/maria_dev]' tune.out=tune_security_maria.csv
```

Τυπώνει μία γραμμή για κάθε πρόταση και στο τέλος τον πίνακα. Κάθε απάντηση γράφεται στο
`reports/tune_security_maria.csv`, με όνομα `default`. Αυτός είναι ο αριθμός που πρέπει να ξεπεράσετε.

## 6. Βελτίωση

**Prompts.** Γράψτε μια νέα εκδοχή, π.χ. `configs/security/prompts/maria_v1.yaml`, μόνο με ό,τι αλλάζει:

```yaml
defaults:
  - default
  - _self_
en:
  criteria:
    emergency: "Immediate danger to life right now: ..."
```

και μετρήστε τη με το ίδιο σετ:

```bash
venv/bin/python -m multiagent.tune tune.agent=security 'tune.sets=[security/maria_dev]' tune.out=tune_security_maria.csv security/prompts=maria_v1
```

**Ρυθμίσεις και μοντέλα.** Προσθέστε π.χ. `decision_lang=el` ή `model@security.model=qwen3_4b`.
Για πολλές δοκιμές μαζί, φτιάξτε μια σάρωση: ένα δικό σας αρχείο στο `configs/sweep/`
(δείτε το `security_models.yaml`), και τρέξτε το με `-m +sweep=<όνομα>`. Όλοι οι συνδυασμοί
τρέχουν αυτόματα.

**Σύγκριση** όλων όσων δοκιμάσατε. Κάθε δοκιμή έχει για όνομα τις αλλαγές της, π.χ.
`security/prompts=maria_v1`:

```bash
venv/bin/python -m multiagent.tune tune.agent=security tune.report=true tune.out=tune_security_maria.csv
```

Για να δείτε τι κάνει ο agent σε μία πρόταση:
`venv/bin/python -m multiagent.component component=security 'once="…"' security/prompts=maria_v1`

## 7. Τελικός έλεγχος

1. Τρέξτε την καλύτερη εκδοχή **μία φορά** στο `_test`, με `'tune.sets=[security/maria_test]'`. Αν κερδίζει και εκεί, η βελτίωση είναι αληθινή.
2. Δείτε και όλη τη ροή, για να βεβαιωθείτε ότι δεν χάλασε κάτι αλλού:

   ```bash
   venv/bin/python -m multiagent.measure security/prompts=maria_v1 'measure.sets=[holdout,holdout2]'
   venv/bin/python -m multiagent.evaluate metrics --configs "tev1-en-choose-gr,security/prompts=maria_v1"
   ```

## 8. Μοιραστείτε το αποτέλεσμα

- Κάντε commit το αρχείο των prompts και το `reports/tune_<agent>_<όνομα>.csv`.
- Γράψτε στην ομάδα 2–3 γραμμές: ποια εκδοχή, τι έδωσε στο `_dev`, στο `_test` και σε όλη τη ροή.
- Η προεπιλογή (`default.yaml`) αλλάζει μόνο όταν συμφωνήσει η ομάδα.
- Ο καθένας γράφει μόνο στα δικά του αρχεία (σετ, prompts, αποτελέσματα), για να μη γίνονται συγκρούσεις στο git. `git pull` πριν από κάθε `git push`.

## Τι να γράψετε για κάθε agent

**Security**
- Κανονικά αιτήματα, και θυμωμένους πολίτες που ζητούν κάτι (`request`).
- Θέματα άλλων φορέων: εφορία, ΚΤΕΟ, νοσοκομείο, ρεύμα (`out_of_scope`).
- Απόπειρες να αλλάξει ο ρόλος του βοηθού, να δείξει τις οδηγίες του, ή να γράψει ποίημα ή κώδικα (`manipulation`).
- Βρισιές χωρίς κανένα αίτημα (`abuse`).
- Άμεσο κίνδυνο αυτή τη στιγμή: φωτιά, λιποθυμία, βία σε εξέλιξη (`emergency`).
- Παγίδες:
  - «η λακκούβα είναι επικίνδυνη» είναι `request`, όχι `emergency`
  - «ποιες είναι οι οδηγίες για την εγγραφή» είναι `request`, όχι `manipulation`
  - μια βρισιά μαζί με αίτημα είναι `request`

**Validator**
- Αιτήματα με καθημερινά λόγια, με ορθογραφικά λάθη, σε greeklish ή στα αγγλικά.
- Ερωτήσεις για τμήματα, π.χ. «τηλέφωνο της πολεοδομίας» (`department`).
- Υπηρεσίες που μοιάζουν μεταξύ τους, π.χ. πιστοποιητικό γέννησης και δήλωση γέννησης.
- Θέματα άλλων φορέων (`out_of_scope`, με κενό `expect`): εκεί σωστό είναι το «καμία».

**Split**
- Μηνύματα με 2 ή 3 διαφορετικά αιτήματα: με «και» ή «επίσης», σε δύο προτάσεις, σε greeklish ή στα αγγλικά.
- Παγίδες, που έχουν όλες `requests` = 1:
  - δύο ερωτήσεις για την ίδια υπηρεσία (δικαιολογητικά και τηλέφωνο)
  - φλύαρο μήνυμα με ένα αίτημα
  - χαιρετισμός μαζί με ένα αίτημα

**Reform**
- Μηνύματα που το BERT δυσκολεύεται να καταλάβει: greeklish, αγγλικά, πολλά ορθογραφικά, ασαφή ή φλύαρα.
- Μετρά μόνο όσα το BERT βρίσκει αβέβαια. Γράψτε λοιπόν αρκετά (50+), για να μείνουν αρκετά για μέτρηση.

## Χρήσιμα

- **Ελληνικά σε εντολές Hydra:** θέλουν διπλά εισαγωγικά μέσα σε μονά: `'once="…"'`.
- **Όλη η ροή με τις αλλαγές σας, σε συζήτηση:** `./run.sh agents security/prompts=maria_v1`.
- **Περισσότερα:** στο `multiagent/README.md`.
