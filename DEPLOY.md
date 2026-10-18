# Web έκδοση του βοηθού (για το showcase)

Ο ίδιος βοηθός με το desktop παράθυρο, ως web εφαρμογή: ένας μικρός server
(`app/server.py`), η οθόνη συνομιλίας (`/chat`) και ένα αιωρούμενο κουμπί
(`/widget.js`) που μπαίνει σε οποιαδήποτε σελίδα με μία γραμμή.

**Κατάσταση: πρόχειρη έκδοση για παρουσίαση, όχι για κοινό.** Τι λείπει
για κανονική λειτουργία: στο τέλος.

## 1. Τοπικά (laptop)

```bash
cd ~/Documents/Projects/Praktiki/New-praktiki
./run.sh web
```

Μετά ανοίγεις το **http://127.0.0.1:8000/heraklion** (η επίσημη αρχική με
το κουμπί «Βοηθός» κάτω δεξιά) ή το **/eservices**. Το σκέτο
`http://127.0.0.1:8000/` ανοίγει το `/heraklion`. Δεν χρειάζεται Ollama. Χωρίς αυτό δεν απαντά στα
αγγλικά, και στις 11 υπηρεσίες χωρίς ελεγμένη σελίδα δίνει το τηλέφωνο του
δήμου.

Για να το ανοίξουν άλλοι στο ίδιο δίκτυο (π.χ. στην αίθουσα της παρουσίασης):

```bash
BOT_HOST=0.0.0.0 ./run.sh web
```

Μετά από άλλο υπολογιστή ανοίγουν το `http://<IP-του-laptop>:8000/heraklion`.

### Πάνω στην επίσημη σελίδα του δήμου (για το showcase)

- **http://127.0.0.1:8000/heraklion**: η αρχική του heraklion.gr
- **http://127.0.0.1:8000/eservices**: η αρχική του eservices.heraklion.gr

Ο server φέρνει **ζωντανά** την επίσημη σελίδα και προσθέτει μόνο το κουμπί.
Χρειάζεται internet. Τίποτα δεν αλλάζει στον ιστότοπο του δήμου: το βλέπει
μόνο όποιος ανοίγει το laptop.

Γιατί όχι κατευθείαν στο https://www.heraklion.gr με σελιδοδείκτη: ο browser
απαγορεύει σε δημόσια σελίδα να φορτώσει οτιδήποτε από το `localhost`
(δοκιμάστηκε 18/10: «Failed to fetch»). Το heraklion.gr δεν έχει άλλον
περιορισμό (CSP), άρα **μόλις ο βοηθός φιλοξενηθεί σε server με HTTPS**,
αυτός ο σελιδοδείκτης βάζει το κουμπί στην πραγματική σελίδα, μόνο στον
browser του παρουσιαστή:

```
javascript:(function(){var s=document.createElement('script');s.src='https://<server-του-βοηθού>/widget.js';document.body.appendChild(s);})();
```

**Μόνο τοπικά.** Σε δημόσιο server οι διευθύνσεις `/heraklion` και `/eservices`
θα έδειχναν την επίσημη σελίδα κάτω από άλλη διεύθυνση. Γι' αυτό κλείνουν με
`BOT_SHOWCASE=0`, όπως κάνει ήδη το Dockerfile.

## 2. Για το float (αιωρούμενο κουμπί)

Το `web/widget.js` είναι μια βασική εκδοχή που δουλεύει. Αλλάζει ελεύθερα
(εμφάνιση, θέση, κείμενο). Μπαίνει σε **οποιαδήποτε** σελίδα:

```html
<script src="http://127.0.0.1:8000/widget.js" defer></script>
```

Ανοίγει το `/chat` του ίδιου server σε iframe. Αν το float φτιάξει δική του
οθόνη συνομιλίας αντί για iframe, μιλά απευθείας με το API:

| Κλήση | Σώμα (JSON) | Απάντηση |
|---|---|---|
| `GET /api/health` | — | `{"ready": true, "error": null}` |
| `POST /api/chat` | `{"message": "θέλω βεβαίωση κατοικίας"}` | `{"text": "...", "options": [...], "lang": "el"}` |
| `POST /api/choose` | `{"intent": "<intent από options>"}` | ίδια μορφή |
| `POST /api/reset` | `{}` | `{"ok": true}` |

- Κάθε `POST` θέλει header **`X-Session-Id`**: τυχαίο id ανά επισκέπτη (8–64
  χαρακτήρες, γράμματα/αριθμοί/παύλες), π.χ. `crypto.randomUUID()` κρατημένο
  στο `sessionStorage`. Σε αυτό στηρίζεται η μνήμη της συζήτησης («τι
  δικαιολογητικά;» → «για τη βεβαίωση κατοικίας»).
- `options` = κουμπιά του «Μήπως εννοείτε;»: `{intent, title, kind}`, όπου
  `kind` είναι `link` (σελίδα) ή `phone` (τηλέφωνο τμήματος). Στο πάτημα
  στέλνεις `POST /api/choose` με το `intent`.
- Το `text` περιέχει συνδέσμους ως απλό κείμενο. Εμφανίζεται **πάντα** με
  `textContent`, ποτέ με `innerHTML`. Ανοίγουν μόνο σύνδεσμοι του
  `heraklion.gr` / `deyah.gr` (δες `gui.py`, `addMessage`).
- Όριο: 500 χαρακτήρες ανά μήνυμα.

## 3. Για το hosting

Δύο παραλλαγές — στη συνάντηση αποφασίζουμε ποια ανεβαίνει:

- **Α) Σκέτη εικόνα, ΧΩΡΙΣ Ollama** (φθηνό, ~2 GB, μόνο ελληνικά):
  `docker build -t heraklion-bot .` και `docker run -p 8000:8000 heraklion-bot`.
  Το build φέρνει τα βάρη του μοντέλου (`models/intent-model-234-v2/model.safetensors`,
  432 MB, **δεν είναι στο git** — πάνω από το όριο 100 MB του GitHub, όπως και
  του 234/218). Dataset v6 με την αναδιάρθρωση copy-signal στα ληξιαρχικά.
  Δοκιμασμένο build + run (18/10/2026).
  **Τα μικρά αρχεία του μοντέλου (config/tokenizer/label_encoder) ΕΙΝΑΙ στο git·
  λείπει ΜΟΝΟ το `model.safetensors`.** Πάρ' το από το zip του retrain (Colab)
  και βάλ' το στον φάκελο `models/intent-model-234-v2/` πριν το build. Έλεγχος
  ότι είναι το σωστό: `sha256sum` =
  `eab1f24b9eef6c4ccd101403296551fa5995507eb46d82a2f8c7db561002d286`.
- **Β) Με Ollama** (αγγλικά + διατύπωση με LLM, θέλει μνήμη για το Llama):
  `OLLAMA_MODEL=llama3.2:3b docker compose -f docker-compose.ollama.yml up --build`.
  Σηκώνει δύο containers (app + ollama), κατεβάζει το μοντέλο μία φορά.
  `llama3.2:3b` ~4-6 GB, `llama3.1` 8B ~10-12 GB. Το Ollama ΔΕΝ εκτίθεται έξω.
- **Χωρίς Docker:** Python 3.9+, `pip install -r requirements.txt`, και μετά
  `BOT_HOST=0.0.0.0 PYTHONPATH=app python app/server.py`.
- **Πόροι:** ~2 GB μνήμη, χωρίς κάρτα γραφικών· ~20 ms ανά ερώτηση. Η
  εκκίνηση φορτώνει το μοντέλο σε λίγα δευτερόλεπτα, και ως τότε το
  `/api/health` λέει `ready: false`.
- **Ρυθμίσεις** (μεταβλητές περιβάλλοντος):

| Μεταβλητή | Προεπιλογή | Τι κάνει |
|---|---|---|
| `BOT_HOST` | `127.0.0.1` | `0.0.0.0` για πρόσβαση από έξω |
| `BOT_PORT` | `8000` | |
| `BOT_ALLOWED_ORIGINS` | `*` | ποιες σελίδες καλούν το API, π.χ. `https://www.heraklion.gr,https://eservices.heraklion.gr` |
| `BOT_FRAME_ANCESTORS` | `*` | ποιες σελίδες δείχνουν το `/chat` σε iframe (ίδια μορφή, με κενά) |
| `BOT_LOG` | `0` | `1` = καταγραφή ερωτήσεων στο `logs/usage.jsonl`, **με προσωπικά δεδομένα** |
| `BOT_SHOWCASE` | `1` | `/heraklion`, `/eservices` (μόνο τοπικά· **0 σε δημόσιο server**) |
| `BOT_MODEL` | `models/intent-model-234-v2` | ποιο μοντέλο φορτώνεται (το Dockerfile το καρφώνει στο v2) |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | στο compose με Ollama: `http://ollama:11434/api/generate` |
| `OLLAMA_MODEL` | `llama3.1` | μοντέλο Llama για αγγλικά + διατύπωση (π.χ. `llama3.2:3b`) |

## Τι λείπει για κανονική λειτουργία με κοινό

1. **HTTPS και όριο ερωτήσεων ανά IP:** reverse proxy (nginx ή Caddy) μπροστά από τον server.
2. **GDPR:** το log μένει κλειστό ως το τέλος της ανωνυμοποίησης και της ενημέρωσης των χρηστών.
3. **Προσβασιμότητα (WCAG):** έλεγχος του κουμπιού και της οθόνης συνομιλίας.
4. **Επιβεβαιώσεις του δήμου:** τηλέφωνο εφεδρείας 2813 409185, intents της Αττικής (HANDOFF §8Δ).
5. **Production server:** ο ενσωματωμένος server της Python αρκεί για demo· για κανονική κίνηση, FastAPI/uvicorn ή gunicorn.
