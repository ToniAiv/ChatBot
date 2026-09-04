"""
Desktop GUI — Εικονικός Βοηθός Δήμου Ηρακλείου  (pywebview edition)
====================================================================
Native παράθυρο με HTML/CSS interior. Χρησιμοποιεί απευθείας
τις συναρτήσεις του connector.py (BERT → TF-IDF → Llama).

Χρήση:
    python3 gui.py

Προαπαιτούμενα:
    pip install pywebview
    Ollama να τρέχει τοπικά με llama3.1
"""

import base64
import json
import threading
import time
import webbrowser
from urllib.parse import urlparse
from pathlib import Path

import webview

import usage_log
from connector import (
    TranslationUnavailable,
    JSON_PATH,
    MSG,
    compose_answer,
    department_answer,
    find_candidates,
    load_departments,
    load_no_service,
    load_service_table,
    resolve_query,
    should_suggest,
    suggestions,
    MIN_BERT_CONFIDENCE,
    MIN_TFIDF_SCORE,
    build_tfidf_index,
    detect_intent,
    load_bert,
)

BASE_DIR       = Path(__file__).parent.resolve()
LOGO_PATH      = BASE_DIR / "logo.png"
PHONE_FALLBACK = "2813 409185"


# ══════════════════════════════════════════════════════════════
# HTML template (embedded)
# ══════════════════════════════════════════════════════════════
def _logo_data_url() -> str:
    if not LOGO_PATH.exists():
        return ""
    b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="el">
<head>
<meta charset="UTF-8">
<title>Δήμος Ηρακλείου — Εικονικός Βοηθός</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif; }
  .options { display: flex; flex-direction: column; gap: 6px; margin-top: 10px; }
  .option {
    text-align: left; padding: 9px 12px; border: 1px solid #c7d2e0;
    background: #fff; border-radius: 8px; cursor: pointer;
    font-size: 13px; line-height: 1.35; color: #16324f;
  }
  .option:hover:not(:disabled) { background: #eef4fb; border-color: #7aa7d4; }
  .option:disabled { opacity: .45; cursor: default; }
  body {
    display: flex; flex-direction: column;
    background: #f5f7fa;
    overflow: hidden;
    color: #1a1a1a;
  }
  .header {
    background: #ffffff;
    padding: 14px 20px;
    border-bottom: 1px solid #e4e8ee;
    display: flex; align-items: center; justify-content: space-between;
    flex-shrink: 0;
  }
  .header img { height: 36px; display: block; }
  .header .brand-text {
    font-weight: 700; font-size: 14px; color: #1a1a1a;
    letter-spacing: 0.5px;
  }
  .status { font-size: 12px; color: #8899aa; display: flex; align-items: center; gap: 6px; }
  .status::before {
    content: ''; width: 8px; height: 8px; border-radius: 50%;
    background: #22c55e; display: inline-block;
    box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.15);
  }
  .chat {
    flex: 1; overflow-y: auto;
    padding: 16px 14px;
    display: flex; flex-direction: column; gap: 8px;
  }
  .msg {
    max-width: 78%; padding: 10px 14px; border-radius: 18px;
    font-size: 14px; line-height: 1.45; word-wrap: break-word; white-space: pre-wrap;
  }
  .bot {
    background: #ffffff; color: #1a1a1a;
    border-bottom-left-radius: 4px;
    align-self: flex-start;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  }
  .user {
    background: #003d7a; color: #ffffff;
    border-bottom-right-radius: 4px;
    align-self: flex-end;
  }
  a { color: #003d7a; text-decoration: underline; }
  .user a { color: #ffffff; }
  .typing { display: inline-flex; gap: 4px; align-items: center; }
  .typing span {
    width: 7px; height: 7px; background: #8899aa; border-radius: 50%;
    animation: blink 1.2s infinite;
  }
  .typing span:nth-child(2) { animation-delay: 0.15s; }
  .typing span:nth-child(3) { animation-delay: 0.3s; }
  @keyframes blink { 0%, 80%, 100% { opacity: 0.25; transform: scale(0.9); } 40% { opacity: 1; transform: scale(1); } }
  .input-bar {
    display: flex; padding: 12px 14px; background: #ffffff;
    border-top: 1px solid #e4e8ee; gap: 8px;
    flex-shrink: 0;
  }
  .input-bar input {
    flex: 1; padding: 10px 14px; border: none; outline: none;
    background: #eef1f5; border-radius: 20px; font-size: 14px;
    font-family: inherit; color: #1a1a1a;
  }
  .input-bar input::placeholder { color: #8899aa; }
  .input-bar input:disabled { opacity: 0.6; }
  .input-bar button {
    padding: 10px 18px; border: none; cursor: pointer;
    background: #003d7a; color: white; border-radius: 20px;
    font-size: 13px; font-weight: 600;
    font-family: inherit; transition: background 0.15s;
  }
  .input-bar button:disabled { background: #a0b0c5; cursor: not-allowed; }
  .input-bar button:hover:not(:disabled) { background: #002b57; }

  /* Scrollbar minimal */
  .chat::-webkit-scrollbar { width: 6px; }
  .chat::-webkit-scrollbar-thumb { background: #c5ced8; border-radius: 3px; }
  .chat::-webkit-scrollbar-track { background: transparent; }
</style>
</head>
<body>
  <div class="header">
    <!-- LOGO_PLACEHOLDER -->
    <div class="status">Online</div>
  </div>

  <div class="chat" id="chat"></div>

  <div class="input-bar">
    <input id="input" type="text" placeholder="Φόρτωση μοντέλου…" disabled>
    <button id="send" disabled>Αποστολή</button>
  </div>

<script>
  const chat  = document.getElementById('chat');
  const input = document.getElementById('input');
  const btn   = document.getElementById('send');

  // Το κείμενο μπαίνει ΠΑΝΤΑ με textContent, ποτέ innerHTML.
  //
  // Παλιά αυτή η συνάρτηση έκανε `div.innerHTML = text` — και από εδώ
  // περνάει τόσο ό,τι πληκτρολογεί ο πολίτης όσο και η απάντηση του
  // Llama. Ένα «<img src=x onerror=...>» εκτελούνταν, και το JS στο
  // pywebview έχει πρόσβαση στο window.pywebview.api, δηλαδή και στο
  // open_url. Τώρα οι σύνδεσμοι χτίζονται ως κόμβοι DOM: ό,τι δεν είναι
  // αναγνωρισμένο URL παραμένει αδρανές κείμενο.
  function addMessage(text, sender) {
    const div = document.createElement('div');
    div.className = 'msg ' + sender;

    const str = String(text);
    const re  = /(https?:\/\/[^\s]+)/g;
    let last = 0, m;
    while ((m = re.exec(str)) !== null) {
      if (m.index > last) {
        div.appendChild(document.createTextNode(str.slice(last, m.index)));
      }
      // Το url ΠΡΕΠΕΙ να πιαστεί σε const μέσα στην επανάληψη. Το `m`
      // δηλώνεται έξω από τον βρόχο (το απαιτεί το re.exec), οπότε ο
      // listener θα έκλεινε πάνω στη μεταβλητή και όχι στην τιμή: όταν
      // τελειώνει ο βρόχος το m γίνεται null και κάθε κλικ έσκαγε στο m[0].
      const url = m[0];
      const a = document.createElement('a');
      a.href = '#';
      a.textContent = url;
      a.addEventListener('click', async ev => {
        ev.preventDefault();
        try {
          const ok = await window.pywebview.api.open_url(url);
          if (!ok) addMessage('Ο σύνδεσμος δεν μπόρεσε να ανοίξει: ' + url, 'bot');
        } catch (e) {
          addMessage('Σφάλμα ανοίγματος συνδέσμου: ' + e, 'bot');
        }
      });
      div.appendChild(a);
      last = m.index + url.length;
    }
    if (last < str.length) {
      div.appendChild(document.createTextNode(str.slice(last)));
    }

    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    return div;
  }

  // Εμφανίζει την απάντηση: κείμενο, και αν το μοντέλο δεν ήταν
  // σίγουρο, τις επιλογές ως κουμπιά. Οι τίτλοι μπαίνουν με textContent.
  function render(reply) {
    if (!reply) { addMessage('(κενή απάντηση)', 'bot'); return; }
    const text = (typeof reply === 'string') ? reply : reply.text;
    const opts = (typeof reply === 'string') ? [] : (reply.options || []);
    const div = addMessage(text || '(κενή απάντηση)', 'bot');
    if (!opts.length) return;

    const box = document.createElement('div');
    box.className = 'options';
    opts.forEach(o => {
      const b = document.createElement('button');
      b.className = 'option';
      b.textContent = (o.kind === 'phone' ? '\u260E  ' : '\u2192  ') + o.title;
      b.addEventListener('click', async () => {
        box.querySelectorAll('button').forEach(x => x.disabled = true);
        addMessage(o.title, 'user');
        const t = addTyping();
        const res = await window.pywebview.api.choose(o.intent);
        t.remove();
        render(res);
      });
      box.appendChild(b);
    });
    div.appendChild(box);
    chat.scrollTop = chat.scrollHeight;
  }

  function addTyping() {
    const div = document.createElement('div');
    div.className = 'msg bot typing';
    div.innerHTML = '<span></span><span></span><span></span>';
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    return div;
  }

  function setEnabled(on) {
    input.disabled = !on;
    btn.disabled   = !on;
    if (on) input.focus();
  }

  async function send() {
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    addMessage(text, 'user');
    setEnabled(false);
    const typing = addTyping();
    try {
      const reply = await window.pywebview.api.chat(text);
      typing.remove();
      render(reply);
    } catch (e) {
      typing.remove();
      addMessage('Σφάλμα: ' + e, 'bot');
    }
    setEnabled(true);
  }

  btn.addEventListener('click', send);
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter') send();
  });

  // Polling μέχρι να φορτώσει το BERT/TF-IDF
  async function waitReady() {
    while (true) {
      try {
        const ok = await window.pywebview.api.is_ready();
        if (ok) break;
      } catch (e) {}
      await new Promise(r => setTimeout(r, 300));
    }
    input.placeholder = 'Γράψτε την ερώτησή σας…';
    setEnabled(true);
    addMessage(
      'Καλησπέρα! Είμαι ο εικονικός βοηθός του Δήμου Ηρακλείου. ' +
      'Πώς μπορώ να σας βοηθήσω;',
      'bot'
    );
  }

  window.addEventListener('pywebviewready', waitReady);
</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════
# Pipeline state (module-level — NOT on the Api object, γιατί
# το pywebview κάνει recursive walk στα attributes του js_api
# και πέφτει σε άπειρο loop με scipy sparse .H property)
# ══════════════════════════════════════════════════════════════
_STATE: dict = {
    "ready":         False,
    "error":         None,
    "tokenizer":     None,
    "model":         None,
    "label_encoder": None,
    "service_table": None,
    "no_service": None,
    "vectorizer":    None,
    "tfidf_matrix":  None,
    "valid_kb":      None,
}


def _load_pipeline() -> None:
    try:
        tokenizer, model, label_encoder = load_bert()
        with open(JSON_PATH, encoding="utf-8") as f:
            kb = json.load(f)
        vectorizer, tfidf_matrix, valid_kb = build_tfidf_index(kb)
        _STATE.update({
            "tokenizer":     tokenizer,
            "model":         model,
            "label_encoder": label_encoder,
            "service_table": load_service_table(valid_kb),
            "no_service": load_no_service(load_departments()),
            "vectorizer":    vectorizer,
            "tfidf_matrix":  tfidf_matrix,
            "valid_kb":      valid_kb,
            "ready":         True,
        })
    except Exception as e:
        _STATE["error"] = str(e)
        print(f"[!] Σφάλμα φόρτωσης μοντέλου: {e}")


# ══════════════════════════════════════════════════════════════
# JS ↔ Python bridge  (μόνο methods, ΧΩΡΙΣ heavy attributes)
# ══════════════════════════════════════════════════════════════
# Έλεγχος συνδέσμων
# ══════════════════════════════════════════════════════════════
ALLOWED_HOSTS = ("heraklion.gr", "deyah.gr")


def _is_allowed_url(url: str) -> bool:
    """https και domain του Δήμου (ή υποτομέας του). Τίποτα άλλο."""
    try:
        u = urlparse(str(url))
    except Exception:
        return False
    if u.scheme != "https" or not u.hostname:
        return False
    host = u.hostname.lower()
    return any(host == d or host.endswith("." + d) for d in ALLOWED_HOSTS)


# ══════════════════════════════════════════════════════════════
class Api:
    def is_ready(self) -> bool:
        return bool(_STATE["ready"])


    def choose(self, intent: str) -> dict:
        """
        Ο πολίτης πάτησε μία από τις προτεινόμενες επιλογές.

        Παρακάμπτεται το κατώφλι εμπιστοσύνης — και σωστά: το κατώφλι
        υπάρχει για να μη ΜΑΝΤΕΨΕΙ το σύστημα, ενώ εδώ η επιλογή έγινε
        από άνθρωπο. Δέχεται μόνο intents που ξέρει το μοντέλο, ώστε να
        μη γίνει η μέθοδος τρόπος να ζητηθεί αυθαίρετο περιεχόμενο από
        τη σελίδα.

        Κάθε επιλογή καταγράφεται: είναι ετικετοποιημένο δεδομένο
        εκπαίδευσης, δωρεάν, από πραγματικό χρήστη.
        """
        if _STATE["error"] or not _STATE["ready"]:
            return self._reply("Το μοντέλο δεν είναι έτοιμο.")
        known = set(_STATE["label_encoder"].classes_)
        if intent not in known:
            return self._reply("Άγνωστη επιλογή.")
        try:
            if intent in (_STATE["no_service"] or {}):
                dept = _STATE["no_service"][intent]
                usage_log.log(f"[επιλογή] {intent}", "el", "", intent, None,
                              "chosen_phone", dept.get("name", ""))
                return self._reply(department_answer(dept))
            candidates, source = find_candidates(
                intent, _STATE["vectorizer"], _STATE["tfidf_matrix"],
                _STATE["valid_kb"], _STATE["service_table"])
            if not candidates:
                return self._reply(MSG["el"]["outofscope"])
            usage_log.log(f"[επιλογή] {intent}", "el", "", intent, None,
                          "chosen_link", candidates[0][1]["title"])
            return self._reply(compose_answer("", intent, candidates, source))
        except Exception as e:
            return self._reply(f"Σφάλμα: {e}")

    def open_url(self, url: str) -> bool:
        """
        Ανοίγει σύνδεσμο υπηρεσίας στον browser — ΜΟΝΟ του Δήμου.

        Χωρίς αυτόν τον έλεγχο δεχόταν οτιδήποτε: file://, custom
        schemes, ξένα domains. Ο βοηθός δείχνει αποκλειστικά σελίδες
        του heraklion.gr, οπότε το να το επιβάλλουμε εδώ δεν κοστίζει
        τίποτα λειτουργικά και κλείνει τη διαδρομή που θα εκμεταλλευόταν
        είτε ένα hallucinated URL είτε ένεση κώδικα στη σελίδα.
        """
        if not _is_allowed_url(url):
            print(f"[!] Απορρίφθηκε σύνδεσμος εκτός Δήμου: {url[:120]}")
            return False
        try:
            webbrowser.open(url, new=2)
            return True
        except Exception as e:
            print(f"[!] open_url failed: {e}")
            return False

    @staticmethod
    def _reply(text, options=None, lang="el"):
        """
        Η chat() επιστρέφει πάντα δομή, όχι σκέτο κείμενο: όταν το
        μοντέλο δεν είναι σίγουρο, η απάντηση δεν είναι μήνυμα αλλά
        τρεις επιλογές που πρέπει να γίνουν κουμπιά.
        """
        return {"text": text, "options": options or [], "lang": lang}

    def chat(self, query: str) -> dict:
        if _STATE["error"]:
            return self._reply(f"Σφάλμα κατά τη φόρτωση: {_STATE['error']}")
        if not _STATE["ready"]:
            return self._reply("Το μοντέλο φορτώνει ακόμα. Δοκιμάστε ξανά σε λίγο.")

        query = (query or "").strip()
        if not query:
            return self._reply("Παρακαλώ γράψτε μια ερώτηση.")

        try:
            t0 = time.monotonic()
            try:
                greek_query, lang = resolve_query(query)
            except TranslationUnavailable as exc:
                usage_log.log(query, "en", "", None, None,
                              "translation_failed", str(exc)[:120])
                return self._reply(MSG["en"]["tr_fail"], lang="en")

            intents = detect_intent(
                greek_query,
                _STATE["tokenizer"], _STATE["model"], _STATE["label_encoder"],
            )
            top_intent, top_score = intents[0]

            ms = (time.monotonic() - t0) * 1000
            if top_score < MIN_BERT_CONFIDENCE:
                opts = []
                if should_suggest(intents):
                    opts = suggestions(
                        intents, _STATE["vectorizer"], _STATE["tfidf_matrix"],
                        _STATE["valid_kb"], _STATE["service_table"],
                        _STATE["no_service"])
                if opts:
                    usage_log.log(query, lang, greek_query, top_intent, top_score,
                                  "suggest", " | ".join(o["title"][:40] for o in opts), ms)
                    return self._reply(
                        MSG[lang]["suggest"],
                        [{"intent": o["intent"], "title": o["title"], "kind": o["kind"]}
                         for o in opts], lang)
                usage_log.log(query, lang, greek_query, top_intent, top_score,
                              "refuse_low_confidence", "", ms)
                return self._reply(MSG[lang]["unknown"], lang=lang)

            if top_intent in (_STATE["no_service"] or {}):
                dept = _STATE["no_service"][top_intent]
                usage_log.log(query, lang, greek_query, top_intent, top_score,
                              "phone", dept.get("name", ""), ms)
                return self._reply(department_answer(dept, lang), lang=lang)

            candidates, source = find_candidates(
                top_intent,
                _STATE["vectorizer"], _STATE["tfidf_matrix"], _STATE["valid_kb"],
                _STATE["service_table"],
            )
            if not candidates or candidates[0][0] < MIN_TFIDF_SCORE:
                usage_log.log(query, lang, greek_query, top_intent, top_score,
                              "refuse_out_of_scope", "", ms)
                return self._reply(MSG[lang]["outofscope"], lang=lang)

            answer = compose_answer(greek_query, top_intent, candidates, source, lang)
            usage_log.log(query, lang, greek_query, top_intent, top_score,
                          "link", f"{source}: {candidates[0][1]['title']}",
                          (time.monotonic() - t0) * 1000)
            return self._reply(answer, lang=lang)
        except Exception as e:
            return self._reply(f"Σφάλμα: {e}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main() -> None:
    api = Api()
    threading.Thread(target=_load_pipeline, daemon=True).start()

    logo_url = _logo_data_url()
    if logo_url:
        logo_tag = f'<img src="{logo_url}" alt="Δήμος Ηρακλείου">'
    else:
        logo_tag = '<div class="brand-text">ΔΗΜΟΣ ΗΡΑΚΛΕΙΟΥ</div>'

    html = HTML_TEMPLATE.replace("<!-- LOGO_PLACEHOLDER -->", logo_tag)

    webview.create_window(
        "Δήμος Ηρακλείου — Εικονικός Βοηθός",
        html=html,
        js_api=api,
        width=460,
        height=700,
        resizable=True,
        min_size=(380, 520),
    )
    webview.start()


if __name__ == "__main__":
    main()
