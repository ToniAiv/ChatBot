"""
Έλεγχος του JavaScript του παραθύρου, με ψεύτικο DOM σε node.

ΓΙΑΤΙ ΥΠΑΡΧΕΙ: το regression.py ελέγχει το pipeline της Python, αλλά
τίποτα δεν έλεγχε το JS. Έτσι πέρασε απαρατήρητο ότι μετά τη διόρθωση
του XSS οι σύνδεσμοι έπαψαν να ανοίγουν — ο listener έκλεινε πάνω στη
μεταβλητή του βρόχου αντί στην τιμή της, και το σφάλμα έσκαγε σιωπηλά
στην κονσόλα του webview.

Ο κώδικας ΔΕΝ αντιγράφεται εδώ: εξάγεται από το ίδιο το app/gui.py, ώστε
ο έλεγχος να μην μπορεί να ξεφύγει από την πραγματικότητα.

Χρήση:  python3 scripts/test_gui_js.py
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from paths import ROOT

GUI = ROOT / "app" / "gui.py"

HARNESS = r"""
// ── ψεύτικο DOM, όσο χρειάζεται για να τρέξει το script ──
let OPENED = [], CHOSEN = [], CHAT_CALLS = [];

function mkEl(tag) {
  return {
    tag, children: [], listeners: {}, className: '', disabled: false,
    _text: '', style: {}, dataset: {},
    set textContent(v) { this._text = String(v); this.children = []; },
    get textContent() {
      if (this.children.length) return this.children.map(c => c.textContent ?? c.text ?? '').join('');
      return this._text;
    },
    set innerHTML(v) { this._html = String(v); },
    get innerHTML() { return this._html || ''; },
    appendChild(c) { this.children.push(c); return c; },
    remove() {},
    addEventListener(k, f) { this.listeners[k] = f; },
    querySelectorAll(sel) {
      const out = [];
      const walk = n => { (n.children || []).forEach(c => { out.push(c); walk(c); }); };
      walk(this);
      return out.filter(c => c.tag === 'button');
    },
    focus() {},
  };
}

const _els = {};
global.document = {
  createElement: mkEl,
  createTextNode: t => ({ tag: '#text', text: String(t), textContent: String(t), children: [] }),
  getElementById: id => (_els[id] = _els[id] || mkEl('div')),
  addEventListener: () => {},
};
global.window = {
  addEventListener: () => {},
  pywebview: { api: {
    open_url: async u => { OPENED.push(u); return true; },
    choose:   async i => { CHOSEN.push(i); return { text: 'ΑΠΑΝΤΗΣΗ ΓΙΑ ' + i, options: [] }; },
    chat:     async q => { CHAT_CALLS.push(q); return { text: 'ok', options: [] }; },
    is_ready: async () => true,
  } },
};
global.setTimeout = (f) => f();

__SCRIPT__

// ── έλεγχοι ──
const fails = [];
function check(name, cond, extra) {
  if (cond) console.log('  ✓ ' + name);
  else { console.log('  ✗ ' + name + (extra ? '  → ' + extra : '')); fails.push(name); }
}

(async () => {
  // 1. ένας σύνδεσμος ανοίγει το σωστό URL
  OPENED = [];
  let d = addMessage('Η υπηρεσία:\nhttps://eservices.heraklion.gr/a.html', 'bot');
  let links = d.children.filter(c => c.tag === 'a');
  check('βρίσκει τον σύνδεσμο', links.length === 1, 'βρέθηκαν ' + links.length);
  if (links.length) await links[0].listeners.click({ preventDefault() {} });
  check('το κλικ ανοίγει το URL', OPENED.length === 1 && OPENED[0].endsWith('a.html'),
        JSON.stringify(OPENED));

  // 2. δύο σύνδεσμοι — ο καθένας το δικό του (το σφάλμα του closure)
  OPENED = [];
  d = addMessage('α https://eservices.heraklion.gr/1.html β https://www.heraklion.gr/2.html', 'bot');
  links = d.children.filter(c => c.tag === 'a');
  for (const l of links) await l.listeners.click({ preventDefault() {} });
  check('κάθε σύνδεσμος ανοίγει το ΔΙΚΟ του URL',
        OPENED.length === 2 && OPENED[0].endsWith('1.html') && OPENED[1].endsWith('2.html'),
        JSON.stringify(OPENED));

  // 3. XSS: το κείμενο δεν ερμηνεύεται ως HTML
  const evil = '<img src=x onerror="window.HACKED=1">';
  d = addMessage(evil, 'user');
  check('το κείμενο δεν γίνεται HTML', !global.HACKED && d.textContent === evil,
        JSON.stringify(d.textContent));
  check('δεν δημιουργήθηκε στοιχείο <img>',
        !d.children.some(c => c.tag === 'img'));

  // 4. οι επιλογές γίνονται κουμπιά και στέλνουν το σωστό intent
  CHOSEN = [];
  render({ text: 'Μήπως εννοείτε;', options: [
    { intent: 'intent_a', title: 'Υπηρεσία Α', kind: 'link' },
    { intent: 'intent_b', title: 'Υπηρεσία Β', kind: 'phone' },
  ] });
  const chat = document.getElementById('chat');
  const btns = chat.querySelectorAll('button');
  check('δύο επιλογές → δύο κουμπιά', btns.length === 2, 'βρέθηκαν ' + btns.length);
  if (btns.length === 2) {
    await btns[1].listeners.click();
    check('το κλικ στέλνει το σωστό intent',
          CHOSEN.length === 1 && CHOSEN[0] === 'intent_b', JSON.stringify(CHOSEN));
    check('τα υπόλοιπα κουμπιά απενεργοποιούνται', btns[0].disabled === true);
  }

  // 5. συμβατότητα: σκέτο string
  let threw = false;
  try { render('απλό κείμενο'); } catch (e) { threw = true; }
  check('η render δέχεται και σκέτο string', !threw);

  console.log(fails.length ? '\n  ΑΠΟΤΥΧΙΕΣ: ' + fails.length : '\n  Όλοι οι έλεγχοι πέρασαν');
  process.exit(fails.length ? 1 : 0);
})();
"""


def main() -> int:
    src = GUI.read_text(encoding="utf-8")
    m = re.search(r"<script>(.*?)</script>", src, re.S)
    if not m:
        print("✗ δεν βρέθηκε μπλοκ <script> στο app/gui.py")
        return 1
    script = m.group(1)

    # Το script στο τέλος δένει listeners σε κουμπιά/πεδία — αβλαβές με
    # το ψεύτικο DOM, αλλά το `await` σε top-level δεν επιτρέπεται σε
    # σκέτο node script, οπότε τυλίγεται όλο σε συνάρτηση.
    harness = HARNESS.replace("__SCRIPT__", script)

    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                     encoding="utf-8") as f:
        f.write(harness)
        path = f.name
    try:
        r = subprocess.run(["node", path], capture_output=True, text=True)
    except FileNotFoundError:
        print("  ⚠ δεν βρέθηκε node — ο έλεγχος παραλείπεται")
        return 0
    finally:
        Path(path).unlink(missing_ok=True)

    print(r.stdout.rstrip())
    if r.stderr.strip():
        print("  stderr:", r.stderr.strip()[:600])
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
