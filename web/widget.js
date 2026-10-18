/*
 * Αιωρούμενο κουμπί του Εικονικού Βοηθού (βασική εκδοχή — ελεύθερο να
 * αλλάξει εμφάνιση από όποιον έχει το «float»).
 *
 * Ενσωμάτωση σε ΟΠΟΙΑΔΗΠΟΤΕ σελίδα, με μία γραμμή:
 *   <script src="https://<server-του-βοηθού>/widget.js" defer></script>
 *
 * Ανοίγει την οθόνη συνομιλίας (/chat του ίδιου server) σε iframe. Τα στυλ
 * ζουν σε Shadow DOM, ώστε ούτε η σελίδα να χαλάει το κουμπί ούτε το κουμπί
 * τη σελίδα. Το iframe φορτώνει μόνο στο πρώτο άνοιγμα.
 */
(function () {
  if (window.__heraklionBot) return;            // μία φορά ανά σελίδα
  window.__heraklionBot = true;

  var script = document.currentScript;
  var base = script ? new URL(script.src, location.href).origin : location.origin;
  var CHAT_URL = base + '/chat';

  var host = document.createElement('div');
  host.id = 'heraklion-bot';
  var root = host.attachShadow ? host.attachShadow({ mode: 'open' }) : host;

  root.innerHTML =
    '<style>' +
    ':host { all: initial; }' +
    '.hb-btn { position: fixed; right: 20px; bottom: 20px; z-index: 2147483647;' +
    '  display: flex; align-items: center; gap: 10px; padding: 12px 18px 12px 14px;' +
    '  border: 0; border-radius: 999px; background: #16324f; color: #fff; cursor: pointer;' +
    '  font: 600 15px/1 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;' +
    '  box-shadow: 0 6px 20px rgba(22, 50, 79, .35); }' +
    '.hb-btn:hover { background: #1f4670; }' +
    '.hb-btn:focus-visible, .hb-close:focus-visible { outline: 3px solid #f2b705; outline-offset: 3px; }' +
    '.hb-btn svg { width: 24px; height: 24px; flex: none; }' +
    '.hb-panel { position: fixed; right: 20px; bottom: 88px; z-index: 2147483647;' +
    '  width: 400px; height: min(660px, calc(100vh - 120px)); display: none;' +
    '  flex-direction: column; border-radius: 14px; overflow: hidden; background: #f5f7fa;' +
    '  box-shadow: 0 12px 40px rgba(0, 0, 0, .28); }' +
    '.hb-panel.open { display: flex; }' +
    '.hb-top { flex: none; display: flex; align-items: center; justify-content: space-between;' +
    '  padding: 0 6px 0 14px; height: 38px; background: #16324f; color: #fff;' +
    '  font: 600 13px/1 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; }' +
    '.hb-panel iframe { flex: 1; width: 100%; min-height: 0; border: 0; display: block; }' +
    '.hb-close { width: 30px; height: 30px; border: 0; border-radius: 50%; background: transparent;' +
    '  color: #fff; font: 400 22px/30px Arial, sans-serif; cursor: pointer; }' +
    '.hb-close:hover { background: rgba(255, 255, 255, .15); }' +
    '@media (max-width: 520px) {' +
    '  .hb-panel { right: 0; bottom: 0; width: 100vw; height: 100%; border-radius: 0; }' +
    '  .hb-panel.open + .hb-btn { display: none; } }' +
    '@media (prefers-reduced-motion: no-preference) {' +
    '  .hb-panel.open { animation: hb-in .18s ease-out; }' +
    '  @keyframes hb-in { from { opacity: 0; transform: translateY(8px); } } }' +
    '</style>' +
    '<div class="hb-panel" role="dialog" aria-label="Εικονικός Βοηθός Δήμου Ηρακλείου">' +
    '  <div class="hb-top"><span>Εικονικός Βοηθός · πρόχειρη έκδοση</span>' +
    '    <button class="hb-close" type="button" aria-label="Κλείσιμο βοηθού">×</button></div>' +
    '</div>' +
    '<button class="hb-btn" type="button" aria-expanded="false" aria-label="Άνοιγμα Εικονικού Βοηθού">' +
    '  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"' +
    '       stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '    <path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/>' +
    '  </svg>' +
    '  <span>Βοηθός</span>' +
    '</button>';

  var panel = root.querySelector('.hb-panel');
  var btn = root.querySelector('.hb-btn');
  var close = root.querySelector('.hb-close');
  var frame = null;

  function open() {
    if (!frame) {
      frame = document.createElement('iframe');
      frame.src = CHAT_URL;
      frame.title = 'Εικονικός Βοηθός Δήμου Ηρακλείου';
      panel.appendChild(frame);
    }
    panel.classList.add('open');
    btn.setAttribute('aria-expanded', 'true');
    btn.setAttribute('aria-label', 'Κλείσιμο Εικονικού Βοηθού');
    frame.focus();
  }
  function hide() {
    panel.classList.remove('open');
    btn.setAttribute('aria-expanded', 'false');
    btn.setAttribute('aria-label', 'Άνοιγμα Εικονικού Βοηθού');
    btn.focus();
  }
  btn.addEventListener('click', function () {
    panel.classList.contains('open') ? hide() : open();
  });
  close.addEventListener('click', hide);
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && panel.classList.contains('open')) hide();
  });

  function mount() { document.body.appendChild(host); }
  if (document.body) mount();
  else document.addEventListener('DOMContentLoaded', mount);
})();
