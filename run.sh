#!/usr/bin/env bash
# Εκκίνηση του εικονικού βοηθού. Τρέχει από οπουδήποτε.
#   ./run.sh            παράθυρο (desktop)
#   ./run.sh cli --query "..."
#   ./run.sh web        web έκδοση: http://127.0.0.1:8000/heraklion και /eservices (DEPLOY.md)
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/venv/bin/python"

ollama_up() { curl -s -m 3 http://localhost:11434/api/tags >/dev/null 2>&1; }

case "${1:-gui}" in
  web)
    # Η web έκδοση δουλεύει και χωρίς Ollama: χωρίς αυτό δεν απαντά στα
    # αγγλικά, στις 11 υπηρεσίες χωρίς ελεγμένη σελίδα δίνει το τηλέφωνο, και οι
    # ερωτήσεις συνέχειας αναγνωρίζονται μόνο με κανόνες (όχι με embeddings).
    ollama_up || echo "ℹ  Το Ollama δεν τρέχει: χωρίς αγγλικά, χωρίς την εφεδρεία Llama, χωρίς embeddings (bge-m3)."
    shift; cd "$ROOT"; PYTHONPATH="$ROOT/app" exec "$PY" "$ROOT/app/server.py" "$@" ;;
  gui|cli)
    if ! ollama_up; then
      echo "⚠  Το Ollama δεν τρέχει. Ξεκίνα το σε άλλο terminal:  ollama serve"
      echo "   (χωρίς αυτό ο βοηθός βρίσκει την υπηρεσία αλλά δεν διατυπώνει απάντηση)"
      exit 1
    fi
    if [ "$1" = gui ] || [ -z "${1:-}" ]; then exec "$PY" "$ROOT/app/gui.py"; fi
    shift; exec "$PY" "$ROOT/app/connector.py" "$@" ;;
  *)    echo "Χρήση: ./run.sh [gui|cli --query \"...\"|web]"; exit 1 ;;
esac
