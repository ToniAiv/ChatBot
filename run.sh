#!/usr/bin/env bash
# Εκκίνηση του εικονικού βοηθού. Τρέχει από οπουδήποτε.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/venv/bin/python"

if ! curl -s -m 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "⚠  Το Ollama δεν τρέχει. Ξεκίνα το σε άλλο terminal:  ollama serve"
  echo "   (χωρίς αυτό ο βοηθός βρίσκει την υπηρεσία αλλά δεν διατυπώνει απάντηση)"
  exit 1
fi

case "${1:-gui}" in
  gui)  exec "$PY" "$ROOT/app/gui.py" ;;
  cli)  shift; exec "$PY" "$ROOT/app/connector.py" "$@" ;;
  *)    echo "Χρήση: ./run.sh [gui|cli --query \"...\"]"; exit 1 ;;
esac
