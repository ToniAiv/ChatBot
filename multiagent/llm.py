"""
Οι δύο τρόποι με τους οποίους ένας agent ρωτά ένα μοντέλο του Ollama.

    chat_json  LLM με έξοδο JSON (/api/chat με format=schema)
    decide     decision model (/v1/systemone): ερωτήσεις ναι/όχι ή επιλογής,
               με πιθανότητα για κάθε απάντηση· δεν γράφει κείμενο

Και οι δύο επιστρέφουν (αποτέλεσμα, δευτερόλεπτα) και σηκώνουν ModelError
όταν το μοντέλο δεν απαντήσει. Ο agent αποφασίζει τι κάνει τότε.
"""
from __future__ import annotations

import json
import time

import requests


class ModelError(RuntimeError):
    """Το μοντέλο δεν απάντησε, ή απάντησε κάτι που δεν διαβάζεται."""


# Το Ollama ξεφορτώνει ένα μοντέλο μετά από 5 λεπτά αδράνειας. Σε συζήτηση
# αυτό σημαίνει κρύα φόρτωση στο επόμενο μήνυμα.
KEEP_ALIVE = "30m"

_THINKING: dict = {}


def _supports_thinking(url: str, model: str) -> bool:
    if (url, model) not in _THINKING:
        try:
            show = requests.post(f"{url}/api/show", json={"model": model}, timeout=30).json()
            _THINKING[(url, model)] = "thinking" in (show.get("capabilities") or [])
        except (requests.RequestException, ValueError):
            _THINKING[(url, model)] = False
    return _THINKING[(url, model)]


def chat_json(url: str, model: str, system: str, user: str, schema: dict,
              timeout: int = 120, num_predict: int = 400) -> tuple:
    payload = {
        "model": model, "format": schema, "stream": False, "keep_alive": KEEP_ALIVE,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": num_predict}}
    if _supports_thinking(url, model):     # στο Qwen ρητά false· αλλιώς «σκέφτεται» εξ ορισμού
        payload["think"] = False
    t0 = time.monotonic()
    try:
        r = requests.post(f"{url}/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
        out = json.loads(r.json()["message"]["content"])
    except (requests.RequestException, ValueError, KeyError) as exc:
        raise ModelError(f"{model}: {exc}") from exc
    return out, round(time.monotonic() - t0, 2)


def decide(url: str, model: str, state, questions: dict, timeout: int = 120) -> tuple:
    """questions: {όνομα: {"type": "noul" | "choice", "instructions": …, "criteria": …}}.
    Επιστρέφει τα answers του Ollama: {"noul": p} ή {"choice": …, "probabilities": {…}}."""
    payload = {"model": model, "state": state, "questions": questions, "keep_alive": KEEP_ALIVE}
    t0 = time.monotonic()
    try:
        r = requests.post(f"{url}/v1/systemone", json=payload, timeout=timeout)
        r.raise_for_status()
        out = r.json()["answers"]
    except (requests.RequestException, ValueError, KeyError) as exc:
        raise ModelError(f"{model}: {exc}") from exc
    return out, round(time.monotonic() - t0, 2)
