"""
Οι δύο τρόποι με τους οποίους ένας agent ρωτά ένα μοντέλο του Ollama.

    chat_json  LLM με έξοδο JSON (/api/chat με format=schema)
    decide     decision model (/v1/systemone): ερωτήσεις ναι/όχι ή επιλογής,
               με πιθανότητα για κάθε απάντηση· δεν γράφει κείμενο

Και οι δύο επιστρέφουν (αποτέλεσμα, δευτερόλεπτα) και σηκώνουν ModelError
όταν το μοντέλο δεν απαντήσει. Ο agent αποφασίζει τι κάνει τότε.

OllamaModel: ένα μοντέλο όπως το παίρνει ένας agent (όνομα, server, χρονικό
όριο). Τα configs/model/*.yaml φτιάχνουν ένα για κάθε agent.
"""
from __future__ import annotations

import json
import time

import requests

from multiagent.config import OLLAMA_URL, is_decision_model


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


def reachable(url: str, timeout: float = 3) -> bool:
    """Απαντά ο server του Ollama;"""
    try:
        return requests.get(f"{url}/api/tags", timeout=timeout).ok
    except requests.RequestException:
        return False


def require(models) -> None:
    """Σταματά με μήνυμα αν δεν απαντά ο server κάποιου μοντέλου (εκτός από τα FailingModel).
    Για τις μετρήσεις: χωρίς αυτό, ένα κλειστό Ollama γράφει απαντήσεις «χωρίς μοντέλο»."""
    for url in sorted({m.url for m in models if type(m) is OllamaModel}):
        if not reachable(url):
            raise SystemExit(f"Το Ollama δεν απαντά στο {url}. Ανοίξτε το (π.χ. ollama serve) και ξανατρέξτε.")


class OllamaModel:
    """Ένα μοντέλο του Ollama για έναν agent. Ο agent δεν ξέρει πού τρέχει το μοντέλο."""

    def __init__(self, *, name: str, url: str = OLLAMA_URL, timeout: int = 120):
        self.name, self.url, self.timeout = name, str(url).rstrip("/"), int(timeout)

    @property
    def decision(self) -> bool:
        """Decision model (decide) ή LLM με έξοδο JSON (chat_json)."""
        return is_decision_model(self.name)

    def chat_json(self, system: str, user: str, schema: dict, num_predict: int = 400) -> tuple:
        return chat_json(self.url, self.name, system, user, schema, self.timeout, num_predict)

    def decide(self, state, questions: dict) -> tuple:
        return decide(self.url, self.name, state, questions, self.timeout)

    def params(self) -> dict:
        return {"name": self.name, "url": self.url, "timeout": self.timeout}

    def __repr__(self) -> str:
        return f"OllamaModel({self.name} @ {self.url})"


class FailingModel(OllamaModel):
    """Μοντέλο που αποτυγχάνει πάντα, για να δούμε τι κάνει το pipeline όταν ένας agent
    μένει χωρίς απάντηση (configs/model/fail.yaml). decision: ποιον δρόμο ακολουθεί ο agent
    (decision model ή LLM)· αποτυγχάνουν και οι δύο."""

    def __init__(self, *, name: str = "fail", decision: bool = True, url: str = "", timeout: int = 0):
        self.name, self.url, self.timeout, self._decision = name, url, int(timeout), bool(decision)

    @property
    def decision(self) -> bool:
        return self._decision

    def chat_json(self, system: str, user: str, schema: dict, num_predict: int = 400) -> tuple:
        raise ModelError(f"{self.name}: δοκιμή αποτυχίας")

    def decide(self, state, questions: dict) -> tuple:
        raise ModelError(f"{self.name}: δοκιμή αποτυχίας")

    def params(self) -> dict:
        return {"name": self.name, "decision": self._decision}
