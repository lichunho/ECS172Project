"""Thin Ollama HTTP client.

One job: talk to an Ollama server over its REST API. No retries, no streaming,
no caching. Host/port are passed in by the caller (read from the environment);
model and options come from the loaded config.
"""

from __future__ import annotations

import requests


class OllamaError(RuntimeError):
    """Raised when the Ollama server is unreachable or returns an error."""


def _base_url(host: str, port: int | str) -> str:
    return f"http://{host}:{port}"


def health_check(host: str, port: int | str, timeout: float = 10.0) -> list[str]:
    """GET /api/tags. Return the list of model tags available on the server.

    Raises OllamaError on connection failure, timeout, or a non-200 response.
    """
    url = f"{_base_url(host, port)}/api/tags"
    try:
        resp = requests.get(url, timeout=timeout)
    except requests.RequestException as exc:
        raise OllamaError(f"cannot reach Ollama at {url}: {exc}") from exc
    if resp.status_code != 200:
        raise OllamaError(f"GET /api/tags -> {resp.status_code}: {resp.text[:200]}")
    models = resp.json().get("models", [])
    return [m["name"] for m in models]


def chat(
    host: str,
    port: int | str,
    model: str,
    messages: list[dict],
    options: dict | None = None,
    fmt: str | dict | None = None,
    timeout: float = 120.0,
) -> str:
    """POST /api/chat (non-streaming). Return the assistant message content.

    `messages` is the Ollama chat format, e.g. [{"role": "user", "content": "..."}].
    `options` maps to Ollama sampling params (temperature, etc.).
    `fmt` enables structured output: "json" or a JSON schema dict.
    """
    url = f"{_base_url(host, port)}/api/chat"
    payload: dict = {"model": model, "messages": messages, "stream": False}
    if options:
        payload["options"] = options
    if fmt is not None:
        payload["format"] = fmt
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise OllamaError(f"cannot reach Ollama at {url}: {exc}") from exc
    if resp.status_code != 200:
        raise OllamaError(f"POST /api/chat -> {resp.status_code}: {resp.text[:200]}")
    return resp.json()["message"]["content"]
