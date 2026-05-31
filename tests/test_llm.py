"""Connectivity + response test for the Ollama LLM server.

Run after copying .env.example -> .env and setting OLLAMA_HOST:

    python tests/test_llm.py --config configs/data.yaml

Pings the server (lists models, checks the configured model is present) and sends
one sample annotation-style prompt asking for strict JSON, then prints the result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import llm  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    load_dotenv()
    host = os.environ.get("OLLAMA_HOST", "").strip()
    port = os.environ.get("OLLAMA_PORT", "11434").strip()
    if not host:
        print("OLLAMA_HOST is not set. Copy .env.example to .env and set it.")
        return 1

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    llm_cfg = cfg["llm"]
    model = llm_cfg["model"]
    timeout = float(llm_cfg.get("timeout_s", 120))
    options = llm_cfg.get("options")

    print(f"Server: http://{host}:{port}  model: {model}")

    try:
        tags = llm.health_check(host, port, timeout=10.0)
    except llm.OllamaError as exc:
        print(f"Health check failed: {exc}")
        return 1
    print(f"Reachable. Available models: {tags}")
    if model not in tags:
        print(f"WARNING: configured model '{model}' not in the server's model list.")

    messages = [
        {
            "role": "user",
            "content": (
                "For the board game 'Codenames', rate party_friendliness, "
                "interaction_level, and competitiveness as integers from 1 to 5."
            ),
        }
    ]
    # JSON schema constrains decoding so the model can't degenerate into free text.
    schema = {
        "type": "object",
        "properties": {
            "party_friendliness": {"type": "integer"},
            "interaction_level": {"type": "integer"},
            "competitiveness": {"type": "integer"},
        },
        "required": ["party_friendliness", "interaction_level", "competitiveness"],
    }
    try:
        content = llm.chat(
            host, port, model, messages, options=options, fmt=schema, timeout=timeout
        )
    except llm.OllamaError as exc:
        print(f"Chat request failed: {exc}")
        return 1

    print("\nRaw response:")
    print(content)
    try:
        print("\nParsed JSON:")
        print(json.dumps(json.loads(content), indent=2))
    except json.JSONDecodeError:
        print("\nWARNING: response was not valid JSON.")
        return 1

    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
