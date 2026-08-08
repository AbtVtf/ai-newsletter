"""OpenRouter client with role-based model selection from config/models.json."""

import json
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
_config = json.loads((ROOT / "config" / "models.json").read_text())
_spent = {"usd": 0.0}


def _api_key():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith(_config["api_key_env"] + "="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"{_config['api_key_env']} not found in .env")


def spent():
    return _spent["usd"]


def chat(role, prompt, max_tokens=2000, temperature=0.7, retries=3):
    role_cfg = _config["roles"][role]
    model = role_cfg["model"]
    for attempt in range(retries):
        payload = {"model": model,
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens,
                   "temperature": temperature,
                   "usage": {"include": True}}
        if role_cfg.get("reasoning_effort"):
            # cap thinking so reasoning models (GLM) don't spend the whole budget on it
            payload["reasoning"] = {"effort": role_cfg["reasoning_effort"]}
        resp = requests.post(
            f"{_config['base_url']}/chat/completions",
            headers={"Authorization": f"Bearer {_api_key()}",
                     "Content-Type": "application/json"},
            json=payload,
            timeout=180,
        )
        body = resp.json()
        if "choices" in body:
            _spent["usd"] += body.get("usage", {}).get("cost") or 0.0
            message = body["choices"][0]["message"]
            content = message.get("content")
            if content:
                return content
            # reasoning models can spend the whole budget thinking; give them more room
            if attempt < retries - 1:
                max_tokens *= 2
                continue
            # last resort: the answer is often sitting inside the reasoning trace
            if message.get("reasoning"):
                return message["reasoning"]
            raise RuntimeError(f"{model}: empty content after {retries} attempts (raise max_tokens)")
        if body.get("error", {}).get("code") == 429 and attempt < retries - 1:
            time.sleep(5 * (attempt + 1))
            continue
        raise RuntimeError(f"{model}: {body.get('error')}")
    raise RuntimeError(f"{model}: exhausted retries")


def chat_json(role, prompt, max_tokens=3000, retries=2):
    """chat() + robust JSON extraction, with one re-ask on parse failure."""
    text = chat(role, prompt, max_tokens=max_tokens)
    for _ in range(retries):
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        text = chat(role, prompt + "\n\nRespond with ONLY a valid JSON object, no prose, no code fences.",
                    max_tokens=max_tokens)
    raise RuntimeError(f"could not parse JSON from {role} response: {text[:300]}")
