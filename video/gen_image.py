"""Generate images through OpenRouter.

Note on models: Qwen's image *generator* is not on OpenRouter — the Qwen
models there are vision-language (image in, text out). Image output comes
from Google's Gemini image line and OpenAI's GPT-5-image line. Gemini flash
is by far the cheapest and is the default here.

  .venv/bin/python -m video.gen_image --prompt "..." --out logo.png
  .venv/bin/python -m video.gen_image --prompt "..." --out a.png -n 4
"""

import argparse
import base64
import json
from pathlib import Path

import requests

from pipeline import llm

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "google/gemini-2.5-flash-image"


def generate(prompt, model=DEFAULT_MODEL, timeout=180):
    """Returns a list of raw image bytes."""
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {llm._api_key()}",
                 "Content-Type": "application/json"},
        json={"model": model,
              "messages": [{"role": "user", "content": prompt}],
              "modalities": ["image", "text"]},
        timeout=timeout,
    )
    body = resp.json()
    if "choices" not in body:
        raise RuntimeError(f"{model}: {json.dumps(body)[:400]}")
    images = body["choices"][0]["message"].get("images") or []
    out = []
    for img in images:
        url = img.get("image_url", {}).get("url", "")
        if url.startswith("data:"):
            out.append(base64.b64decode(url.split(",", 1)[1]))
    if not out:
        text = body["choices"][0]["message"].get("content", "")
        raise RuntimeError(f"{model} returned no image. Said: {text[:300]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True, help="png path; -n>1 appends an index")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("-n", type=int, default=1, help="how many attempts")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        images = generate(args.prompt, args.model)
        for j, data in enumerate(images):
            path = out if args.n == 1 and j == 0 else \
                out.with_name(f"{out.stem}-{i + 1}{j or ''}{out.suffix}")
            path.write_bytes(data)
            print(f"[image] {path}  ({len(data) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
