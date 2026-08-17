"""Thumbnail for a short: masthead, one line of text, no face.

Same palette, type and CRT treatment as the video (all from design.py), but
the head is gone and the hook carries the frame. Needs no GPU.

  python3 video/make_thumb.py --work-dir output/<date>/shorts/story-1 \
      --hook "OPENAI PAUSED ITS OWN MODEL" --date-line "..." --out thumb.png
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from design import (FOOTER_Y, H, THEMES, W, CrtPost, center, font, masthead,
                    over, wrap)

# The hook block is centred in the space the head occupies in the video —
# between the masthead rule and the footer — rather than anchored to a fixed
# top, so a two-line hook and a five-line hook both sit balanced.
BLOCK_CENTER_Y = 1090
HOOK_MAX_LINES = 4
HOOK_MAX_PT = 68
HOOK_MIN_PT = 34
KICKER_GAP = 54
SOURCE_GAP = 46
RULE_GAP = 90


def glow_field(theme, cx, cy, radius):
    """A soft radial lift behind the text. Without it the empty middle of the
    frame reads as a rendering failure rather than a design choice."""
    y = (np.arange(H, dtype=np.float32) - cy) / radius
    x = (np.arange(W, dtype=np.float32) - cx) / radius
    r = np.sqrt(y[:, None] ** 2 + x[None, :] ** 2)
    falloff = np.clip(1.0 - r, 0, 1) ** 2.2
    tint = np.array(theme["glow"][:3], dtype=np.float32) / 255.0
    return (falloff[:, :, None] * tint[None, None, :] * 34.0)


def fit_hook(draw, hook):
    """Largest type that keeps the hook within HOOK_MAX_LINES. Shrinking beats
    dropping lines — a truncated hook is a wrong hook."""
    for pt in range(HOOK_MAX_PT, HOOK_MIN_PT - 1, -2):
        fnt = font("press", pt)
        lines = wrap(draw, hook.upper(), fnt, W - 150)
        if len(lines) <= HOOK_MAX_LINES:
            return fnt, lines
    fnt = font("press", HOOK_MIN_PT)
    return fnt, wrap(draw, hook.upper(), fnt, W - 150)


def build(theme, hook, date_line, source="", kicker=""):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    masthead(d, theme, date_line)

    kicker_font = font("vt323", 46)
    source_font = font("vt323", 38)
    fnt, lines = fit_hook(d, hook)
    lead = int(fnt.size * 1.5)

    # Measure the whole stack, then centre it as one unit.
    hook_h = len(lines) * lead
    height = hook_h
    if kicker:
        height += kicker_font.size + KICKER_GAP
    if source:
        height += SOURCE_GAP + 3 + SOURCE_GAP // 2 + source_font.size
    y = BLOCK_CENTER_Y - height // 2

    if kicker:
        center(d, y, kicker.upper(), kicker_font, theme["glow"])
        y += kicker_font.size + KICKER_GAP
    for line in lines:
        center(d, y, line, fnt, theme["ink"])
        y += lead
    if source:
        y += SOURCE_GAP
        d.rectangle([RULE_GAP, y, W - RULE_GAP, y + 3], fill=theme["rule"])
        y += 3 + SOURCE_GAP // 2
        center(d, y, f"SOURCE: {source.upper()}", source_font, theme["dim"])

    center(d, FOOTER_Y, "DAILY AI NEWS · FULL STORY IN BIO",
           font("vt323", 36), theme["foot"])
    return np.asarray(img).astype(np.int32)


def render(theme_name, hook, date_line, source="", kicker="", grain=6.0):
    theme = THEMES[theme_name]
    bg = np.zeros((H, W, 3), dtype=np.float32) + np.array(theme["bg"], dtype=np.float32)
    bg += glow_field(theme, W / 2, BLOCK_CENTER_Y, W * 0.95)
    frame = np.clip(bg, 0, 255).astype(np.uint8)
    frame = over(frame, build(theme, hook, date_line, source, kicker)).astype(np.uint8)
    # Grain pool of 1: a still only needs one field, not sixteen.
    return CrtPost(grain=grain, pool=1)(frame)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hook", required=True, help="the one-liner")
    ap.add_argument("--out", required=True)
    ap.add_argument("--date-line", default="")
    ap.add_argument("--source", default="")
    ap.add_argument("--kicker", default="", help="small line above the hook")
    ap.add_argument("--theme", default="phosphor", choices=sorted(THEMES))
    ap.add_argument("--grain", type=float, default=6.0)
    args = ap.parse_args()

    date_line = args.date_line or "The Daily Prompt · daily AI news"
    img = render(args.theme, args.hook, date_line, args.source, args.kicker,
                 args.grain)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(args.out)
    print(f"[thumb] {args.out}")


if __name__ == "__main__":
    main()
