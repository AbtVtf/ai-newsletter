"""Shared look for the shorts: frame size, palettes, type, CRT treatment.

Both the video renderer and the thumbnail renderer import from here, so the
two cannot drift apart. Nothing in this module needs a GPU.
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFont

FONTS = Path(__file__).resolve().parent / "assets"

W, H = 1080, 1920

THEMES = {
    # Green CRT phosphor — matches the newsletter's dark theme.
    "phosphor": {
        "head": "head_green",
        "hemi_sky": (0.66, 1.00, 0.74), "hemi_ground": (0.02, 0.09, 0.04),
        "key": (0.80, 1.00, 0.84), "rim": (0.20, 1.00, 0.42),
        "rim2": (0.10, 0.70, 0.30), "emissive": (0.02, 0.13, 0.05),
        "bg": (6, 13, 8),
        "ink": (216, 255, 224, 255), "dim": (96, 176, 118, 255),
        "glow": (126, 255, 160, 255), "rule": (28, 78, 44, 255),
        "foot": (78, 140, 96, 255),
    },
    # The original Jarvis hologram blue.
    "digital": {
        "head": "head_blue",
        "hemi_sky": (0.749, 0.878, 1.0), "hemi_ground": (0.039, 0.078, 0.157),
        "key": (0.812, 0.902, 1.0), "rim": (0.247, 0.816, 1.0),
        "rim2": (0.184, 0.498, 1.0), "emissive": (0.039, 0.118, 0.235),
        "bg": (11, 13, 18),
        "ink": (232, 236, 244, 255), "dim": (127, 154, 196, 255),
        "glow": (159, 216, 255, 255), "rule": (42, 58, 90, 255),
        "foot": (100, 122, 160, 255),
    },
}

# Layout (1080x1920). In the video the head occupies roughly y 630-1290, so
# the headline lives in the gap under the masthead and captions sit below the
# bust. The thumbnail reuses the same masthead block and fills the middle.
MASTHEAD_Y = 52
DATE_Y = 206
RULE_Y = 268
HEADLINE_Y = 340
HEADLINE_LEAD = 70
SOURCE_Y = 570
CAPTION_Y = 1580
CAPTION_LEAD = 58
FOOTER_Y = 1836


def font(name, size):
    return ImageFont.truetype(str(FONTS / f"{name}.ttf"), size)


def wrap(draw, text, fnt, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=fnt) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def center(draw, y, text, fnt, fill, width=W):
    draw.text(((width - draw.textlength(text, font=fnt)) / 2, y), text,
              font=fnt, fill=fill)


def masthead(draw, theme, date_line, width=W):
    """The block every frame and every thumbnail shares."""
    center(draw, MASTHEAD_Y, "The Daily Prompt", font("jacquard", 128),
           theme["ink"], width)
    center(draw, DATE_Y, date_line.upper(), font("vt323", 42), theme["dim"], width)
    draw.rectangle([90, RULE_Y, width - 90, RULE_Y + 3], fill=theme["rule"])
    draw.rectangle([90, RULE_Y + 8, width - 90, RULE_Y + 10], fill=theme["rule"])


def over(frame, layer, alpha_scale=1.0):
    """Alpha-composite a layer onto a frame. int32 throughout: frame*(255-a)
    reaches 65025, which silently wraps in int16."""
    a = layer[:, :, 3:4]
    if alpha_scale != 1.0:
        a = (a * alpha_scale).astype(np.int32)
    return ((frame.astype(np.int32) * (255 - a) + layer[:, :, :3] * a) // 255)


class CrtPost:
    """Scanlines, vignette and film grain. All the static parts are
    precomputed; per frame this is two multiplies and an add.

    Grain is generated coarse and upscaled. Per-pixel noise reads as digital
    sensor noise, looks nothing like film, and — being incompressible — costs
    more bitrate than the whole rest of the picture.
    """

    GRAIN_SCALE = 3

    def __init__(self, width=W, height=H, scanline=0.10, vignette=0.22,
                 grain=6.0, seed=7, pool=16):
        self.w, self.h = width, height
        rows = np.arange(height, dtype=np.float32)
        scan = 1.0 - scanline * (0.5 + 0.5 * np.sin(rows * math.pi))
        y = (rows / (height - 1) - 0.5) * 2
        x = (np.arange(width, dtype=np.float32) / (width - 1) - 0.5) * 2
        r = np.sqrt(y[:, None] ** 2 * 0.55 + x[None, :] ** 2)
        vig = 1.0 - vignette * np.clip(r, 0, 1) ** 2
        self.mask = (scan[:, None] * vig)[:, :, None].astype(np.float32)

        self.grain = None
        if grain > 0:
            rng = np.random.default_rng(seed)
            s = self.GRAIN_SCALE
            # Bilinear upscaling smooths the noise, so pre-compensate the sigma.
            sigma = grain * s * 0.55
            self.grain = []
            for _ in range(pool):
                small = rng.normal(0, sigma, (height // s, width // s)).astype(np.float32)
                up = Image.fromarray(small, mode="F").resize((width, height),
                                                             Image.BILINEAR)
                self.grain.append(np.rint(np.asarray(up)).astype(np.int16)[:, :, None])
        self.i = 0

    def __call__(self, frame):
        out = frame.astype(np.float32) * self.mask
        if self.grain:
            out += self.grain[self.i % len(self.grain)]
            self.i += 1
        return np.clip(out, 0, 255).astype(np.uint8)
