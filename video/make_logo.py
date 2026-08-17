"""The Daily Prompt logo, drawn on a pixel grid rather than generated.

The mark is a terminal prompt — a chevron and a cursor block — on a 32x32
grid. Drawing it deterministically is what makes it usable as a platform
avatar: exact brand colours, true pixel edges, dead centre, and every export
size an integer multiple of the grid so nothing ever resamples.

  python3 video/make_logo.py            # everything into video/assets/logo/
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from design import THEMES, font

OUT = Path(__file__).resolve().parent / "assets" / "logo"
GRID = 32
# Integer multiples of GRID, so a cell is always a whole number of pixels.
SIZES = [1024, 512, 256, 128, 64, 32]

PALETTES = {
    "dark": {"bg": (6, 13, 8), "ink": (126, 255, 160), "scanline": True},
    # Newsprint, matching the paper's light theme.
    "light": {"bg": (237, 233, 222), "ink": (16, 24, 15), "scanline": False},
}

CHEVRON_ROWS = 13      # odd, so the chevron has a single apex row
CHEVRON_THICK = 3
CURSOR_W = 7
CURSOR_H = 11
GAP = 3                # cells between chevron and cursor


def mark_cells():
    """The mark as a set of (col, row) grid cells, centred in GRID x GRID."""
    cells = set()
    half = CHEVRON_ROWS // 2
    for r in range(CHEVRON_ROWS):
        d = abs(r - half)
        x = half - d                      # apex at the right, on the centre row
        for t in range(CHEVRON_THICK):
            cells.add((x + t, r))

    cursor_x = half + CHEVRON_THICK + GAP
    top = (CHEVRON_ROWS - CURSOR_H) // 2
    for cx in range(CURSOR_W):
        for cy in range(CURSOR_H):
            cells.add((cursor_x + cx, top + cy))

    # Centre by bounding box rather than by assumption.
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    dx = (GRID - (max(xs) - min(xs) + 1)) // 2 - min(xs)
    dy = (GRID - (max(ys) - min(ys) + 1)) // 2 - min(ys)
    return {(x + dx, y + dy) for x, y in cells}


CELLS = mark_cells()


def render_icon(size, palette="dark", transparent=False, radius_frac=0.18):
    if size % GRID:
        raise ValueError(f"{size} is not a multiple of the {GRID}px grid")
    cell = size // GRID
    pal = PALETTES[palette]

    if transparent:
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    else:
        img = Image.new("RGBA", (size, size), pal["bg"] + (255,))
    d = ImageDraw.Draw(img)
    for x, y in CELLS:
        d.rectangle([x * cell, y * cell, (x + 1) * cell - 1, (y + 1) * cell - 1],
                    fill=pal["ink"] + (255,))

    # Scanlines only above 256px: at avatar sizes they alias into stripes.
    if pal["scanline"] and not transparent and size >= 256:
        arr = np.asarray(img).astype(np.float32)
        rows = np.arange(size)
        mask = np.where(rows % max(2, size // 128) == 0, 0.86, 1.0)
        arr[:, :, :3] *= mask[:, None, None]
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    if not transparent:
        # Rounded square, so it reads as an app icon where platforms don't crop.
        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            [0, 0, size - 1, size - 1], radius=int(size * radius_frac), fill=255)
        out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        out.paste(img, (0, 0), mask)
        img = out
    return img


def render_wordmark(palette="dark", height=420, pad=60):
    """Horizontal lockup: the mark, then the masthead in the paper's own type."""
    pal = PALETTES[palette]
    icon_px = (height - pad * 2) // GRID * GRID
    icon = render_icon(icon_px, palette, transparent=True)

    fnt = font("jacquard", int(icon_px * 0.86))
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    text = "The Daily Prompt"
    tw = int(probe.textlength(text, font=fnt))

    width = pad + icon_px + pad + tw + pad
    img = Image.new("RGBA", (width, height), pal["bg"] + (255,))
    img.paste(icon, (pad, (height - icon_px) // 2), icon)
    d = ImageDraw.Draw(img)
    bbox = d.textbbox((0, 0), text, font=fnt)
    d.text((pad + icon_px + pad, (height - (bbox[3] - bbox[1])) // 2 - bbox[1]),
           text, font=fnt, fill=pal["ink"] + (255,))
    return img


def render_banner(width=2560, height=1440, palette="dark"):
    """YouTube channel art. Everything that matters lives in the 1546x423
    centre box, which is the only region shown on every device."""
    pal = PALETTES[palette]
    img = Image.new("RGBA", (width, height), pal["bg"] + (255,))
    lock = render_wordmark(palette, height=380)
    safe_w = 1546
    if lock.width > safe_w:
        lock = lock.resize((safe_w, int(lock.height * safe_w / lock.width)),
                           Image.LANCZOS)
    img.paste(lock, ((width - lock.width) // 2, (height - lock.height) // 2 - 30),
              lock)

    d = ImageDraw.Draw(img)
    tag = "DAILY AI NEWS · WRITTEN BY AGENTS · NEW EDITION EVERY MORNING"
    fnt = font("vt323", 46)
    d.text(((width - d.textlength(tag, font=fnt)) / 2, height // 2 + 190),
           tag, font=fnt, fill=THEMES["phosphor"]["dim"])
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for pal in PALETTES:
        for size in SIZES:
            p = out / f"icon-{pal}-{size}.png"
            render_icon(size, pal).save(p)
        render_wordmark(pal).save(out / f"wordmark-{pal}.png")
        print(f"[logo] {pal}: icons {SIZES} + wordmark")

    render_icon(1024, "dark", transparent=True).save(out / "mark-transparent.png")
    render_banner().save(out / "banner-2560x1440.png")
    print(f"[logo] mark-transparent.png, banner-2560x1440.png -> {out}")


if __name__ == "__main__":
    main()
