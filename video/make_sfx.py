"""Synthesized opening sounds for the shorts.

All numpy, no samples, nothing to license. Several distinct directions
rather than one, because this is a taste call and auditioning beats
guessing.

  python3 video/make_sfx.py --demo        # write every preset
  python3 video/make_sfx.py --preset boom # write one as the default sfx
"""

import argparse
import wave
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"
SFX_DIR = ASSETS / "sfx"
SR = 44100


def t_axis(dur):
    return np.arange(int(SR * dur)) / SR


def env(t, attack, decay, start=0.0):
    """Linear attack into exponential decay."""
    e = np.zeros_like(t)
    m = t >= start
    tt = t[m] - start
    e[m] = np.where(tt < attack, tt / max(attack, 1e-6),
                    np.exp(-(tt - attack) / decay))
    return np.clip(e, 0, 1)


def sweep(t, f0, f1, curve=1.0):
    """Sine sweeping f0 -> f1 over the whole array."""
    p = np.clip(np.linspace(0, 1, len(t)) ** curve, 0, 1)
    return np.sin(2 * np.pi * np.cumsum(f0 + (f1 - f0) * p) / SR)


def lowpass(x, cutoff):
    a = np.exp(-2 * np.pi * cutoff / SR)
    out = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc = (1 - a) * v + a * acc
        out[i] = acc
    return out


def highpass(x, cutoff):
    return x - lowpass(x, cutoff)


def noise(n, seed=3):
    return np.random.default_rng(seed).normal(0, 1, n)


# ---------- presets ----------

def p_thump():
    """Warm low drop. Minimal, gets out of the way of the first word."""
    t = t_axis(0.9)
    body = sweep(t, 95, 38, curve=0.35) * env(t, 0.003, 0.16)
    puff = lowpass(noise(len(t)), 700) * env(t, 0.004, 0.05) * 0.5
    return body + puff


def p_boom():
    """Cinematic sub hit. Weight and a longer tail."""
    t = t_axis(1.4)
    sub = sweep(t, 72, 28, curve=0.3) * env(t, 0.004, 0.45)
    upper = sweep(t, 144, 56, curve=0.3) * env(t, 0.004, 0.22) * 0.35
    crack = highpass(noise(len(t), 5), 900) * env(t, 0.001, 0.02) * 0.4
    return np.tanh((sub + upper) * 1.6) * 0.8 + crack


def p_switch():
    """Dry mechanical click and thud. No tail, no tone, nothing to ring."""
    t = t_axis(0.35)
    click = highpass(noise(len(t), 7), 2200) * env(t, 0.0004, 0.008)
    thud = sweep(t, 130, 58, curve=0.4) * env(t, 0.002, 0.045) * 0.9
    return click * 0.8 + thud


def p_blip():
    """Clean two-tone data chirp. Reads as a system coming online."""
    t = t_axis(0.55)
    a = np.sin(2 * np.pi * 784 * t) * env(t, 0.002, 0.035, start=0.0)
    b = np.sin(2 * np.pi * 1175 * t) * env(t, 0.002, 0.055, start=0.085)
    sub = sweep(t, 90, 45, curve=0.4) * env(t, 0.003, 0.09) * 0.55
    return (a * 0.5 + b * 0.6 + sub)


def p_riser():
    """Short sweep landing on a hit. The conventional short-form opener."""
    t = t_axis(1.1)
    land = 0.42
    n = noise(len(t), 11)
    # Band of noise climbing, amplitude rising into the landing.
    band = highpass(lowpass(n, 900), 200)
    climb = np.clip(t / land, 0, 1)
    band_env = (climb ** 2) * (t < land)
    swoosh = band * band_env * 2.2
    hit = sweep(t, 84, 32, curve=0.3) * env(t, 0.004, 0.30, start=land)
    tail = highpass(noise(len(t), 13), 1500) * env(t, 0.002, 0.05, start=land) * 0.3
    return swoosh + hit + tail


def p_static():
    """The old CRT sound with the flyback whine removed."""
    t = t_axis(1.2)
    thump = sweep(t, 78, 34, curve=0.35) * env(t, 0.004, 0.14)
    click = noise(len(t), 17) * env(t, 0.0006, 0.006)
    burst = env(t, 0.012, 0.10, start=0.02) + 0.5 * env(t, 0.02, 0.20, start=0.28)
    hiss = lowpass(noise(len(t), 19), 6000) * np.clip(burst, 0, 1)
    return thump * 0.9 + click * 0.5 + hiss * 0.4


def reverb(x, decay=1.1, cutoff=2600, mix=0.3, seed=23):
    """Convolution with a decaying-noise impulse.

    Every other preset here is bone dry, which is most likely why they read as
    test tones rather than sounds: a real impact happens somewhere, and the
    room is a large part of what makes it sound real.
    """
    n = int(SR * decay)
    tt = np.arange(n) / SR
    ir = np.random.default_rng(seed).normal(0, 1, n) * np.exp(-tt / (decay / 4.5))
    ir = lowpass(ir, cutoff)
    ir /= max(np.abs(ir).max(), 1e-9)
    wet = np.convolve(x, ir)[:len(x)]
    wet /= max(np.abs(wet).max(), 1e-9)
    wet *= max(np.abs(x).max(), 1e-9)
    return (1 - mix) * x + mix * wet


def p_impact():
    """Low impact with a room around it. Layered and saturated, not a tone."""
    t = t_axis(1.6)
    sub = sweep(t, 66, 30, curve=0.3) * env(t, 0.005, 0.38)
    body = lowpass(noise(len(t), 29), 260) * env(t, 0.004, 0.16) * 1.2
    air = highpass(noise(len(t), 31), 1800) * env(t, 0.001, 0.03) * 0.35
    dry = np.tanh((sub + body) * 1.5) * 0.85 + air
    return reverb(dry, decay=1.2, cutoff=2200, mix=0.32)


def p_whine():
    """The original: thump, static and a flyback tone. Kept for A/B."""
    t = t_axis(2.2)
    base = p_static()
    base = np.pad(base, (0, len(t) - len(base)))
    slide = np.clip((t - 0.05) / 0.12, 0, 1)
    w = np.sin(2 * np.pi * np.cumsum(15734 / 4 * (0.45 + 0.55 * slide)) / SR)
    w *= np.clip((t - 0.05) / 0.10, 0, 1) * np.exp(-np.maximum(t - 0.55, 0) / 0.5)
    return base + 0.07 * w


PRESETS = {"thump": p_thump, "boom": p_boom, "switch": p_switch, "impact": p_impact,
           "blip": p_blip, "riser": p_riser, "static": p_static,
           "whine": p_whine}


def render(name):
    mono = PRESETS[name]()
    mono = mono / max(np.abs(mono).max(), 1e-9) * 0.85
    fade = np.clip(np.linspace(len(mono) / SR, 0, len(mono)) / 0.08, 0, 1)
    mono = mono * fade
    return np.stack([mono, mono], axis=1)


def write(stereo, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((stereo * 32767).astype("<i2").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=None, choices=sorted(PRESETS))
    ap.add_argument("--demo", action="store_true", help="write every preset")
    ap.add_argument("--out", default=str(ASSETS / "crt-power-on.wav"))
    args = ap.parse_args()

    if args.demo or not args.preset:
        for name in sorted(PRESETS):
            s = render(name)
            p = SFX_DIR / f"sfx-{name}.wav"
            write(s, p)
            print(f"[sfx] {name:7s} {len(s) / SR:4.2f}s -> {p}")
        return

    write(render(args.preset), Path(args.out))
    print(f"[sfx] {args.preset} -> {args.out}")


if __name__ == "__main__":
    main()
