"""Scene-synced sound effects, synthesized per clip.

One function per visual event. render_face.py builds the track AFTER
configure_sequence() has fitted the beats to the clip, so every sound starts
and ends exactly with the motion it belongs to — no hand-syncing, and a
20-second story and a 30-second one both stay aligned.

Subtle by design: event peaks sit around 0.2 against a narration that peaks
near 0.9, and the default mix gain drops them another 6 dB. These are felt
more than heard.
"""

import wave

import numpy as np

SR = 44100


def _noise(n, seed):
    return np.random.default_rng(seed).normal(0.0, 1.0, n)


def _lowpass(x, cutoff):
    a = float(np.exp(-2 * np.pi * cutoff / SR))
    out = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc = (1 - a) * v + a * acc
        out[i] = acc
    return out


def _highpass(x, cutoff):
    return x - _lowpass(x, cutoff)


def _fade(n, up, down):
    """Linear rise over `up` samples, fall over `down`."""
    env = np.ones(n)
    up = min(up, n)
    down = min(down, n)
    env[:up] = np.linspace(0, 1, up)
    env[n - down:] = np.minimum(env[n - down:], np.linspace(1, 0, down))
    return env


def zoom_whir(dur, seed=31):
    """Camera push: low motor whirring. Band-limited noise with a slow
    flutter, plus a faint tone rising as the move accelerates."""
    n = int(SR * dur)
    t = np.arange(n) / SR
    body = _highpass(_lowpass(_noise(n, seed), 230), 70)
    body *= 1.0 + 0.30 * np.sin(2 * np.pi * 24 * t)
    tone = np.sin(2 * np.pi * (54 + 16 * t / dur) * t) * 0.5
    out = (0.72 * body + 0.28 * tone) * _fade(n, int(0.25 * SR), int(0.6 * SR))
    return out / max(np.abs(out).max(), 1e-9) * 0.20


def arm_descend(dur, seed=37):
    """Servo whine sliding down as the arm lowers."""
    n = int(SR * dur)
    t = np.arange(n) / SR
    f = 640 - 180 * (t / dur)
    whine = np.sin(2 * np.pi * np.cumsum(f) / SR)
    whine *= 1.0 + 0.18 * np.sin(2 * np.pi * 47 * t)
    hiss = _highpass(_lowpass(_noise(n, seed), 900), 300) * 0.4
    out = (whine * 0.6 + hiss) * _fade(n, int(0.15 * SR), int(0.35 * SR))
    return out / max(np.abs(out).max(), 1e-9) * 0.16


def unscrew(dur, seed=41):
    """The cap turning: a slow ratchet — soft periodic clicks over a low
    rotation hum. Clicks at 5 Hz read as heavy machinery; 12 Hz reads as a
    toy."""
    n = int(SR * dur)
    t = np.arange(n) / SR
    hum = np.sin(2 * np.pi * 82 * t) * 0.5
    clicks = np.zeros(n)
    step = int(SR / 5.0)
    burst = int(0.012 * SR)
    rng = np.random.default_rng(seed)
    for start in range(int(0.05 * SR), n - burst, step):
        clicks[start:start + burst] += (
            _highpass(rng.normal(0, 1, burst), 1400) * np.linspace(1, 0, burst))
    out = (hum + clicks * 0.8) * _fade(n, int(0.1 * SR), int(0.3 * SR))
    return out / max(np.abs(out).max(), 1e-9) * 0.17


def tube_sink(dur, seed=43):
    """The rig sliding away: descending rumble that dies out."""
    n = int(SR * dur)
    t = np.arange(n) / SR
    f = 64 - 26 * (t / dur)
    rumble = np.sin(2 * np.pi * np.cumsum(f) / SR)
    grit = _lowpass(_noise(n, seed), 140) * 0.7
    out = (rumble * 0.6 + grit) * _fade(n, int(0.3 * SR), int(1.0 * SR))
    return out / max(np.abs(out).max(), 1e-9) * 0.20


def helmet_descend(dur, seed=47):
    """Hydraulic lowering: like the arm but deeper and smoother."""
    n = int(SR * dur)
    t = np.arange(n) / SR
    f = 420 - 150 * (t / dur)
    whine = np.sin(2 * np.pi * np.cumsum(f) / SR)
    air = _highpass(_lowpass(_noise(n, seed), 700), 200) * 0.5
    out = (whine * 0.55 + air) * _fade(n, int(0.2 * SR), int(0.25 * SR))
    return out / max(np.abs(out).max(), 1e-9) * 0.15


def seat_click(seed=53):
    """The helmet locking on: click, clunk, and a short damped ring."""
    n = int(0.55 * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    click = _highpass(rng.normal(0, 1, int(0.004 * SR)), 2000)
    out[:len(click)] += click * 0.9
    clunk = np.sin(2 * np.pi * 150 * t) * np.exp(-t / 0.05)
    ring = np.sin(2 * np.pi * 1150 * t) * np.exp(-t / 0.07) * 0.25
    out += clunk + ring
    return out / max(np.abs(out).max(), 1e-9) * 0.30


def build_track(duration, beats):
    """Assemble the full-length mono track from (name, t0, t1) beats.

    `beats` times are in CLIP seconds (title card offset already applied).
    Unknown names are ignored so the caller can pass beats for events that a
    particular clip does not have.
    """
    total = np.zeros(int(SR * duration) + SR)

    def place(sig, t0):
        i = int(t0 * SR)
        if i < 0 or i >= len(total):
            return
        seg = sig[:len(total) - i]
        total[i:i + len(seg)] += seg

    for name, t0, t1 in beats:
        dur = max(t1 - t0, 0.1)
        if name == "zoom":
            place(zoom_whir(dur), t0)
        elif name == "arm_down":
            place(arm_descend(dur), t0)
        elif name == "unscrew":
            place(unscrew(dur), t0)
        elif name == "tube_sink":
            place(tube_sink(dur), t0)
        elif name == "helmet_down":
            place(helmet_descend(dur), t0)
        elif name == "seat_click":
            place(seat_click(), t0)

    peak = np.abs(total).max()
    if peak > 0.5:                     # events may overlap; never let the sum spike
        total *= 0.5 / peak
    return total[:int(SR * duration)]


def save(track, path):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((np.clip(track, -1, 1) * 32767).astype("<i2").tobytes())
