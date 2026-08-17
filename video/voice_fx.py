"""Menacing-machine voice processing for the narration.

Built from synthesis, not from anyone's voice: pitch and body shifting,
ring modulation, comb resonance, a detuned swarm layer, drive and space.
That is where the "cold intelligence" character actually comes from — you
do not need someone else's larynx to get it.

Applied to the full narration AFTER forced alignment, so onsets never move
and the viseme timeline stays exactly valid.

  ../jarvis-agent/.venv/bin/python video/voice_fx.py \
      --in work/audio.wav --preset overlord --out work/audio_fx.wav
  ../jarvis-agent/.venv/bin/python video/voice_fx.py \
      --in work/audio.wav --demo work/demo/ --seconds 9
"""

import argparse
import math
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as AF

# Every value is a processing choice, not a resemblance target.
PRESETS = {
    # No processing — the plain cloned voice.
    "clean": {},

    # Light machine: a hint of metal, still clearly a person.
    "machine": {
        "pitch": -1.0, "ring_depth": 0.06, "ring_hz": 62.0,
        "comb_ms": 2.6, "comb_mix": 0.12, "drive": 1.4,
        "swarm": [], "reverb": 0.10, "reverb_decay": 0.7,
        "hp": 95, "mud_db": -2.5, "presence_db": 2.5, "lp": 9500,
    },

    # The news anchor as cold intelligence: deeper, wider, unhurried.
    "overlord": {
        "pitch": -2.0, "ring_depth": 0.11, "ring_hz": 54.0,
        "comb_ms": 3.4, "comb_mix": 0.20, "drive": 2.1,
        # (delay ms, detune semitones, gain) — a small chorus of itself.
        "swarm": [(14.0, -0.10, 0.30), (27.0, 0.12, 0.22)],
        "reverb": 0.12, "reverb_decay": 0.7,
        "hp": 90, "mud_db": -4.0, "presence_db": 3.5, "lp": 8800,
    },

    # More machine than voice: heavy modulation, hollow, distant.
    "swarm": {
        "pitch": -4.5, "ring_depth": 0.20, "ring_hz": 44.0,
        "comb_ms": 5.0, "comb_mix": 0.30, "drive": 2.8,
        "swarm": [(11.0, -0.22, 0.34), (23.0, 0.18, 0.30), (38.0, -0.35, 0.22)],
        "reverb": 0.30, "reverb_decay": 1.5,
        "hp": 85, "mud_db": -5.0, "presence_db": 4.0, "lp": 8000,
    },
}


def pitch_shift(wav, sr, semitones):
    if abs(semitones) < 0.01:
        return wav
    return torchaudio.transforms.PitchShift(sr, semitones)(wav)


def ring_mod(wav, sr, depth, hz):
    """Amplitude modulation in the tens of hertz. Too deep and it buzzes into
    nonsense; this stays shallow so speech reads as speech with a metal edge."""
    if depth <= 0:
        return wav
    t = torch.arange(wav.shape[-1], dtype=torch.float32) / sr
    return wav * (1.0 - depth + depth * torch.sin(2 * math.pi * hz * t))


def comb(wav, sr, delay_ms, mix):
    """A few milliseconds of feedback-free delay adds a fixed resonance —
    the hollow, cabinet-like ring of something speaking through hardware."""
    if mix <= 0:
        return wav
    d = int(sr * delay_ms / 1000)
    delayed = torch.nn.functional.pad(wav, (d, 0))[..., :wav.shape[-1]]
    return wav + mix * delayed


def swarm(wav, sr, layers):
    """Copies of itself, delayed and detuned. Reads as more-than-one-speaker
    without ever being a second speaker."""
    out = wav.clone()
    for delay_ms, detune, gain in layers:
        layer = pitch_shift(wav, sr, detune)
        d = int(sr * delay_ms / 1000)
        layer = torch.nn.functional.pad(layer, (d, 0))[..., :wav.shape[-1]]
        out = out + gain * layer
    return out


def reverb(wav, sr, mix, decay):
    """Convolution with a synthetic decaying-noise impulse: gives the voice a
    room, which is most of what makes something sound large."""
    if mix <= 0:
        return wav
    n = int(sr * decay)
    t = torch.arange(n, dtype=torch.float32) / sr
    gen = torch.Generator().manual_seed(5)
    ir = torch.randn(n, generator=gen) * torch.exp(-t / (decay / 4.5))
    ir = AF.lowpass_biquad(ir.unsqueeze(0), sr, 3800)
    ir = ir / ir.abs().max().clamp(min=1e-9)
    wet = AF.fftconvolve(wav, ir, mode="full")[..., :wav.shape[-1]]
    wet = wet / wet.abs().max().clamp(min=1e-9) * wav.abs().max()
    return (1 - mix) * wav + mix * wet


def apply_preset(wav, sr, name):
    p = PRESETS[name]
    if not p:
        return wav
    peak = wav.abs().max().clamp(min=1e-6)

    out = pitch_shift(wav, sr, p["pitch"])
    if p["swarm"]:
        out = swarm(out, sr, p["swarm"])
    out = ring_mod(out, sr, p["ring_depth"], p["ring_hz"])
    out = comb(out, sr, p["comb_ms"], p["comb_mix"])

    # Soft clip: menace lives in the harmonics, not in the level.
    out = torch.tanh(out / out.abs().max().clamp(min=1e-6) * p["drive"])

    out = AF.highpass_biquad(out, sr, p["hp"])
    out = AF.equalizer_biquad(out, sr, 350.0, p["mud_db"], Q=0.9)
    out = AF.equalizer_biquad(out, sr, 2600.0, p["presence_db"], Q=0.8)
    out = AF.lowpass_biquad(out, sr, p["lp"])
    out = reverb(out, sr, p["reverb"], p["reverb_decay"])

    return out / out.abs().max().clamp(min=1e-9) * peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--preset", default="overlord", choices=sorted(PRESETS))
    ap.add_argument("--out")
    ap.add_argument("--demo", help="write one clip per preset into this dir")
    ap.add_argument("--seconds", type=float, default=9.0, help="demo clip length")
    args = ap.parse_args()

    wav, sr = torchaudio.load(args.src)
    wav = wav.mean(dim=0, keepdim=True)

    if args.demo:
        out_dir = Path(args.demo)
        out_dir.mkdir(parents=True, exist_ok=True)
        clip = wav[..., :int(sr * args.seconds)]
        for name in sorted(PRESETS):
            path = out_dir / f"voice-{name}.wav"
            torchaudio.save(str(path), apply_preset(clip, sr, name), sr)
            print(f"[fx] {name:9s} -> {path}")
        return

    if not args.out:
        raise SystemExit("pass --out or --demo")
    torchaudio.save(str(args.out), apply_preset(wav, sr, args.preset), sr)
    print(f"[fx] {args.preset} -> {args.out}")


if __name__ == "__main__":
    main()
