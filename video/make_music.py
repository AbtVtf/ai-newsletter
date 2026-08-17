"""Compose the Daily Prompt music bed as an LMMS project, then render it.

Ominous space ambience: a D drone under slow pads that drift through
D Phrygian, plus a sparse sonar ping. Everything is long and sustained —
no drums, nothing that competes with a voice.

The project is written as a real .mmp, so it opens in the LMMS GUI and can
be edited by hand afterwards. Rendering goes through the flatpak LMMS CLI.

  python3 video/make_music.py                 # write + render
  python3 video/make_music.py --no-render     # just write the .mmp
"""

import argparse
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path.home() / "Music" / "lmms-jam"

BAR = 192          # LMMS ticks per bar (48 per beat at 4/4)
BEAT = BAR // 4
BPM = 58
BARS = 32          # ~2m12s at 58 bpm; ffmpeg loops it if a video runs longer

# TripleOscillator wave indices.
SINE, TRIANGLE, SAW, SQUARE = 0, 1, 2, 3

# Quintal harmony over a D pedal: stacks of fifths and fourths, no thirds
# anywhere. The third is what carries major/minor feeling, so leaving it out
# reads as cold and vast rather than sad. Tension comes from the intervals
# instead — the sus4 stack refuses to resolve, and the Eb stack puts a flat
# second against the drone.
# MIDI note numbers, as LMMS uses them.
D1, D2 = 26, 38
PROGRESSION = [                    # (bars, voicing)
    (4, [50, 57, 62, 69]),         # D  A  D  A   open fifths — vast, neutral
    (4, [50, 55, 62, 67]),         # D  G  D  G   fourths — suspended
    (4, [51, 58, 63, 70]),         # Eb Bb Eb Bb  flat second over D — ominous
    (4, [48, 55, 60, 67]),         # C  G  C  G   modal drop, still open
]
# Same rule up here: D, G, A, C, Eb only. No F, no F#, so nothing reads minor.
AIR_NOTES = [                      # (bar, midi note) — sparse high shimmer
    (1, 74), (3, 69), (5, 79), (7, 72), (9, 81), (11, 74), (13, 75), (15, 70),
    (17, 74), (19, 79), (21, 86), (23, 74), (25, 72), (27, 79), (29, 81), (31, 74),
]
# A slow low pulse every four bars: forward motion and unease, where a static
# pad alone would just sit there and mope.
PULSE_BARS = list(range(0, 32, 4))
PULSE_NOTE = 38
PING_BARS = [3, 11, 19, 27]
PING_NOTE = 86


def osc(**kw):
    """TripleOscillator element. Unspecified oscillators stay silent."""
    defaults = {
        "vol0": 33, "vol1": 33, "vol2": 33, "wave0": 0, "wave1": 0, "wave2": 0,
        "coarse0": 0, "coarse1": 0, "coarse2": 0,
        "finel0": 0, "finer0": 0, "finel1": 0, "finer1": 0, "finel2": 0, "finer2": 0,
        "pan0": 0, "pan1": 0, "pan2": 0,
        "phoffset0": 0, "phoffset1": 0, "phoffset2": 0,
        "stphdetun0": 0, "stphdetun1": 0, "stphdetun2": 0,
        "modalgo1": 2, "modalgo2": 2, "modalgo3": 2,
        "userwavefile0": "", "userwavefile1": "", "userwavefile2": "",
    }
    defaults.update(kw)
    attrs = " ".join(f'{k}="{v}"' for k, v in defaults.items())
    return f'<instrument name="tripleoscillator"><tripleoscillator {attrs}/></instrument>'


def envelope(cut, res, att, dec, sus, rel, hold=0.0, cut_env=None):
    """eldata: lowpass filter plus the volume envelope (and optionally a slow
    filter-cutoff envelope, which is what makes a pad breathe)."""
    vol = (f'<elvol amt="1" pdel="0" att="{att}" hold="{hold}" dec="{dec}" '
           f'sus="{sus}" rel="{rel}" x100="0" lamt="0" latt="0" lspd="0.1" '
           f'lshp="0" lpdel="0" ctlenvamt="0"/>')
    parts = [vol]
    if cut_env:
        amt, catt, cdec, csus, crel = cut_env
        parts.append(f'<elcut amt="{amt}" pdel="0" att="{catt}" hold="0" '
                     f'dec="{cdec}" sus="{csus}" rel="{crel}" x100="0" '
                     f'lamt="0" latt="0" lspd="0.1" lshp="0" lpdel="0" '
                     f'ctlenvamt="0"/>')
    return (f'<eldata fwet="1" ftype="0" fcut="{cut}" fres="{res}">'
            + "".join(parts) + "</eldata>")


def track(name, vol, pan, instrument, eldata, notes, length):
    rows = "\n".join(
        f'        <note pos="{pos}" key="{key}" len="{ln}" vol="{nv}" pan="0"/>'
        for pos, key, ln, nv in notes)
    return f"""      <track type="0" name="{escape(name)}" muted="0" solo="0">
      <instrumenttrack pan="{pan}" fxch="0" pitch="0" pitchrange="1" basenote="57" vol="{vol}" usemasterpitch="1">
        {instrument}
        {eldata}
      </instrumenttrack>
      <pattern pos="0" len="{length}" muted="0" steps="16" name="{escape(name)}" type="1" frozen="0">
{rows}
      </pattern>
    </track>"""


def build_notes():
    total = BARS * BAR

    # Drone: two octaves of D, re-struck every 8 bars so the attack never
    # fully disappears under the pad.
    # Notes overlap into the next one everywhere, so the bed never gaps and
    # the level stays flat enough to sit under a voice.
    drone = []
    for start in range(0, BARS, 8):
        pos = start * BAR
        ln = 8 * BAR + BEAT
        drone += [(pos, D1, ln, 70), (pos, D2, ln, 52)]

    # Pad: the progression, looped, each chord held for its whole span.
    pad = []
    bar = 0
    while bar < BARS:
        for span, chord in PROGRESSION:
            if bar >= BARS:
                break
            span = min(span, BARS - bar)
            pos, ln = bar * BAR, span * BAR + BEAT
            for i, key in enumerate(chord):
                pad.append((pos, key, ln, 52 - i * 4))
            bar += span

    air = [(b * BAR, key, BAR * 2, 46) for b, key in AIR_NOTES if b < BARS]
    ping = [(b * BAR + BAR // 2, PING_NOTE, BEAT, 40) for b in PING_BARS if b < BARS]
    pulse = [(b * BAR, PULSE_NOTE, BEAT * 2, 44) for b in PULSE_BARS if b < BARS]
    return drone, pad, air, ping, pulse, total


def build_project():
    drone, pad, air, ping, pulse, total = build_notes()

    tracks = [
        # Sub drone: sine plus a soft octave, slow swell, heavily filtered.
        # Sits under the voice rather than in front of it.
        track("Drone", 82, 0,
              osc(vol0=58, vol1=24, vol2=0, wave0=SINE, wave1=SINE, coarse1=12,
                  stphdetun1=8),
              envelope(340, 0.18, 0.6, 0.9, 1.0, 0.95),
              drone, total),
        # Pad: detuned saws with a slow cutoff envelope so each chord opens.
        track("Pad", 66, 0,
              osc(vol0=30, vol1=26, vol2=20, wave0=SAW, wave1=SAW, wave2=SINE,
                  coarse2=-12, finer0=5, finel1=-6,
                  stphdetun0=14, stphdetun1=18, pan0=-28, pan1=28),
              envelope(1150, 0.32, 0.55, 0.9, 0.95, 0.95,
                       cut_env=(0.42, 0.8, 0.9, 0.6, 0.9)),
              pad, total),
        # Air: high, quiet, wide. The "space" in space music.
        track("Air", 72, 0,
              osc(vol0=34, vol1=26, vol2=0, wave0=TRIANGLE, wave1=SINE, coarse1=12,
                  stphdetun0=26, stphdetun1=30, pan0=-45, pan1=45),
              envelope(5200, 0.12, 0.7, 0.95, 0.85, 0.95),
              air, total),
        # Ping: a sonar blip, long decay, far away.
        track("Ping", 34, 0,
              osc(vol0=40, vol1=0, vol2=0, wave0=SINE, stphdetun0=10),
              envelope(3200, 0.4, 0.01, 0.95, 0.0, 0.9),
              ping, total),
        # Pulse: a soft low hit every four bars. Movement, not melody.
        track("Pulse", 44, 0,
              osc(vol0=48, vol1=20, vol2=0, wave0=SINE, wave1=TRIANGLE,
                  coarse1=-12),
              envelope(240, 0.2, 0.02, 0.45, 0.0, 0.6),
              pulse, total),
    ]

    return f"""<?xml version="1.0"?>
<!DOCTYPE lmms-project>
<lmms-project version="1.0" type="song" creator="LMMS" creatorversion="1.2.2">
  <head timesig_numerator="4" timesig_denominator="4" bpm="{BPM}" masterpitch="0" mastervol="82"/>
  <song>
    <trackcontainer type="song" width="800" height="400" x="5" y="5" visible="1" maximized="0" minimized="0">
{chr(10).join(tracks)}
    </trackcontainer>
  </song>
</lmms-project>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="daily-prompt-bed")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mmp = OUT_DIR / f"{args.name}.mmp"
    mmp.write_text(build_project())
    print(f"wrote {mmp}  ({BARS} bars @ {BPM} bpm ≈ {BARS * 4 * 60 / BPM:.0f}s)")
    if args.no_render:
        return

    wav = OUT_DIR / f"{args.name}.wav"
    cmd = ["flatpak", "run", "io.lmms.LMMS", "render", str(mmp),
           "-o", str(wav), "-f", "wav", "-s", "44100"]
    print("+ " + " ".join(cmd))
    if subprocess.run(cmd).returncode or not wav.exists():
        sys.exit("LMMS render failed")

    # LMMS renders this very dynamic and very bass-heavy. Under a voice that
    # would pump and muddy the low mids, so: carve 250 Hz, lift the top for
    # air, compress hard enough to sit flat, then normalize.
    chain = ("highpass=f=32,"
             "equalizer=f=250:width_type=o:width=1.4:g=-6,"
             "equalizer=f=3200:width_type=o:width=2:g=4,"
             "acompressor=threshold=-32dB:ratio=6:attack=150:release=900:makeup=3,"
             "loudnorm=I=-20:TP=-3:LRA=6,"
             "alimiter=limit=0.9")
    dest = ROOT / "video" / "assets" / f"{args.name}.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                    "-af", chain, "-ar", "44100", str(dest)], check=True)
    print(f"rendered -> {dest}")


if __name__ == "__main__":
    main()
