# Shorts pipeline

Turns an edition article into a vertical video: a green-phosphor GNM head
reads a shortened version of the story out loud, lip-synced, over the Daily
Prompt masthead, behind a CRT treatment, over an ominous synth bed.

```
output/<date>/articles.json
   │
   ▼ script_writer.py   (newsletter venv — writer model via OpenRouter)
script.txt   ~90 spoken words, plain modern English
script.json  title, on-screen overlay, social caption + hashtags
   │
   ▼ voice_track.py     (jarvis-agent venv — Chatterbox + MMS_FA)
audio.wav      narration, voice cloned from jarvis data/voice/reference.wav
timeline.json  viseme segments + per-sentence time spans (for captions)
   │
   ▼ render_face.py     (jarvis-agent venv — moderngl headless EGL + ffmpeg)
   ▼ make_thumb.py      (newsletter venv — PIL only, no GPU)
short.mp4        1080x1920, 30 fps, h264 + aac, normalized to -14 LUFS
thumb.png        1080x1920, no face, hook text carries the frame
description.txt  ready to paste, with the source URL and hashtags
```

`design.py` holds the frame size, palettes, layout, type and CRT treatment.
Both the video and the thumbnail import it, so the two cannot drift apart.

## Run

```sh
.venv/bin/python -m video.make_short --date 2026-08-08 --story story-1
.venv/bin/python -m video.make_short --date 2026-08-08 --all
```

Stages are skipped when their output already exists. `--force` redoes them.
To re-render only the picture, call `render_face.py` directly — it re-uses
the existing audio and timeline, so framing, theme, speed, grain and music
are all tunable without paying for TTS again.

`render_face.py --still 6.0 --out frame.png` renders one frame instead of the
video. Use it when tuning layout; it takes a second instead of 80.

## Two venvs

The newsletter venv has `requests`/`jinja2`. Chatterbox, torch and the
aligner live in `../jarvis-agent/.venv` (~17 GB of models, CUDA). `make_short.py`
calls each stage with the right interpreter, so you only ever run the
newsletter one.

## Tuning

| Flag | Default | Effect |
|---|---|---|
| `--min-words` / `--max-words` | 85 / 110 | script length; ~120 spoken words per minute before `--speed` |
| `--speed` | 1.15 | playback rate; scales the viseme timeline and atempos the voice |
| `--theme` | `phosphor` | `phosphor` (green CRT) or `digital` (the Jarvis blue) |
| `--grain` | 6.0 | film grain sigma; 0 disables |
| `--glitch` | 1.0 | glitch burst frequency multiplier; 0 disables |
| `--pulse` | 3.6 | headline breathing period in seconds; 0 disables |
| `--music` | `assets/daily-prompt-bed.wav` | bed under the narration; `""` for none |
| `--music-gain` | -19 dB | how far under the voice the bed sits |
| `--scene-sfx-gain` | -6 dB | event sounds (zoom, arm, ratchet, click); `--no-scene-sfx` disables |
| `--head-height` | 0.34 | head size as fraction of frame height |
| `--head-y` | 0.50 | head center as fraction of frame height, from the top |
| `--exaggeration` | 0.35 | Chatterbox emotion intensity |
| `--cfg` | 0.45 | delivery pacing; lower is slower and more deliberate |
| `--fx` | `clean` | voice character: `clean`, `machine`, `overlord`, `swarm` |

`render_face.py` also has `--no-crt-intro`, `--scanline` and `--exposure`.
Overlay layout constants are at the top of that file under `# Layout`.

## How the picture is built

`render_face.py` reads a packed head — 17,821 vertices, 33,788 triangles and
19 morph targets (15 Oculus visemes, blink, three gaze shapes). The GLSL
shader reproduces the three.js light rig from `jarvis-agent/web/app.js`
(hemisphere + key + two rims, ACES tone mapping) with the light colors as
uniforms, and the animation loop reproduces its `tick()`: viseme
attack/release smoothing, the randomized blink schedule, and the head sway
that widens while speaking.

`export_head.py` writes our green copy of that head into `video/assets/`,
reusing the viseme morph targets Jarvis already solved. Jarvis stays blue;
run it again if you change the palette.

The CRT treatment is four parts.

**Power-on**, first 1.25 s: a bright line snaps across the middle, the picture
unfolds vertically out of it, then the overbright settles with a mains
flicker. `make_sfx.py` synthesizes the matching sound — degauss thump, static
burst, and a flyback whine sliding up to pitch. The real flyback frequency is
15.734 kHz, which most people cannot hear and most codecs discard, so it is
pitched two octaves down.

**Scanlines and vignette**, static, precomputed once.

**Grain**, generated at a third of the resolution and upscaled. Per-pixel
noise looks like a dirty sensor rather than film, and being incompressible it
costs more bitrate than the entire rest of the picture — the first version
produced a 436 MB file for 40 seconds. Even coarse, grain dominates the
encode, hence `-crf 22 -maxrate 8M`.

**Glitch bursts**, 2-5 frames each, a few seconds apart, on a seeded schedule
so a re-render is identical. Each burst combines horizontal slice
displacement, chroma split, an occasional vertical roll and a brightness pop.
Every frame inside a burst re-rolls its own parameters: holding one
distortion still for several frames reads as a rendering mistake, while
changing it each frame reads as interference. The audio runs straight
through, which is what makes it a transmission fault rather than a broken
render.

The headline sits on its own layer so its alpha can breathe on a 3.6 s cycle,
independent of the caption underneath it.

Render is roughly 80 s for a 40 s clip on the 3090.

## Music

`make_music.py` writes `~/Music/lmms-jam/daily-prompt-bed.mmp` and renders it
through the flatpak LMMS CLI. It is a real project file, so you can open and
edit it in the LMMS GUI.

Five tracks, 32 bars at 58 bpm (about 2m20s; ffmpeg loops it if a video runs
longer): a sine drone on D, a detuned saw pad, a sparse high shimmer, a sonar
ping every eight bars, and a low pulse every four.

The harmony is **quintal — stacks of fifths and fourths, with no thirds
anywhere**. The third is what carries major/minor feeling, so leaving it out
reads as cold and vast instead of sad. The first version used D minor triads
and came out mournful. Tension now comes from intervals instead: the sus4
stack refuses to resolve, and the Eb stack puts a flat second against the D
drone. Every voice stays on D, G, A, C or Eb — no F, so nothing can read as
minor.

LMMS renders it very dynamic and very bass-heavy, so `make_music.py`
post-processes: cut 250 Hz, lift 3.2 kHz, compress hard, normalize. That took
the level range from 58 dB to 21 dB, which is what lets it sit under a voice
without pumping.

Two things to know if you edit the .mmp by hand: LMMS ticks are **192 per
bar**, not per beat, and TripleOscillator wave indices are
0=sine, 1=triangle, 2=saw, 3=square.

## Thumbnails and publishing metadata

`script_writer.py` produces the publishing fields alongside the script:
`title`, `overlay` (the on-screen headline), `hook` (the thumbnail one-liner),
`kicker` (a small section label), `description` and `hashtags`.

`--metadata-only` regenerates just those fields for shorts whose audio
already exists. It reads the recorded script and never rewrites it, because
the narration and the viseme timeline were built from that exact text:

```sh
.venv/bin/python -m video.script_writer --date 2026-08-08 --metadata-only
```

`description.txt` is assembled in code, not by the model. The source URL is
pasted in from the edition data so it cannot be invented — the model only
writes the prose, and it is told not to produce links.

`make_thumb.py` renders the thumbnail: same masthead, palette and CRT
treatment as the video, no face, with the hook as the hero. The hook block —
kicker, hook, rule, source — is measured and centred as one unit in the space
the head occupies in the video, so a two-line hook and a four-line hook both
sit balanced. Type shrinks from 68pt until the hook fits in four lines rather
than dropping a line, because a truncated hook is a wrong hook.

**Check hooks against the source before posting.** Hooks compress hardest and
are the most likely field to overstate. In the first batch one read
"10 TRILLION PARAMETER AI MODEL" for a model the article said "may reach"
that size in early training — true as a target, false as a fact.

## Voice character

`voice_fx.py` turns the narration into a machine voice through processing
alone — pitch and body shifting, shallow ring modulation, a comb resonance,
a detuned swarm of itself, soft-clip drive, and convolution reverb. Four
presets, increasingly synthetic: `clean`, `machine`, `overlord`, `swarm`.
On `overlord` the fundamental drops from 107 Hz to 91 Hz and the energy
above 2.5 kHz roughly triples.

Audition them without re-rendering anything:

```sh
../jarvis-agent/.venv/bin/python video/voice_fx.py \
    --in output/<date>/shorts/story-1/audio.wav \
    --demo output/<date>/shorts/story-1/voice-demo --seconds 10
```

Two things make this cheap to iterate on. The FX run on the finished
narration rather than inside TTS, so switching presets costs seconds instead
of a full re-synthesis. And they run *after* forced alignment, and every stage
preserves length, so onsets never move and the viseme timeline stays valid —
verified by comparing speech onsets before and after.

Delivery matters as much as processing: lower `--cfg` makes the read slower
and more deliberate, lower `--exaggeration` makes it flatter and colder.

**On cloning voices.** `data/voice/reference.wav` in jarvis-agent is what
Chatterbox clones, and it should only ever be a voice you have the right to
use — your own, or one you have permission for. Do not point it at a film,
a show, or a recording of a public figure: that is someone's identifiable
performance and usually someone else's copyrighted audio. If you want a
specific character, record yourself performing it and clone that, then stack
these presets on top. You get the delivery you wanted and you own it.

## Sound effects

`make_sfx.py` synthesizes `assets/crt-power-on.wav` with numpy — no samples,
no licensing to worry about. It plays at t=0 under the visual power-on and
mixes at -7 dB.

The mix uses `amix=normalize=0`. With amix's default the level would jump
upward the moment the 2 s power-on sound ends, because it divides by the
number of live inputs. Everything lands at -14 LUFS, which is what the
platforms normalize to anyway.
