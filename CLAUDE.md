# The Daily Prompt — working notes

Two products from one repo:

1. **The newspaper** — `pipeline/` builds a daily 1-bit pixel-art broadsheet.
   See `README.md`. Unchanged by the video work except that `render.py` now
   also writes `articles.json`, which is what the video stage reads.
2. **The shorts** — `video/` turns each article into a vertical video where a
   green-phosphor GNM head reads a shortened version of the story from inside
   a containment tube. See `video/README.md` for the flags.

## Run it

```sh
.venv/bin/python -m pipeline.main                       # build the edition
.venv/bin/python -m video.make_short --date 2026-08-08 --all --fx overlord
```

`make_short` chains four stages, each in the venv that has its dependencies,
and skips stages whose output already exists (`--force` redoes them):

| Stage | Venv | Output |
|---|---|---|
| `script_writer` | this repo's `.venv` | `script.txt`, `script.json`, `description.txt` |
| `voice_track` | `../jarvis-agent/.venv` | `audio.wav`, `timeline.json` |
| `voice_fx` | `../jarvis-agent/.venv` | `audio-<preset>.wav` |
| `render_face` | `../jarvis-agent/.venv` | `short.mp4` |
| `make_thumb` | this repo's `.venv` | `thumb.png` |

**Two venvs.** This repo's has `requests`/`jinja2`/`PIL`. Chatterbox, torch,
the aligner, moderngl and trimesh live in `../jarvis-agent/.venv` (~17 GB of
models, CUDA). Never merge them; `make_short` calls each stage with the right
interpreter.

## Iterating on the look

Full renders are slow and burn goodwill. Use these instead:

```sh
# one frame, ~1 s
render_face.py --still 6.0 --out frame.png
# 5 s of the set alone, no masthead, no audio
render_face.py --bare --max-seconds 5 --music "" --sfx "" --out preview.mp4
```

**Measure before changing anything visual.** Nearly every bug below was found
by printing numbers and would have cost several wrong guesses otherwise. Ask
"is the thing on screen?" in world units, not by squinting at a thumbnail.

## Editorial rules that matter

- **A capability is not an event.** If a source says a model *could* do
  something in testing, never write that it *did*, escaped, or attacked. Never
  move an event from one model onto another. This is the single most common
  way AI news gets distorted and it is enforced in both prompts in
  `script_writer.py`.
- **Hooks compress hardest and overstate first.** Check every number in a hook
  against the article body before posting.
- **Never clone a voice you do not have rights to.** `reference.wav` in
  jarvis-agent is what Chatterbox clones. Not films, not shows, not public
  figures. Record yourself and stack `voice_fx.py` presets on top.
- Descriptions are assembled in code; the source URL is pasted from edition
  data and never asked of the model, so it cannot be invented.

## Scene architecture

`render_face.py` renders the head with moderngl on headless EGL and pipes
frames to ffmpeg. The head is the packed GNM mesh (17,821 verts, 19 morph
targets); the shader reproduces the three.js light rig from
`jarvis-agent/web/app.js`, and the animation reproduces its `tick()`.

Everything else is **procedural** — `set_pieces.py` (tube, wall, greebles,
screens, LEDs) and `cables.py`. Only the helmet uses TRELLIS.

Groups move independently: the **wall** is bolted to the world; the **rig**,
**glass**, bust and cables ride the opening spin; the **cap** breaks away; the
**arm** and **helmet** have their own transforms.

`scene_sfx.py` synthesizes per-event sounds (zoom whir, servo, ratchet,
rumble, seat click) from the SAME beat constants after they are fitted, so
audio stays synced to the motion at any clip length. Subtle by design:
events sit 10-26 dB under the narration at the default `--scene-sfx-gain
-6`. All synthesized — no sample licences to track.

Beat times are derived from the clip length in `configure_sequence()`, so a
20 s and a 30 s story both get a proportioned sequence. The sequence is:
rise + spin (locked to the camera push) → arm descends → unscrews cap → cap
lifts away → water drains → cables detach and wiggle → tube slides out →
helmet lowers onto the head.

## Gotchas — all of these cost real time

**`moderngl.Context` has no `depth_mask`.** Assigning `ctx.depth_mask = False`
silently creates a plain Python attribute and changes no GL state. It lives on
the **Framebuffer**. This made the glass write depth and reject every bubble
behind it, with no error.

**Rotate about the tube's axis, not the origin.** Everything is built around
`z = 0.0271`, so a bare `rot_y()` orbits the piece around a pole 27 mm away.
Use `spin()`.

**glTF +Y-up negates Blender's Y into renderer Z.** Blender `y = -0.36` lands
at renderer `z = +0.36` — in front of the camera. `build_scene.py` has `by()`.

**Alpha blending needs int32.** `frame * (255 - a)` reaches 65,025 and wraps
silently in int16.

**Tie motion to the frame, not to world units, when the camera is moving.**
The opening push shrinks the visible height ~4×; a fixed drop left the tube
climbing into a receding frame and it vanished for most of the move. See
`rise_at()`.

**Film grain dominates the encode.** Per-pixel noise produced a 436 MB file
for 40 s. Generate it coarse and upscale; keep `-crf 22 -maxrate 8M`.

**`amix` normalizes by input count** — the level jumps when the 2 s power-on
sound ends. Use `normalize=0`.

**LMMS ticks are 192 per _bar_,** not per beat. TripleOscillator waves are
0=sine, 1=triangle, 2=saw, 3=square.

**A wiggle is displacement, not an orbit.** Rotating loose cable ends about
the tube axis with an angle that grows with time gave full revolutions and
apparent size changes. Displace; do not rotate.

**Re-synthesize when the script changes.** `make_short` compares mtimes;
without it a rewritten script silently keeps the old narration.

## TRELLIS

Only worth it for a single hero object (the helmet). It cannot tile, cannot
leave a hole for the tube, and cannot carry blinking lights, so the wall and
cables are procedural — 4k triangles versus ~1M for one generated padlock.

- **Thin protrusions blow it up.** Antennae and fine panel lines produced
  16.4 M faces and OOM'd mesh extraction on a clean 24 GB GPU. Thick, simple,
  closed forms go through fine.
- **Check the raw width/height ratio before fitting.** A head is ~1.19. A mesh
  at 0.83 is a full-face design with a long jaw: scaled by width it hangs past
  the neck, scaled by height it ends up narrower than the skull. This one
  number predicted two failed helmets.
- **The MCP server leaks VRAM.** After hours resident it held 5.6 GB it no
  longer tracked and its unload reported nothing loaded. Restart the process.
  `video/blender/run_prop_gen.sh` runs the pipeline directly in the `trellis2`
  conda env and does not need MCP at all.
- The call is `pipeline.run(image)[0]`, then `mesh.simplify()`, then
  `o_voxel.postprocess.to_glb(...)`. There is no `envmap` argument and the
  pipeline does not return a glb.
- Decimate with `fast-simplification` (installed in the jarvis venv).

## Blender

`video/blender/build_scene.py` rebuilds the set as editable objects;
`tdp_props_addon.py` adds a prompt-to-prop panel. Export to glb and render
with `--set-glb`. Keep the world origin and scale, and put "glass" in the name
of anything transparent.

The addon must strip `PYTHONHOME`/`PYTHONPATH` before shelling out — Blender
exports them and they break conda's Python — and must log to a file, not a
pipe, or it deadlocks when TRELLIS fills the buffer.

**Currently behind the renderer:** the Blender scene has the tube, wall and
greebles but not the separated cap, arm, water surface or screens. Sync it
before doing scene work in Blender, or treat the procedural scene as the
source of truth.

## The actor stage (ARDY)

`../ardy` (NVIDIA ARDY, text-to-motion) supplies a skinned human actor for
cutaway inserts. `video/actor.py` does the LBS from a generated npz
(`global_rot_mats` + `posed_joints` against the cskel27 skin);
`video/render_actor.py` renders inserts in the house look (chrome, captions,
CRT post); `video/splice_inserts.py` overlays them onto a finished short at
sentence windows while the audio runs through.

- **Generate with the encoder server up** (`scripts/run_text_encoder_server.py`
  in ardy) and `LLM2VEC_BASE=$PWD/models/llm2vec-mntp-merged` set, or it
  re-downloads 14 GB from HF instead of using the local copy.
- **Measure the motion before rendering it.** "Types on a keyboard" produced
  4 mm of hand movement — invisible, and cskel27 has no fingers. "Pounds
  with both fists" moves the hands 0.8-1.0 m. Print the std dev first.
- **Harvest first, then place props.** Hand-authored poses miss by 55-76 cm
  (ardy reach tests). Generate the motion, find where the hands actually go,
  put the TRELLIS/procedural prop there.
- **Props parent to the hand midpoint from the grasp frame** — and stay
  glued when the model "finishes" the action and opens its arms. Until
  there is a release rule, only splice windows that end before the release.
- **Model status:** core is the only skinned option. soma: no public
  checkpoint. G1 robot: checkpoint but no skin — needs a rigid-body loader
  for its MuJoCo meshes (~a day; fits the lab aesthetic).

## Known-open

- The arm reads as a fitting lowering onto the cap rather than an articulated
  arm; it needs a visible elbow.
- The helmet's visor does not read as glass — the shader ignores the mesh
  texture, so it shades as another facet. Darken vertices in that height band.
- Only `story-6` has the full sequence. The others need re-rendering.
- Retention data from 3 uploads: ~100 views, 4% completion, 2 s average watch.
  That drove the shorter scripts, the harder openings and the title-card cut.
  Change one variable at a time from here so the numbers mean something.
