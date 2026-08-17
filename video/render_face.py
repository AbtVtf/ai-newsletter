"""Viseme timeline + audio -> vertical short of the Daily Prompt anchor.

Loads the packed GNM head (positions/colors/triangles/morph deltas), replays
the web/app.js animation model (viseme attack/release, blink schedule, head
sway), shades it with a green-phosphor light rig in moderngl (headless EGL),
overlays the masthead/headline/captions, puts it behind a CRT treatment
(power-on sweep, scanlines, vignette, film grain), and pipes frames into
ffmpeg with the narration and an optional music bed.

Run with the jarvis-agent venv:
  ../jarvis-agent/.venv/bin/python video/render_face.py \
      --work-dir work/ --headline "..." --out work/short.mp4
"""

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import moderngl
import numpy as np

import set_pieces
from PIL import Image, ImageDraw

from design import (CAPTION_LEAD, CAPTION_Y, FONTS, FOOTER_Y, H, HEADLINE_LEAD,
                    HEADLINE_Y, SOURCE_Y, THEMES, W, CrtPost, center, font,
                    masthead, over, wrap)

ROOT = Path(__file__).resolve().parent.parent
VIDEO = Path(__file__).resolve().parent
JARVIS_ASSETS = ROOT.parent / "jarvis-agent" / "web" / "assets"

FPS = 30
# app.js animation constants
ATTACK, RELEASE, LOOKAHEAD, BLINK_LEN = 0.045, 0.075, 0.04, 0.18

VERT = """
#version 330
uniform mat4 mvp;
uniform mat3 nrm_mat;
in vec3 in_pos;
in vec3 in_nrm;
in vec3 in_col;
out vec3 v_nrm;
out vec3 v_col;
out vec3 v_pos;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    v_nrm = nrm_mat * in_nrm;
    v_col = in_col;
    v_pos = in_pos;
}
"""

# Reproduces the three.js MeshStandardMaterial + light rig from
# jarvis-agent/web/app.js, with the light colors lifted out as uniforms.
FRAG = """
#version 330
uniform vec3 cam_pos;
uniform float exposure;
uniform vec3 hemi_sky, hemi_ground, key_col, rim_col, rim2_col, emissive;
uniform float glass;   // >0.5: fresnel alpha, so the tube reads as glass
// The set needs to sit far below the head. Albedo alone cannot do it: the
// emissive term is a constant floor added to every surface regardless of
// colour, so without scaling it here a black wall still glows.
uniform float light_scale;
uniform float emissive_scale;
in vec3 v_nrm;
in vec3 v_col;
in vec3 v_pos;
out vec4 f_col;

const vec3 KEY_DIR = normalize(vec3(0.5, 0.7, 0.8));
const vec3 RIM_DIR = normalize(vec3(-0.7, 0.3, -0.5));
const vec3 RIM2_DIR = normalize(vec3(0.7, 0.2, -0.6));
const float HEMI_I = 0.9, KEY_I = 1.9, RIM_I = 2.2, RIM2_I = 1.4;
const float EMISSIVE_I = 0.55, ROUGH = 0.38, METAL = 0.15;

vec3 RRTAndODTFit(vec3 v) {
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}
vec3 aces(vec3 color) {
    const mat3 inM = mat3(0.59719, 0.07600, 0.02840,
                          0.35458, 0.90834, 0.13383,
                          0.04823, 0.01566, 0.83777);
    const mat3 outM = mat3(1.60475, -0.10208, -0.00327,
                           -0.53108, 1.10813, -0.07276,
                           -0.07367, -0.00605, 1.07602);
    color = inM * (color / 0.6);
    color = RRTAndODTFit(color);
    return clamp(outM * color, 0.0, 1.0);
}

vec3 light(vec3 N, vec3 V, vec3 albedo, vec3 dir, vec3 col, float inten) {
    float ndl = max(dot(N, dir), 0.0);
    vec3 h = normalize(dir + V);
    float spec = pow(max(dot(N, h), 0.0), 2.0 / (ROUGH * ROUGH * ROUGH * ROUGH));
    vec3 f0 = mix(vec3(0.04), albedo, METAL);
    return (albedo * (1.0 - METAL) + f0 * spec * 0.6) * col * inten * ndl;
}

void main() {
    vec3 N = normalize(v_nrm);
    vec3 V = normalize(cam_pos - v_pos);
    vec3 albedo = pow(v_col, vec3(2.2));
    vec3 c = albedo * (1.0 - METAL)
           * mix(hemi_ground, hemi_sky, N.y * 0.5 + 0.5) * HEMI_I;
    c += light(N, V, albedo, KEY_DIR, key_col, KEY_I);
    c += light(N, V, albedo, RIM_DIR, rim_col, RIM_I);
    c += light(N, V, albedo, RIM2_DIR, rim2_col, RIM2_I);
    c *= light_scale;
    c += emissive * EMISSIVE_I * emissive_scale;
    vec3 rgb = pow(aces(c * exposure), vec3(1.0 / 2.2));
    float a = 1.0;
    if (glass > 0.5) {
        // Almost invisible face-on, bright at grazing angles — which is all
        // you actually see of real glass.
        float f = pow(1.0 - abs(dot(N, V)), 2.6);
        a = clamp(0.055 + 0.85 * f, 0.0, 1.0);
        rgb += rim_col * f * 0.55;
    }
    f_col = vec4(rgb, a);
}
"""


# Indicator LEDs: unlit, each blinking on its own phase. A separate tiny
# program because they need no shading at all — only a colour and a clock.
LIGHT_VERT = """
#version 330
uniform mat4 mvp;
uniform float u_time;
in vec3 in_pos;
in vec3 in_col;
in float in_phase;
out vec3 v_col;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    float rate = 0.6 + mod(in_phase, 1.7);
    float on = step(0.42, fract(u_time * rate + in_phase));
    v_col = in_col * (0.16 + 0.84 * on);
}
"""

LIGHT_FRAG = """
#version 330
in vec3 v_col;
out vec4 f_col;
void main() { f_col = vec4(v_col, 1.0); }
"""


# Bubbles rising in the tube. Camera-facing quads whose height is computed
# from the clock, so nothing has to be re-uploaded per frame.
BUBBLE_VERT = """
#version 330
uniform mat4 mvp;
uniform vec3 cam_right, cam_up;
uniform float u_time, y_lo, y_hi;
in vec2 in_corner;
in vec4 in_seed;    // x, z, phase, speed
in float in_size;
out vec2 v_uv;
out float v_fade;
void main() {
    float travel = fract(u_time * in_seed.w + in_seed.z);
    float y = mix(y_lo, y_hi, travel);
    // Drift sideways as it climbs, and swell slightly near the top.
    float wob = sin(travel * 11.0 + in_seed.z * 6.283) * 0.006;
    vec3 c = vec3(in_seed.x + wob, y, in_seed.y);
    float s = in_size * (0.75 + 0.35 * travel);
    gl_Position = mvp * vec4(c + (in_corner.x * cam_right
                                + in_corner.y * cam_up) * s, 1.0);
    v_uv = in_corner;
    // Fade in at the bottom and out at the top so they never pop.
    v_fade = smoothstep(0.0, 0.12, travel) * (1.0 - smoothstep(0.82, 1.0, travel));
}
"""

BUBBLE_FRAG = """
#version 330
uniform vec3 tint;
in vec2 v_uv;
in float v_fade;
out vec4 f_col;
void main() {
    float r = length(v_uv);
    if (r > 1.0) discard;
    // Thin bright shell, hollow centre — a bubble is mostly its edge.
    float shell = smoothstep(0.62, 0.99, r) * (1.0 - smoothstep(0.99, 1.0, r));
    float glint = pow(max(0.0, 1.0 - length(v_uv - vec2(-0.34, 0.34))), 3.5);
    float a = (shell * 0.85 + glint * 0.5) * v_fade;
    f_col = vec4(tint * (0.55 + 0.45 * shell + glint), a);
}
"""


# Terminal screens: unlit, scrolling a text strip. Emissive by nature, so
# they read regardless of how dark the wall around them is.
SCREEN_VERT = """
#version 330
uniform mat4 mvp;
uniform float u_time;
in vec3 in_pos;
in vec2 in_uv;
in vec2 in_scroll;   // speed, phase
out vec2 v_uv;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    v_uv = vec2(in_uv.x, in_uv.y + u_time * in_scroll.x + in_scroll.y);
}
"""

SCREEN_FRAG = """
#version 330
uniform sampler2D screen_tex;
uniform vec3 tint;
in vec2 v_uv;
out vec4 f_col;
void main() {
    vec3 c = texture(screen_tex, vec2(v_uv.x, fract(v_uv.y))).rgb;
    // Scanlines across the panel itself, and a soft edge falloff so it looks
    // like a lit screen rather than a pasted rectangle.
    float scan = 0.86 + 0.14 * sin(v_uv.y * 900.0);
    float edge = smoothstep(0.0, 0.05, v_uv.x) * smoothstep(1.0, 0.95, v_uv.x);
    // Lifted well above 1.0: these are the only self-lit surfaces on the
    // wall, so they have to carry it on their own.
    f_col = vec4(c * tint * scan * (0.62 + 0.38 * edge) * 2.1, 1.0);
}
"""


# Cables get their own vertex stage so they can drift. Same fragment shader,
# so they light identically to the rest of the set.
CABLE_VERT = """
#version 330
uniform mat4 mvp;
uniform mat3 nrm_mat;
uniform float u_time, head_lift, sway, drain;
uniform float tube_r, tube_z;
in vec3 in_pos;
in vec3 in_nrm;
in vec3 in_col;
in float in_attach;   // 1 at the bust end, 0 bolted to the pedestal
in float in_phase;
in float in_sway;     // per-run: taut lines barely move, slack ones wander
in float in_tube;     // 1 = must stay inside the glass, 0 = wall run
out vec3 v_nrm;
out vec3 v_col;
out vec3 v_pos;
void main() {
    vec3 p = in_pos;
    p.y += head_lift * in_attach;

    // Both ends stay pinned, so the weight peaks mid-run and the fittings
    // never move. `drain` goes 0 -> 1 as the tube empties.
    float slack = in_attach * (1.0 - in_attach) * 4.0 * in_sway;

    // Motion comes from the fluid, so it fades out with the fluid, and the
    // slack settles downward once nothing is holding it up. Earlier versions
    // had the runs release and whip around like tentacles; that read as
    // chaos, and every attempt to tame it added another artefact.
    float wet = 1.0 - drain;
    float w = u_time * 0.55 + in_phase;
    p.x += sin(w) * sway * slack * wet;
    p.z += cos(w * 0.83 + 1.3) * sway * slack * wet;
    p.y += sin(w * 1.27) * sway * 0.45 * slack * wet;
    p.y -= drain * slack * 0.016;

    // Hard stop at the glass. The swirl amplitude depends on several tuned
    // numbers, and any of them can be raised later; this guarantees a run
    // that belongs inside the tube cannot leave it whatever they are set to.
    if (in_tube > 0.5) {
        vec2 rel = vec2(p.x, p.z - tube_z);
        float r = length(rel);
        if (r > tube_r) {
            rel *= tube_r / r;
            p.x = rel.x;
            p.z = tube_z + rel.y;
        }
    }
    v_nrm = nrm_mat * in_nrm;
    v_col = in_col;
    v_pos = p;
    gl_Position = mvp * vec4(p, 1.0);
}
"""


def build_bubbles(count=22, radius=0.150, seed=12, streams=2, per_stream=13):
    """Two kinds: steady streams and sparse strays.

    A stream is a fixed point releasing evenly spaced bubbles at one rate, so
    it reads as a valve. Strays are scattered and irregular. Evenly scattered
    bubbles alone read as a particle effect rather than as fluid.
    """
    rng = np.random.default_rng(seed)
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)
    corner, seeds, sizes, faces = [], [], [], []
    n = 0

    def add(x, z, phase, speed, size):
        nonlocal n
        corner.append(corners)
        seeds.append(np.tile(np.array([x, z, phase, speed], np.float32), (4, 1)))
        sizes.append(np.full(4, size, dtype=np.float32))
        b = n * 4
        faces.append(np.array([[b, b + 1, b + 2], [b, b + 2, b + 3]], dtype=np.uint32))
        n += 1

    for s in range(streams):
        ang = rng.uniform(0, 2 * np.pi)
        rad = radius * rng.uniform(0.45, 0.80)
        sx, sz = float(np.cos(ang) * rad), float(0.0271 + np.sin(ang) * rad)
        speed = float(rng.uniform(0.085, 0.125))
        base = float(rng.uniform(0.004, 0.008))
        for k in range(per_stream):
            # Evenly spaced phases: a continuous column from one nozzle.
            add(sx + float(rng.normal(0, 0.0016)),
                sz + float(rng.normal(0, 0.0016)),
                k / per_stream, speed, base * float(rng.uniform(0.85, 1.2)))

    for _ in range(count):
        ang = rng.uniform(0, 2 * np.pi)
        rad = radius * np.sqrt(rng.uniform(0.02, 0.88))
        add(float(np.cos(ang) * rad), float(0.0271 + np.sin(ang) * rad),
            float(rng.random()), float(rng.uniform(0.045, 0.14)),
            float(rng.uniform(0.005, 0.015)))

    return {"corner": np.ascontiguousarray(np.concatenate(corner)),
            "seed": np.ascontiguousarray(np.concatenate(seeds)),
            "size": np.ascontiguousarray(np.concatenate(sizes)),
            "faces": np.ascontiguousarray(np.concatenate(faces))}


def load_head(name):
    """Reads the packed head written by export_head.py (ours, green) or by
    jarvis-agent/tools/export_head.py (theirs, blue)."""
    if name == "head_blue":
        base, stem = JARVIS_ASSETS, "head"
    else:
        base, stem = FONTS, name
    manifest = json.loads((base / f"{stem}.json").read_text())
    buf = (base / f"{stem}.bin").read_bytes()

    def block(bname):
        b = manifest["blocks"][bname]
        n = int(np.prod(b["shape"]))
        arr = np.frombuffer(buf, dtype=np.dtype(b["dtype"]), count=n, offset=b["offset"])
        return arr.reshape(b["shape"]).copy()

    return {
        "positions": block("positions").astype(np.float32),
        "colors": block("colors").astype(np.float32) / 255.0,
        "faces": block("triangles").astype(np.int32),
        "morphs": {n: block(f"morph_{n}").astype(np.float32)
                   for n in manifest["morphNames"]},
        "names": manifest["morphNames"],
    }


def vertex_normals(pos, faces, flat_idx, nverts):
    tri = pos[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn3 = np.repeat(fn, 3, axis=0)
    vn = np.stack([np.bincount(flat_idx, weights=fn3[:, c], minlength=nverts)
                   for c in range(3)], axis=1).astype(np.float32)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)


class Animator:
    """Replays web/app.js tick(): viseme targets with attack/release
    smoothing, the blink schedule, and idle/speaking head sway."""

    def __init__(self, names, segments, seed=42):
        self.index = {n: i for i, n in enumerate(names)}
        self.segments = segments
        self.n = len(names)
        self.influences = np.zeros(self.n, dtype=np.float32)
        self.rng = random.Random(seed)
        self.next_blink = 1.2
        self.blink_phase = -1.0
        self.sway = 1.0

    def _active(self, t):
        t = t + LOOKAHEAD
        for seg in self.segments:
            if seg["t0"] - 0.012 <= t < seg["t1"]:
                return seg
        return None

    def _speaking(self, t):
        return any(s["t0"] - 0.25 <= t <= s["t1"] + 0.25 for s in self.segments)

    def step(self, t, dt):
        targets = np.zeros(self.n, dtype=np.float32)
        seg = self._active(t)
        if seg and seg["v"] != "sil":
            targets[self.index[seg["v"]]] = seg["w"]

        if self.blink_phase >= 0:
            self.blink_phase += dt
            p = self.blink_phase / BLINK_LEN
            if p >= 1:
                self.blink_phase = -1.0
                self.next_blink = t + 1.8 + self.rng.random() * 4.2
            else:
                targets[self.index["blink"]] = math.sin(math.pi * min(p, 1.0))
        elif t >= self.next_blink:
            self.blink_phase = 0.0

        for i in range(self.n):
            tau = ATTACK if targets[i] > self.influences[i] else RELEASE
            self.influences[i] += (targets[i] - self.influences[i]) * \
                (1 - math.exp(-dt / tau))

        sway_target = 1.6 if self._speaking(t) else 1.0
        self.sway += (sway_target - self.sway) * (1 - math.exp(-dt / 0.4))
        return self.influences, (math.sin(t * 0.23 + 1.7) * 0.018 * self.sway,
                                 math.sin(t * 0.31) * 0.035 * self.sway,
                                 math.sin(t * 0.17 + 0.6) * 0.008)


def rotation(rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (my @ mx @ mz).astype(np.float32)


def perspective(fov_y_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_y_deg) / 2)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye, target):
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = right, up, -fwd
    m[:3, 3] = -m[:3, :3] @ eye
    return m


# ---------- overlay ----------


# ---------- opening sequence ----------
# SCENE time: seconds after the title card cuts away. configure_sequence()
# rescales these to the clip so a short story does not run out of room.
SPIN_END = 3.5
ARM_IN = 4.4
ARM_GRIP = 6.6
UNSCREW_END = 8.8
LIFT_END = 11.4
DRAIN_START = 9.2
DRAIN_END = 15.5
DETACH_START = 11.5
EXIT_START = 15.5     # tube begins sliding out as the last push starts
EXIT_END = 19.5
HELMET_IN = 16.0      # lowers once the tube is out of the way
HELMET_SEAT = 20.0    # settles onto the head

AXIS_Z = 0.0271       # the tube's axis; nothing here is centred on the origin


def configure_sequence(scene_dur, spin_end):
    """Fit the beats to the clip. Absolute seconds would overrun a short one
    and leave a long one sitting on an empty tube."""
    global SPIN_END, ARM_IN, ARM_GRIP, UNSCREW_END, LIFT_END
    global DRAIN_START, DRAIN_END, EXIT_START, EXIT_END, DETACH_START
    global HELMET_IN, HELMET_SEAT
    SPIN_END = spin_end                       # locked to the camera push
    ARM_IN = spin_end - 0.6
    ARM_GRIP = ARM_IN + 1.9
    UNSCREW_END = ARM_GRIP + 1.9
    LIFT_END = UNSCREW_END + 2.2
    DRAIN_START = UNSCREW_END + 0.4
    DRAIN_END = min(DRAIN_START + 6.3, max(DRAIN_START + 2.0, scene_dur - 5.0))
    DETACH_START = DRAIN_START + 0.32 * (DRAIN_END - DRAIN_START)
    EXIT_START = DRAIN_END
    EXIT_END = min(EXIT_START + 4.0, scene_dur - 0.5)
    # Helmet lands with time to spare before the last word.
    HELMET_SEAT = max(EXIT_END + 0.6, scene_dur - 2.6)
    HELMET_IN = HELMET_SEAT - 3.4


def ease_in_out(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    return m


def translate(x, y, z):
    m = np.eye(4, dtype=np.float32)
    m[0, 3], m[1, 3], m[2, 3] = x, y, z
    return m


def spin(a):
    """Turn about the TUBE's axis, not the world origin.

    Everything here is built around z = AXIS_Z, so a bare rot_y() swings the
    piece around a pole 27 mm in front of it — it orbits instead of spinning,
    which is exactly the wobble the cap had.
    """
    return translate(0, 0, AXIS_Z) @ rot_y(a) @ translate(0, 0, -AXIS_Z)


def intro_transform(ts, y_off):
    """Rise while turning, settling as the camera push completes.

    y_off is supplied by the caller and measured against the CURRENT frame,
    not as a fixed world distance. That matters: the push shrinks the visible
    height nearly fourfold during this move, so a constant drop leaves the
    tube climbing into a frame that is receding faster than it rises — it
    vanished off the bottom for most of the move and only reappeared once it
    was nearly home, which read as the spin starting late.
    """
    if ts >= SPIN_END:
        return np.eye(4, dtype=np.float32)
    p = ease_out(ts / SPIN_END)
    return translate(0.0, y_off, 0.0) @ spin(2 * math.pi * (1 - p))


def exit_transform(ts, drop=1.5):
    """The tube sinks out of frame at the end, leaving the head floating."""
    if ts <= EXIT_START:
        return np.eye(4, dtype=np.float32)
    p = ease_in_out((ts - EXIT_START) / max(EXIT_END - EXIT_START, 1e-6))
    return translate(0.0, -drop * p, 0.0)


def rig_transform(ts, y_off):
    return exit_transform(ts) @ intro_transform(ts, y_off)


def cap_transform(ts, y_off):
    """Rides the intro, is turned loose, then lifts away."""
    if ts < ARM_GRIP:
        return intro_transform(ts, y_off), 0.0
    turn = ease_in_out((ts - ARM_GRIP) / (UNSCREW_END - ARM_GRIP)) * 4 * math.pi
    lift = 0.0
    if ts > UNSCREW_END:
        lift = ease_in_out((ts - UNSCREW_END) / (LIFT_END - UNSCREW_END)) * 1.45
    return translate(0.0, lift, 0.0) @ spin(turn), lift


def arm_transform(ts):
    """Descends turning, grips, unscrews, then withdraws with the cap."""
    if ts < ARM_IN:
        return None
    reach = 1.50
    if ts < ARM_GRIP:
        p = ease_in_out((ts - ARM_IN) / (ARM_GRIP - ARM_IN))
        y = set_pieces.CAP_TOP + reach * (1 - p)
        turn = (1 - p) * 6 * math.pi
    else:
        _cap, lift = cap_transform(ts, 0.0)
        y = set_pieces.CAP_TOP + lift
        turn = ease_in_out((ts - ARM_GRIP)
                           / (UNSCREW_END - ARM_GRIP)) * 4 * math.pi
    if y > set_pieces.CAP_TOP + reach + 0.25:
        return None
    return translate(0.0, y, 0.0) @ spin(turn)


def helmet_transform(ts, drop=1.30):
    """Lowers onto the head in a single continuous move."""
    if ts < HELMET_IN:
        return None
    p = (ts - HELMET_IN) / max(HELMET_SEAT - HELMET_IN, 1e-6)
    if p >= 1.0:
        return np.eye(4, dtype=np.float32)

    # One continuous descent, no rotation and no mid-air stop. An earlier
    # version fell, hovered, then made a second short drop; that reads as two
    # moves stitched together rather than as a single placement.
    return translate(0.0, drop * (1.0 - ease_out(p)), 0.0)


def water_level(ts):
    top = set_pieces.TUBE_TOP - 0.02
    bottom = set_pieces.TUBE_BOTTOM + 0.01
    if ts <= DRAIN_START:
        return top
    p = ease_in_out((ts - DRAIN_START) / max(DRAIN_END - DRAIN_START, 1e-6))
    return top + (bottom - top) * p


def buoyancy(ts):
    """Bob amplitude. Drains to nothing: with no fluid there is nothing to
    float in, so the bust settles instead of hovering in air."""
    if ts <= DRAIN_START:
        return 1.0
    return 1.0 - ease_in_out((ts - DRAIN_START) / max(DRAIN_END - DRAIN_START, 1e-6))


def drain_amount(ts):
    """0 while the tube is full, 1 once it is empty.

    Drives the cables: their motion comes from the fluid, so it fades as the
    fluid does and the slack settles. They stay bolted at both ends.
    """
    if ts <= DRAIN_START:
        return 0.0
    return ease_in_out((ts - DRAIN_START) / max(DRAIN_END - DRAIN_START, 1e-6))


def build_title_card(theme, date_line, hook=""):
    """Opening card: mark and masthead alone, no set behind them.

    Cutting from this to the tube gives the opening an edit, which a
    continuous push does not have — and an edit is a harder attention cue
    than any camera move.
    """
    img = Image.new("RGB", (W, H), theme["bg"])
    d = ImageDraw.Draw(img)

    center(d, 700, "The Daily Prompt", font("jacquard", 156), theme["ink"][:3])
    center(d, 900, date_line.upper(), font("vt323", 44), theme["dim"][:3])
    d.rectangle([230, 872, W - 230, 875], fill=theme["rule"][:3])

    if hook:
        fnt = font("press", 56)
        lines = wrap(d, hook.upper(), fnt, W - 160)[:3]
        y = 1010
        for line in lines:
            center(d, y, line, fnt, theme["glow"][:3])
            y += 86
    return np.asarray(img)


def build_chrome(theme, date_line, caption, source=""):
    """Everything that does not pulse: masthead, rules, source, caption, footer."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    masthead(d, theme, date_line)
    if source:
        center(d, SOURCE_Y, f"SOURCE: {source.upper()}", font("vt323", 38),
               theme["dim"])

    if caption:
        cap_font = font("vt323", 52)
        lines = wrap(d, caption, cap_font, W - 220)[:3]
        box_h = len(lines) * CAPTION_LEAD + 34
        d.rectangle([70, CAPTION_Y, W - 70, CAPTION_Y + box_h],
                    fill=theme["bg"] + (196,))
        cy = CAPTION_Y + 16
        for line in lines:
            center(d, cy, line, cap_font, theme["glow"])
            cy += CAPTION_LEAD

    center(d, FOOTER_Y, "DAILY AI NEWS · FULL STORY IN BIO",
           font("vt323", 36), theme["foot"])
    return np.asarray(img).astype(np.int32)


HOOK_CENTER_Y = 980
HOOK_MAX_LINES = 3
HOOK_MAX_PT = 66
HOOK_MIN_PT = 38


def build_hook(theme, hook):
    """The opening card: big text, readable on frame zero.

    Narration only delivers about four words in the first two seconds, which
    is all the time most viewers give. Reading is faster than listening, so
    the hook goes on screen immediately and the voice catches up.
    """
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for pt in range(HOOK_MAX_PT, HOOK_MIN_PT - 1, -2):
        fnt = font("press", pt)
        lines = wrap(d, hook.upper(), fnt, W - 130)
        if len(lines) <= HOOK_MAX_LINES:
            break
    lead = int(fnt.size * 1.5)
    block_h = len(lines) * lead
    y = HOOK_CENTER_Y - block_h // 2

    # Scrim: the hook sits over the face, and unreadable text is worse than
    # no text.
    d.rectangle([0, y - 46, W, y + block_h + 40], fill=theme["bg"] + (212,))
    d.rectangle([0, y - 50, W, y - 46], fill=theme["glow"])
    d.rectangle([0, y + block_h + 40, W, y + block_h + 44], fill=theme["glow"])
    for line in lines:
        center(d, y, line, fnt, theme["ink"])
        y += lead
    return np.asarray(img).astype(np.int32)


def build_headline(theme, headline):
    """The headline on its own layer, so its brightness can breathe.

    It gets a scrim because the camera push sweeps the tube's top ring up
    through this band: without one the headline is unreadable for most of the
    clip, and which frames it lands on depends on the zoom.
    """
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    fnt = font("press", 44)
    lines = wrap(d, headline.upper(), fnt, W - 150)[:3]
    block_h = len(lines) * HEADLINE_LEAD
    d.rectangle([0, HEADLINE_Y - 30, W, HEADLINE_Y + block_h + 8],
                fill=theme["bg"] + (188,))
    y = HEADLINE_Y
    for line in lines:
        center(d, y, line, fnt, theme["ink"])
        y += HEADLINE_LEAD
    return np.asarray(img).astype(np.int32)


# ---------- CRT treatment ----------

# amix halves the level, and social platforms normalize to about -14 LUFS,
# so land there rather than letting them turn a quiet mix up for us.
LOUDNESS = "loudnorm=I=-14:TP=-1.5:LRA=11,alimiter=limit=0.97"

CRT_LINE_T = 0.30    # bright horizontal line holds until here
CRT_OPEN_T = 0.85    # picture finished unfolding by here
CRT_SETTLE_T = 1.25  # brightness overshoot decayed by here


def ease_out(x):
    return 1 - (1 - x) ** 3


CRT_FLASH_T = 0.34   # "flash" mode: settled by here, never fully black


def crt_flash(frame, t, tint):
    """Power-on that starts with the picture already up.

    The full sequence spends its first third of a second on a black frame,
    which is the single most valuable moment in a short — the viewer decides
    there. This keeps the tube-snapping-on character (a slight vertical
    squash, overbright, mains flicker) while frame zero is already readable.
    """
    if t >= CRT_FLASH_T:
        return None
    p = ease_out(min(t / CRT_FLASH_T, 1.0))
    band = int(H * (0.88 + 0.12 * p))
    if band < H:
        squashed = np.asarray(Image.fromarray(frame).resize((W, band), Image.BILINEAR))
        out = np.zeros((H, W, 3), dtype=np.float32)
        y0 = (H - band) // 2
        out[y0:y0 + band] = squashed
        edge = np.array(tint) * 255.0 * (1.0 - p) ** 0.7
        out[y0:y0 + 2] = edge
        out[y0 + band - 2:y0 + band] = edge
    else:
        out = frame.astype(np.float32)
    out *= (1.0 + 0.45 * (1 - p) ** 2) * (1.0 + 0.05 * math.sin(t * 61.0) * (1 - p))
    return np.clip(out, 0, 255).astype(np.uint8)


def crt_power_on(frame, t, tint):
    """Old-TV turn-on: a bright line snaps across, then the picture unfolds
    vertically out of it and the overbright settles. Returns None past the end
    so the caller can skip the work."""
    if t >= CRT_SETTLE_T:
        return None
    out = np.zeros((H, W, 3), dtype=np.float32)

    if t < CRT_LINE_T:
        # Line grows out from the centre, blindingly bright.
        p = ease_out(min(t / CRT_LINE_T, 1.0))
        half = max(1, int(p * W / 2))
        thick = max(2, int(6 - 3 * p))
        y0 = H // 2 - thick // 2
        out[y0:y0 + thick, W // 2 - half:W // 2 + half] = np.array(tint) * 255.0
        glow = max(2, int(thick * 5))
        g0, g1 = max(0, y0 - glow), min(H, y0 + thick + glow)
        out[g0:g1, W // 2 - half:W // 2 + half] += np.array(tint) * 46.0
    else:
        # Picture unfolds vertically out of the line.
        p = ease_out(min((t - CRT_LINE_T) / (CRT_OPEN_T - CRT_LINE_T), 1.0))
        band = max(2, int(p * H))
        squashed = np.asarray(Image.fromarray(frame).resize((W, band), Image.BILINEAR))
        y0 = (H - band) // 2
        out[y0:y0 + band] = squashed
        if p < 1.0:
            # Residual scan line riding the top and bottom edges.
            edge = np.array(tint) * 255.0 * (1.0 - p) ** 0.6
            out[y0:y0 + 2] = edge
            out[y0 + band - 2:y0 + band] = edge
        # Overbright decays after the picture is open.
        if t > CRT_OPEN_T:
            k = (t - CRT_OPEN_T) / (CRT_SETTLE_T - CRT_OPEN_T)
            out *= 1.0 + 0.55 * (1 - k) ** 2
        else:
            out *= 1.55

    # Mains-hum flicker while the tube stabilizes.
    out *= 1.0 + 0.05 * math.sin(t * 61.0) * max(0.0, 1 - t / CRT_SETTLE_T)
    return np.clip(out, 0, 255).astype(np.uint8)



class Glitch:
    """Short bursts of signal corruption, a few frames each, a few seconds
    apart. Only the picture breaks up — the audio runs straight through, which
    is what sells it as a transmission fault rather than a broken render.

    Every frame inside a burst re-rolls its own parameters. Holding one
    distortion still for several frames reads as a rendering mistake; changing
    it every frame reads as interference.
    """

    def __init__(self, duration, fps, seed=11, start_after=1.6,
                 gap=(2.8, 7.0), frames=(2, 5)):
        self.rng = random.Random(seed)
        self.frames = frames
        self.events = []
        sched = random.Random(seed * 7 + 1)
        t = start_after + sched.uniform(0.5, 2.2)
        while t < duration - 0.5:
            f0 = int(t * fps)
            self.events.append((f0, f0 + sched.randint(*frames),
                                sched.uniform(0.35, 1.0)))
            t += sched.uniform(*gap)

    def strength(self, frame_index):
        for f0, f1, s in self.events:
            if f0 <= frame_index < f1:
                return s
        return 0.0

    def apply(self, frame, s):
        rng = self.rng
        out = frame

        # Horizontal slice displacement — the tape-tear look.
        for _ in range(rng.randint(2, 6)):
            h = rng.randint(6, int(70 * s) + 10)
            y = rng.randint(0, H - h)
            dx = int(rng.gauss(0, 26 * s))
            if dx:
                out[y:y + h] = np.roll(out[y:y + h], dx, axis=1)

        # Chroma split: the colour channels stop agreeing on where things are.
        d = int(rng.uniform(3, 15) * s)
        if d:
            out[:, :, 0] = np.roll(out[:, :, 0], d, axis=1)
            out[:, :, 2] = np.roll(out[:, :, 2], -d, axis=1)

        # Vertical hold slipping.
        if rng.random() < 0.3:
            out = np.roll(out, rng.randint(-20, 20), axis=0)

        # Brightness pop, as if the beam current jumped.
        if rng.random() < 0.4:
            out = np.clip(out.astype(np.float32) * rng.uniform(1.06, 1.28),
                          0, 255).astype(np.uint8)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True,
                    help="dir with audio.wav + timeline.json from voice_track.py")
    ap.add_argument("--audio", default="",
                    help="narration wav; defaults to <work-dir>/audio.wav")
    ap.add_argument("--headline", required=True)
    ap.add_argument("--date-line", default="")
    ap.add_argument("--source", default="", help="attribution line under the headline")
    ap.add_argument("--out", required=True)
    ap.add_argument("--theme", default="phosphor", choices=sorted(THEMES))
    ap.add_argument("--speed", type=float, default=1.15,
                    help="playback rate; scales the timeline and atempos the voice")
    ap.add_argument("--exposure", type=float, default=1.0)
    ap.add_argument("--head-height", type=float, default=0.215,
                    help="head height once the opening push lands")
    ap.add_argument("--head-y", type=float, default=0.50,
                    help="head center as fraction of frame height, from the top")
    ap.add_argument("--music", default="", help="wav/mp3 to sit under the narration")
    ap.add_argument("--music-gain", type=float, default=-19.0, help="dB")
    ap.add_argument("--music-fade-in", type=float, default=1.5,
                    help="0 makes the bed's first hit the opening accent")
    ap.add_argument("--sfx", default=str(FONTS / "crt-power-on.wav"),
                    help='power-on sound at t=0; "" for none')
    ap.add_argument("--sfx-gain", type=float, default=-7.0, help="dB")
    ap.add_argument("--scene-sfx", action="store_true", default=True,
                    help="synthesized event sounds synced to the sequence")
    ap.add_argument("--no-scene-sfx", dest="scene_sfx", action="store_false")
    ap.add_argument("--scene-sfx-gain", type=float, default=-6.0, help="dB")
    ap.add_argument("--pulse", type=float, default=3.6,
                    help="headline breathing period in seconds; 0 disables")
    ap.add_argument("--pulse-depth", type=float, default=0.34)
    ap.add_argument("--glitch", type=float, default=1.0,
                    help="glitch frequency multiplier; 0 disables")
    ap.add_argument("--grain", type=float, default=6.0, help="grain sigma; 0 disables")
    ap.add_argument("--scanline", type=float, default=0.10)
    ap.add_argument("--crt-intro", default="none", choices=("flash", "full", "none"),
                    help="'full' opens from black — good on YouTube, costly on a feed")
    ap.add_argument("--hook", default="", help="big opening text, on screen from t=0")
    ap.add_argument("--hook-hold", type=float, default=1.9,
                    help="seconds the big hook stays before it hands off")
    ap.add_argument("--cuts", action="store_true", default=True,
                    help="cut between framings on sentence boundaries")
    ap.add_argument("--no-cuts", dest="cuts", action="store_false")
    ap.add_argument("--cut-every", type=int, default=2, help="cut every N sentences")
    ap.add_argument("--set-dressing", action="store_true", default=True,
                    help="containment tube, rings and pedestal around the head")
    ap.add_argument("--no-set-dressing", dest="set_dressing", action="store_false")
    ap.add_argument("--title-card", type=float, default=0.0,
                    help="seconds of masthead-only card before the scene; 0 skips")
    ap.add_argument("--bare", action="store_true",
                    help="render the set alone, with no masthead or type")
    ap.add_argument("--max-seconds", type=float, default=0,
                    help="stop after this many seconds; 0 renders the whole clip")
    ap.add_argument("--float-period", type=float, default=7.5,
                    help="seconds per up-down cycle of the floating bust")
    ap.add_argument("--float-amount", type=float, default=0.014,
                    help="vertical drift in world units")
    ap.add_argument("--cable-sway", type=float, default=0.011,
                    help="how far the slack middle of a cable drifts")
    ap.add_argument("--helmet", action="store_true", default=True,
                    help="visor helmet lowers onto the head at the end")
    ap.add_argument("--no-helmet", dest="helmet", action="store_false")
    ap.add_argument("--helmet-glb", default="",
                    help="override the helmet mesh path")
    ap.add_argument("--helmet-rot-x", type=float, default=0.0)
    ap.add_argument("--helmet-rot-y", type=float, default=180.0,
                    help="TRELLIS returns an arbitrary orientation; 180 puts\n                          the visor band level across the eyes")
    ap.add_argument("--helmet-rot-z", type=float, default=0.0)
    ap.add_argument("--bubbles", type=int, default=22,
                    help="bubbles rising inside the tube; 0 disables")
    ap.add_argument("--set-glb", default="",
                    help="use a set built in Blender instead of the procedural one")
    ap.add_argument("--feed", action="append", default=[],
                    help="'N:img.png' overlays wall screen N with a static "
                         "surveillance-feed image; 'X,Y,W,H:img.png' mounts a "
                         "new feed panel on the clear wall at that spot "
                         "(the strip |x| 0.20-0.31 dodges the rack columns)")
    ap.add_argument("--set-gain", type=float, default=0.42,
                    help="how much light the set catches, relative to the head")
    ap.add_argument("--set-emissive", type=float, default=0.10,
                    help="set's share of the hologram glow floor")
    ap.add_argument("--zoom", action="store_true", default=True,
                    help="slow push from the full apparatus in to the head")
    ap.add_argument("--no-zoom", dest="zoom", action="store_false")
    ap.add_argument("--zoom-from", type=float, default=0.055,
                    help="starting head height as a fraction of the frame")
    ap.add_argument("--final-height", type=float, default=0.34,
                    help="head height after the drain; 0 keeps the medium shot")
    ap.add_argument("--final-zoom-time", type=float, default=3.5)
    ap.add_argument("--zoom-time", type=float, default=3.5,
                    help="seconds to complete the push")
    ap.add_argument("--still", type=float, default=None,
                    help="render a single PNG at this time instead of the video")
    args = ap.parse_args()

    theme = THEMES[args.theme]
    work = Path(args.work_dir)
    tl = json.loads((work / "timeline.json").read_text())
    speed = max(0.5, args.speed)

    segments = [{"v": s["v"], "t0": s["t0"] / speed, "t1": s["t1"] / speed,
                 "w": s["w"]} for s in tl["segments"]]
    sentences = [{"text": s["text"], "t0": s["t0"] / speed, "t1": s["t1"] / speed}
                 for s in tl.get("sentences", [])]
    duration = tl["duration"] / speed

    head = load_head(theme["head"])
    nverts = head["positions"].shape[0]
    faces = head["faces"]
    flat_idx = faces.reshape(-1)
    base = head["positions"]

    # The bust hangs clear of the pedestal so the cables have somewhere to
    # hang. Must match HEAD_LIFT in cables.py. Defined before camera() because
    # the framing has to follow the bust to where it actually sits.
    head_lift = 0.055

    # Frame on the head only: the bust's shoulders are wider and lower than
    # the face, so fitting the full bounding box shrinks and drops the head.
    neck_y = base[:, 1].min() + 0.55 * (base[:, 1].max() - base[:, 1].min())
    head_pts = base[base[:, 1] > neck_y]
    head_lo, head_hi = head_pts.min(axis=0), head_pts.max(axis=0)
    head_center = (head_lo + head_hi) / 2
    head_h = head_hi[1] - head_lo[1]

    aspect = W / H
    fov = 23.0
    half_fov = math.tan(math.radians(fov / 2))
    proj = perspective(fov, aspect, 0.05, 10.0)
    # sway pivots at the neck, as the three.js pivot group does
    pivot = np.array([0.0, neck_y, head_center[2]], dtype=np.float64)

    def camera(head_height, head_y):
        dist = head_h / (2 * head_height * half_fov)
        # Frame the bust where it actually sits, not where the mesh was
        # authored — it is lifted clear of the pedestal at render time.
        target_y = head_center[1] + head_lift - (0.5 - head_y) * 2 * dist * half_fov
        eye = np.array([0.0, target_y, head_center[2] + dist], dtype=np.float64)
        target = np.array([0.0, target_y, head_center[2]], dtype=np.float64)
        return look_at(eye, target), eye

    # A slow continuous push is a better motion than hard cuts here: it never
    # stops moving, so there is always something changing, and it reveals the
    # set first and the face second.
    if args.zoom:
        # Precompute one camera per frame; the matrices are tiny.
        n_zoom = max(2, int(math.ceil(duration * FPS)) + 1)
        zoom_frames, frame_bottoms = [], []
        for i in range(n_zoom):
            # Zoom clock starts at the cut, not at t=0, so the push begins
            # when the scene first appears.
            tt = max(0.0, i / FPS - (0.0 if args.bare else args.title_card))
            p = ease_out(min(1.0, tt / max(args.zoom_time, 1e-6)))
            hh = args.zoom_from + (args.head_height - args.zoom_from) * p
            # Second push, after the tube has drained: the medium framing has
            # to keep the cap on screen for the arm, which leaves the face too
            # small to hold for 20 s. So close in once the spectacle is over.
            if args.final_height > 0 and tt > DRAIN_END:
                q = ease_in_out((tt - DRAIN_END) / max(args.final_zoom_time, 1e-6))
                hh = hh + (args.final_height - hh) * q
            zoom_frames.append(camera(hh, args.head_y))
            fh = head_h / hh
            frame_bottoms.append(head_center[1] + head_lift
                                 - (0.5 - args.head_y) * fh - fh / 2)

        def shot_for(t):
            return zoom_frames[min(int(t * FPS), n_zoom - 1)]

        def rise_at(ts):
            """Tube top starts exactly on the frame bottom and lerps home, so
            it is on screen for every frame of the rise."""
            if ts >= SPIN_END:
                return 0.0
            i = min(max(int((ts + (0.0 if args.bare else args.title_card)) * FPS),
                        0), n_zoom - 1)
            return (1.0 - ease_out(ts / SPIN_END)) * (
                frame_bottoms[i] - set_pieces.TUBE_TOP)
    else:
        shot_specs = [(args.head_height, args.head_y)]
        if args.cuts:
            shot_specs += [(args.head_height * 1.5, args.head_y - 0.06),
                           (args.head_height * 0.82, args.head_y + 0.04)]
        shots = [camera(hh, hy) for hh, hy in shot_specs]
        # Cut on sentence boundaries — mid-sentence cuts read as a mistake.
        cut_at = []
        if args.cuts and sentences:
            for n, s in enumerate(sentences):
                cut_at.append((s["t0"] - 0.12, (n // args.cut_every) % len(shots)))

        def shot_for(t):
            idx = 0
            for start, shot in cut_at:
                if t >= start:
                    idx = shot
            return shots[idx]

    ctx = moderngl.create_context(standalone=True, backend="egl")
    ctx.enable(moderngl.DEPTH_TEST)
    prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
    for theme_key, uniform in (("hemi_sky", "hemi_sky"), ("hemi_ground", "hemi_ground"),
                               ("key", "key_col"), ("rim", "rim_col"),
                               ("rim2", "rim2_col"), ("emissive", "emissive")):
        prog[uniform].value = tuple(theme[theme_key])
    prog["exposure"].value = args.exposure  # cam_pos is set per frame, per shot

    vbo_pos = ctx.buffer(reserve=nverts * 12, dynamic=True)
    vbo_nrm = ctx.buffer(reserve=nverts * 12, dynamic=True)
    vbo_col = ctx.buffer(np.ascontiguousarray(head["colors"], dtype=np.float32))
    ibo = ctx.buffer(np.ascontiguousarray(faces, dtype=np.uint32))
    vao = ctx.vertex_array(prog, [(vbo_pos, "3f", "in_pos"),
                                  (vbo_nrm, "3f", "in_nrm"),
                                  (vbo_col, "3f", "in_col")], ibo)

    def static_vao(mesh):
        return ctx.vertex_array(prog, [
            (ctx.buffer(mesh["positions"]), "3f", "in_pos"),
            (ctx.buffer(mesh["normals"]), "3f", "in_nrm"),
            (ctx.buffer(mesh["colors"]), "3f", "in_col"),
        ], ctx.buffer(mesh["faces"]))

    set_opaque = set_glass = set_lights = set_bubbles = set_cables = None
    light_prog = bubble_prog = cable_prog = None
    feed_draws, feed_positions = [], []
    if args.set_dressing:
        groups = set_pieces.build_groups()
        set_wall = static_vao(groups["wall"])
        set_rig = static_vao(groups["rig"])
        set_cap = static_vao(groups["cap"])
        set_arm = static_vao(groups["arm"])
        set_water = static_vao(groups["water"])
        set_opaque, set_glass = set_rig, static_vao(groups["glass"])
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        if args.bubbles > 0:
            bub = build_bubbles(count=args.bubbles)
            bubble_prog = ctx.program(vertex_shader=BUBBLE_VERT,
                                      fragment_shader=BUBBLE_FRAG)
            bubble_prog["tint"].value = tuple(
                c / 255.0 for c in theme["glow"][:3])
            bubble_prog["y_lo"].value = set_pieces.TUBE_BOTTOM + 0.02
            bubble_prog["y_hi"].value = set_pieces.TUBE_TOP - 0.02
            set_bubbles = ctx.vertex_array(bubble_prog, [
                (ctx.buffer(bub["corner"]), "2f", "in_corner"),
                (ctx.buffer(bub["seed"]), "4f", "in_seed"),
                (ctx.buffer(bub["size"]), "1f", "in_size"),
            ], ctx.buffer(bub["faces"]))

        import cables as cables_mod
        cab = cables_mod.pack_animated(cables_mod.build())
        cable_prog = ctx.program(vertex_shader=CABLE_VERT, fragment_shader=FRAG)
        for theme_key, uniform in (("hemi_sky", "hemi_sky"),
                                   ("hemi_ground", "hemi_ground"),
                                   ("key", "key_col"), ("rim", "rim_col"),
                                   ("rim2", "rim2_col"), ("emissive", "emissive")):
            cable_prog[uniform].value = tuple(theme[theme_key])
        cable_prog["exposure"].value = args.exposure
        cable_prog["glass"].value = 0.0
        cable_prog["light_scale"].value = 1.0
        cable_prog["emissive_scale"].value = 1.0
        cable_prog["sway"].value = args.cable_sway
        cable_prog["tube_r"].value = set_pieces.TUBE_RADIUS - 0.012
        cable_prog["tube_z"].value = 0.0271
        cable_prog["nrm_mat"].write(np.ascontiguousarray(np.eye(3, dtype=np.float32)))
        set_cables = ctx.vertex_array(cable_prog, [
            (ctx.buffer(cab["positions"]), "3f", "in_pos"),
            (ctx.buffer(cab["normals"]), "3f", "in_nrm"),
            (ctx.buffer(cab["colors"]), "3f", "in_col"),
            (ctx.buffer(cab["attach"]), "1f", "in_attach"),
            (ctx.buffer(cab["phase"]), "1f", "in_phase"),
            (ctx.buffer(cab["sway_mul"]), "1f", "in_sway"),
            (ctx.buffer(cab["in_tube"]), "1f", "in_tube"),
        ], ctx.buffer(cab["faces"]))

        set_helmet = set_helmet_wires = None
        if args.helmet:
            import helmet as helmet_mod
            hm = helmet_mod.load(path=args.helmet_glb or helmet_mod.GLB,
                                 rot_x=args.helmet_rot_x,
                                 rot_y=args.helmet_rot_y,
                                 rot_z=args.helmet_rot_z)
            set_helmet = static_vao(helmet_mod.pack(hm))
            wire_mesh = helmet_mod.wires(hm)
            set_helmet_wires = static_vao(helmet_mod.pack(
                wire_mesh, color=set_pieces.COLORS["conduit"]))
            print(f"[render] helmet: {len(hm.faces)} faces")

        scr = set_pieces.build_screens()
        tex_img = set_pieces.screen_texture()
        screen_tex = ctx.texture(tex_img.size, 3, tex_img.tobytes())
        screen_tex.repeat_x, screen_tex.repeat_y = False, True
        screen_tex.build_mipmaps()
        screen_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        screen_prog = ctx.program(vertex_shader=SCREEN_VERT,
                                  fragment_shader=SCREEN_FRAG)
        screen_prog["tint"].value = tuple(c / 255.0 for c in theme["glow"][:3])
        set_screens = ctx.vertex_array(screen_prog, [
            (ctx.buffer(scr["positions"]), "3f", "in_pos"),
            (ctx.buffer(scr["uvs"]), "2f", "in_uv"),
            (ctx.buffer(scr["scroll"]), "2f", "in_scroll"),
        ], ctx.buffer(scr["faces"]))

        for spec in args.feed:
            loc, fpath = spec.split(":", 1)
            if "," in loc:
                vals = [float(v) for v in loc.split(",")]
                fx, fy, fw, fh = vals[:4]
                hw, hh = fw / 2, fh / 2
                fz = vals[4] if len(vals) > 4 else set_pieces.WALL_Z + 0.10
                fpos = np.array([[fx - hw, fy - hh, fz], [fx + hw, fy - hh, fz],
                                 [fx + hw, fy + hh, fz], [fx - hw, fy + hh, fz]],
                                dtype=np.float32)
            else:
                i = int(loc)
                fpos = scr["positions"][i * 4:(i + 1) * 4].copy()
                fpos[:, 2] += 0.002   # sits on the text panel, no z-fight
            fimg = Image.open(fpath).convert("RGB")
            ftex = ctx.texture(fimg.size, 3, fimg.tobytes())
            ftex.build_mipmaps()
            ftex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            feed_positions.append(fpos.copy())
            feed_draws.append((ftex, ctx.vertex_array(screen_prog, [
                (ctx.buffer(np.ascontiguousarray(fpos)), "3f", "in_pos"),
                (ctx.buffer(np.array([[0, 1], [1, 1], [1, 0], [0, 0]],
                                     np.float32)), "2f", "in_uv"),
                (ctx.buffer(np.zeros((4, 2), np.float32)), "2f", "in_scroll"),
            ], ctx.buffer(np.array([[0, 1, 2], [0, 2, 3]], np.uint32)))))

        leds = set_pieces.build_lights()
        light_prog = ctx.program(vertex_shader=LIGHT_VERT, fragment_shader=LIGHT_FRAG)
        set_lights = ctx.vertex_array(light_prog, [
            (ctx.buffer(leds["positions"]), "3f", "in_pos"),
            (ctx.buffer(leds["colors"]), "3f", "in_col"),
            (ctx.buffer(leds["phases"]), "1f", "in_phase"),
        ], ctx.buffer(leds["faces"]))

    samples = 8
    fbo_ms = ctx.framebuffer(
        color_attachments=[ctx.renderbuffer((W, H), samples=samples)],
        depth_attachment=ctx.depth_renderbuffer((W, H), samples=samples))
    fbo = ctx.framebuffer(color_attachments=[ctx.renderbuffer((W, H))])

    morph_mat = np.stack([head["morphs"][n] for n in head["names"]])  # (M,V,3)
    anim = Animator(head["names"], segments)

    date_line = args.date_line or "The Daily Prompt · daily AI news"
    chrome = {None: build_chrome(theme, date_line, None, args.source)}
    for s in sentences:
        chrome[s["text"]] = build_chrome(theme, date_line, s["text"], args.source)
    headline_layer = build_headline(theme, args.headline)
    # No hook card over the scene: it sat behind a scrim across the shot and
    # duplicated the headline. The hook still drives the thumbnail.
    hook_layer = None
    title_card = (build_title_card(theme, date_line, args.hook)
                  if args.title_card > 0 and not args.bare else None)
    hook_out = args.hook_hold + 0.45   # hook fades out, headline fades in

    scene_dur = duration - (0.0 if args.bare else args.title_card)
    configure_sequence(scene_dur, args.zoom_time)

    post = CrtPost(scanline=args.scanline, grain=args.grain)
    glitch = None
    if args.glitch > 0:
        gap = (2.8 / args.glitch, 7.0 / args.glitch)
        glitch = Glitch(duration, FPS, gap=gap)
        print(f"[render] {len(glitch.events)} glitch bursts scheduled")
    bg = theme["bg"]
    line_tint = np.array(theme["glow"][:3], dtype=np.float32) / 255.0

    def render_frame(t, dt):
        # Scene clock: the sequence starts when the title card cuts away.
        ts = t - (0.0 if args.bare else args.title_card)
        y_off = rise_at(ts)
        intro = rig_transform(ts, y_off)
        cap_m, _cap_lift = cap_transform(ts, y_off)
        arm_m = arm_transform(ts)
        level = water_level(ts)
        drain = drain_amount(ts)
        helmet_m = helmet_transform(ts)

        inf, (rx, ry, rz) = anim.step(t, dt)
        active = np.nonzero(inf > 1e-3)[0]
        pos = base
        if len(active):
            pos = base + np.tensordot(inf[active], morph_mat[active], axes=1)
        nrm = vertex_normals(pos, faces, flat_idx, nverts)
        rot = rotation(rx, ry, rz)
        # Slow vertical drift, as if neutrally buoyant in the fluid.
        bob = (math.sin(t * 2 * math.pi / args.float_period)
               * args.float_amount * buoyancy(ts))
        model = np.eye(4, dtype=np.float32)
        model[:3, :3] = rot
        model[:3, 3] = pivot - rot @ pivot
        model[1, 3] += head_lift + bob
        view, eye = shot_for(t)
        prog["cam_pos"].value = tuple(eye)
        # Intro only — the head stays put when the tube slides away at the end.
        head_model = intro_transform(ts, y_off) @ model
        prog["mvp"].write(np.ascontiguousarray((proj @ view @ head_model).T))
        prog["nrm_mat"].write(np.ascontiguousarray(head_model[:3, :3]))
        vbo_pos.write(np.ascontiguousarray(pos, dtype=np.float32))
        vbo_nrm.write(np.ascontiguousarray(nrm, dtype=np.float32))
        fbo_ms.use()
        ctx.clear(bg[0] / 255, bg[1] / 255, bg[2] / 255, 1.0)
        prog["glass"].value = 0.0
        prog["light_scale"].value = 1.0
        prog["emissive_scale"].value = 1.0
        vao.render()
        if set_opaque is not None:
            prog["light_scale"].value = args.set_gain
            prog["emissive_scale"].value = args.set_emissive
            eye3 = np.eye(3, dtype=np.float32)
            base_mvp = proj @ view

            # Wall is bolted to the world and never moves.
            prog["mvp"].write(np.ascontiguousarray(base_mvp.T))
            prog["nrm_mat"].write(np.ascontiguousarray(eye3))
            set_wall.render()

            # Rig rides the opening rise-and-spin.
            prog["mvp"].write(np.ascontiguousarray((base_mvp @ intro).T))
            prog["nrm_mat"].write(np.ascontiguousarray(intro[:3, :3]))
            set_rig.render()

            # Cap: rides the intro, then unscrews and lifts away on its own.
            prog["mvp"].write(np.ascontiguousarray((base_mvp @ cap_m).T))
            prog["nrm_mat"].write(np.ascontiguousarray(cap_m[:3, :3]))
            set_cap.render()

            if set_helmet is not None and helmet_m is not None:
                # Lit like the bust, not like the set: it ends up on the face.
                prog["light_scale"].value = 1.0
                prog["emissive_scale"].value = 1.0
                prog["mvp"].write(np.ascontiguousarray((base_mvp @ helmet_m).T))
                prog["nrm_mat"].write(np.ascontiguousarray(helmet_m[:3, :3]))
                set_helmet.render()
                set_helmet_wires.render()
                prog["light_scale"].value = args.set_gain
                prog["emissive_scale"].value = args.set_emissive

            if arm_m is not None:
                prog["light_scale"].value = 1.0
                prog["emissive_scale"].value = 1.0
                prog["mvp"].write(np.ascontiguousarray((base_mvp @ arm_m).T))
                prog["nrm_mat"].write(np.ascontiguousarray(arm_m[:3, :3]))
                set_arm.render()
                prog["light_scale"].value = args.set_gain
                prog["emissive_scale"].value = args.set_emissive
            if set_cables is not None:
                cable_prog["mvp"].write(np.ascontiguousarray((base_mvp @ intro).T))
                cable_prog["cam_pos"].value = tuple(eye)
                cable_prog["u_time"].value = t
                cable_prog["head_lift"].value = bob
                cable_prog["drain"].value = drain
                set_cables.render()
            screen_tex.use(0)
            screen_prog["mvp"].write(np.ascontiguousarray(base_mvp.T))
            screen_prog["u_time"].value = t
            set_screens.render()
            for ftex, fvao in feed_draws:
                ftex.use(0)
                fvao.render()
            light_prog["mvp"].write(np.ascontiguousarray(base_mvp.T))
            light_prog["u_time"].value = t
            set_lights.render()
            # Glass last, blended, without writing depth — otherwise the near
            # wall of the tube occludes the face behind it.
            prog["glass"].value = 1.0
            ctx.enable(moderngl.BLEND)
            # depth_mask lives on the Framebuffer, NOT the Context. Assigning
            # ctx.depth_mask silently creates a plain Python attribute and
            # changes no GL state at all, so the glass kept writing depth and
            # its near wall rejected every bubble behind it.
            fbo_ms.depth_mask = False
            prog["mvp"].write(np.ascontiguousarray((base_mvp @ intro).T))
            prog["nrm_mat"].write(np.ascontiguousarray(intro[:3, :3]))
            set_glass.render()

            # Fluid surface, dropping as the tube drains.
            prog["glass"].value = 1.0
            water_m = intro @ translate(0.0, level, 0.0)
            prog["mvp"].write(np.ascontiguousarray((base_mvp @ water_m).T))
            set_water.render()
            if set_bubbles is not None:
                # Billboards need the camera's own axes, and this shot's view
                # matrix rows are exactly that.
                bubble_prog["mvp"].write(np.ascontiguousarray((base_mvp @ intro).T))
                bubble_prog["y_hi"].value = level - 0.012
                bubble_prog["cam_right"].value = tuple(view[0, :3])
                bubble_prog["cam_up"].value = tuple(view[1, :3])
                bubble_prog["u_time"].value = t
                set_bubbles.render()
            fbo_ms.depth_mask = True
            ctx.disable(moderngl.BLEND)
        ctx.copy_framebuffer(fbo, fbo_ms)
        frame = np.frombuffer(fbo.read(components=3), dtype=np.uint8)
        frame = frame.reshape(H, W, 3)[::-1]

        if title_card is not None and t < args.title_card:
            return post(title_card.copy())

        if args.bare:
            # Set only, no masthead or type — for judging the 3D on its own.
            return post(frame)

        caption = next((s["text"] for s in sentences
                        if s["t0"] - 0.15 <= t <= s["t1"] + 0.35), None)
        out = over(frame, chrome[caption])
        pulse = 1.0
        if args.pulse > 0:
            pulse = 1.0 - args.pulse_depth * (0.5 - 0.5 * math.cos(
                2 * math.pi * t / args.pulse))

        # The big hook owns the opening, then hands the frame to the headline.
        hook_a = 0.0
        if hook_layer is not None and t < hook_out:
            hook_a = 1.0 if t <= args.hook_hold else \
                1.0 - (t - args.hook_hold) / (hook_out - args.hook_hold)
        if hook_a < 1.0:
            out = over(out, headline_layer, pulse * (1.0 - hook_a))
        if hook_a > 0.0:
            out = over(out, hook_layer, hook_a)
        out = out.astype(np.uint8)

        out = post(out)
        if glitch is not None:
            s = glitch.strength(round(t * FPS))
            if s:
                out = glitch.apply(out, s)
        if args.crt_intro == "full":
            intro = crt_power_on(out, t, line_tint)
        elif args.crt_intro == "flash":
            intro = crt_flash(out, t, line_tint)
        else:
            intro = None
        return intro if intro is not None else out

    if args.still is not None:
        for fp in feed_positions:
            view, _eye = shot_for(args.still)
            clip = (proj @ view @ np.vstack([fp.T, np.ones(4)])).T
            ndc = clip[:, :3] / clip[:, 3:4]
            px = (ndc[:, 0] * 0.5 + 0.5) * W
            py = (1 - (ndc[:, 1] * 0.5 + 0.5)) * H
            print(f"[feed] px x {px.min():.0f}-{px.max():.0f}  "
                  f"y {py.min():.0f}-{py.max():.0f}  ndc_z {ndc[0, 2]:.3f}")
        t0 = max(0.0, args.still - 1.0)
        for i in range(int((args.still - t0) * FPS)):  # warm up the smoothing
            anim.step(t0 + i / FPS, 1 / FPS)
        Image.fromarray(render_frame(args.still, 1 / FPS)).save(args.out)
        print(f"[render] still at t={args.still}s -> {args.out}")
        return

    # Voice is time-stretched to match the scaled timeline; the music bed is
    # not, and is ducked under the narration and faded at both ends.
    # Input 0 is the raw video pipe, so the narration is input 1.
    inputs = ["-i", args.audio or str(work / "audio.wav")]
    chains = [f"[1:a]atempo={speed:.4f}[voice]"]
    labels = ["[voice]"]
    idx = 2
    if args.music:
        inputs += ["-stream_loop", "-1", "-i", args.music]
        # A fade-in destroys the bed's own downbeat. Set --music-fade-in 0 to
        # let the music's first hit be the opening accent, instead of laying a
        # separate stinger on top of it.
        fade_in = (f"afade=t=in:st=0:d={args.music_fade_in:.2f},"
                   if args.music_fade_in > 0 else "")
        chains.append(
            f"[{idx}:a]volume={args.music_gain}dB,{fade_in}"
            f"afade=t=out:st={max(0.0, duration - 2.0):.2f}:d=2.0[bed]")
        labels.append("[bed]")
        idx += 1
    if args.sfx and Path(args.sfx).is_file():
        inputs += ["-i", args.sfx]
        chains.append(f"[{idx}:a]volume={args.sfx_gain}dB[sfx]")
        labels.append("[sfx]")
        idx += 1
    if args.scene_sfx:
        # Event sounds generated for THIS clip's beat times, so they follow
        # the sequence wherever configure_sequence() put it. Clip time, not
        # scene time: the track is laid against the final video clock.
        import scene_sfx
        tc = 0.0 if args.bare else args.title_card
        beats = [("zoom", tc, tc + args.zoom_time),
                 ("arm_down", tc + ARM_IN, tc + ARM_GRIP),
                 ("unscrew", tc + ARM_GRIP, tc + UNSCREW_END),
                 ("tube_sink", tc + EXIT_START, tc + EXIT_END),
                 ("zoom", tc + DRAIN_END, tc + DRAIN_END + args.final_zoom_time),
                 ("helmet_down", tc + HELMET_IN, tc + HELMET_SEAT),
                 ("seat_click", tc + HELMET_SEAT, tc + HELMET_SEAT + 0.6)]
        scene_wav = work / "_scene_sfx.wav"
        scene_sfx.save(scene_sfx.build_track(duration, beats), scene_wav)
        inputs += ["-i", str(scene_wav)]
        chains.append(f"[{idx}:a]volume={args.scene_sfx_gain}dB[scn]")
        labels.append("[scn]")
        idx += 1
    if len(labels) > 1:
        # normalize=0: amix's default divides by the input count, so the level
        # would jump upward the moment the 2 s power-on sound ends.
        chains.append("".join(labels) + f"amix=inputs={len(labels)}:duration=first"
                      f":dropout_transition=0:normalize=0,{LOUDNESS}[aout]")
    else:
        chains.append(f"[voice]{LOUDNESS}[aout]")

    if args.max_seconds > 0:
        duration = min(duration, args.max_seconds)
    nframes = int(math.ceil(duration * FPS))
    ff = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-", *inputs,
        "-filter_complex", ";".join(chains), "-map", "0:v", "-map", "[aout]",
        # Grain is expensive to encode; cap the bitrate so a 40 s short stays
        # in the tens of megabytes instead of the hundreds.
        "-c:v", "libx264", "-preset", "medium", "-crf", "22",
        "-maxrate", "8M", "-bufsize", "16M",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
        "-shortest", "-movflags", "+faststart", args.out,
    ], stdin=subprocess.PIPE)

    for i in range(nframes):
        ff.stdin.write(render_frame(i / FPS, 1 / FPS).tobytes())
        if i % (FPS * 5) == 0:
            print(f"[render] {i}/{nframes} frames ({i / FPS:.0f}s)")
    ff.stdin.close()
    ff.wait()
    if ff.returncode:
        sys.exit(f"ffmpeg failed with code {ff.returncode}")
    print(f"[render] done: {nframes} frames, {duration:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
