"""ARDY insert clip in the house look: the skinned actor on a dark stage,
lit and post-processed exactly like the tube scene, so cutaways match.

The stage is deliberately minimal — floor, back wall, optional carried box.
Inserts are mute; the narration continues over them and the splice happens
in the edit, not here.

Run with the jarvis-agent venv:
  ../jarvis-agent/.venv/bin/python video/render_actor.py \
      --motion ../ardy/outputs/typing_s7.npz --out typing.mp4
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

import render_face as rf
from actor import Actor
from design import THEMES, W, H, CrtPost, over
from set_pieces import COLORS

FPS = 30


def _col(name):
    return np.array(COLORS[name], dtype=np.float32) / 255.0


def quad(center, size, axis, rgb):
    """A flat rectangle. axis 'y' = floor (facing up), 'z' = wall (facing +Z)."""
    cx, cy, cz = center
    sx, sy = size
    if axis == "y":
        v = [[cx - sx, cy, cz - sy], [cx + sx, cy, cz - sy],
             [cx + sx, cy, cz + sy], [cx - sx, cy, cz + sy]]
        n = [0.0, 1.0, 0.0]
    else:
        v = [[cx - sx, cy - sy, cz], [cx + sx, cy - sy, cz],
             [cx + sx, cy + sy, cz], [cx - sx, cy + sy, cz]]
        n = [0.0, 0.0, 1.0]
    return {
        "positions": np.ascontiguousarray(v, dtype=np.float32),
        "normals": np.ascontiguousarray(np.tile(n, (4, 1)), dtype=np.float32),
        "colors": np.ascontiguousarray(np.tile(rgb, (4, 1)), dtype=np.float32),
        "faces": np.ascontiguousarray([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
    }


def box_mesh(sx, sy, sz, rgb):
    """Axis-aligned box centred on the origin, flat-shaded."""
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    faces_def = [  # (normal, four corners CCW seen from outside)
        ((0, 0, 1), [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]),
        ((0, 0, -1), [(hx, -hy, -hz), (-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz)]),
        ((1, 0, 0), [(hx, -hy, hz), (hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz)]),
        ((-1, 0, 0), [(-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz)]),
        ((0, 1, 0), [(-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz), (-hx, hy, -hz)]),
        ((0, -1, 0), [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)]),
    ]
    # Per-face albedo variation: a single-colour box under one flat normal
    # per face reads as a pasted rectangle, not an object.
    shade = [0.95, 0.7, 0.8, 0.8, 1.2, 0.5]
    pos, nrm, col, idx = [], [], [], []
    for i, (n, corners) in enumerate(faces_def):
        pos += corners
        nrm += [n] * 4
        col += [np.asarray(rgb) * shade[i]] * 4
        b = i * 4
        idx += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    return {
        "positions": np.ascontiguousarray(pos, dtype=np.float32),
        "normals": np.ascontiguousarray(nrm, dtype=np.float32),
        "colors": np.ascontiguousarray(col, dtype=np.float32),
        "faces": np.ascontiguousarray(idx, dtype=np.uint32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True, help="ARDY .npz from generate.py")
    ap.add_argument("--skin", default="", help="override skin_standard.npz path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--theme", default="phosphor", choices=sorted(THEMES))
    ap.add_argument("--exposure", type=float, default=1.0)
    ap.add_argument("--grain", type=float, default=6.0)
    ap.add_argument("--scanline", type=float, default=0.10)
    ap.add_argument("--fov", type=float, default=26.0)
    ap.add_argument("--height-frac", type=float, default=0.72,
                    help="motion bbox height as a fraction of frame height")
    ap.add_argument("--still", type=float, default=None,
                    help="render one PNG at this time instead of the clip")
    ap.add_argument("--max-seconds", type=float, default=0)
    ap.add_argument("--trim-start", type=float, default=0.0,
                    help="skip the first seconds of the motion (settling)")
    ap.add_argument("--box", default="",
                    help="carried box 'SXxSYxSZ@X,Y,Z', metres; e.g. "
                         "'0.42x0.30x0.34@0,0.15,2.0'")
    ap.add_argument("--box-attach", default="auto",
                    help="seconds when the box starts following the hands; "
                         "'auto' picks the frame the hands are closest to it; "
                         "'never' keeps it static")
    ap.add_argument("--cam-yaw", type=float, default=0.0,
                    help="orbit the camera this many degrees around the actor")
    ap.add_argument("--push", type=float, default=0.0,
                    help="slow push-in: fraction of distance covered over the "
                         "clip, e.g. 0.14")
    ap.add_argument("--captions", default="",
                    help='JSON [{"text","t0","t1"}] in insert-local seconds; '
                         "shown with the standard chrome")
    ap.add_argument("--headline", default="", help="standing headline overlay")
    ap.add_argument("--date-line", default="")
    ap.add_argument("--source", default="")
    args = ap.parse_args()

    theme = THEMES[args.theme]
    act = Actor(args.motion, skin_path=args.skin or None)
    dur = act.duration - args.trim_start
    if args.max_seconds > 0:
        dur = min(dur, args.max_seconds)

    lo, hi = act.bounds()
    center = (lo + hi) / 2
    span = hi - lo
    travel = float(np.linalg.norm(act.joints[-1, 0, [0, 2]]
                                  - act.joints[0, 0, [0, 2]]))
    print(f"[actor] {Path(args.motion).name}: {act.num_frames}f @ {act.fps:g}fps "
          f"({act.duration:.1f}s)  bbox {span[0]:.2f}x{span[1]:.2f}x{span[2]:.2f}m  "
          f"root travel {travel:.2f}m")
    if act.text:
        print(f"[actor] prompt: {act.text[:90]}")

    # Static camera fitted to the whole clip. Fit height AND width — walks
    # travel sideways and portrait is narrow.
    aspect = W / H
    half_fov = math.tan(math.radians(args.fov / 2))
    pad = 0.18  # body extends past the joints (head top, hands)
    dist_h = (span[1] + pad) / (2 * args.height_frac * half_fov)
    dist_w = (span[0] + pad) / (2 * 0.86 * aspect * half_fov)
    dist = max(dist_h, dist_w, 1.2)
    proj = rf.perspective(args.fov, aspect, 0.05, 60.0)
    target = np.array([center[0], center[1], center[2]], dtype=np.float64)
    yaw = math.radians(args.cam_yaw)

    def camera_at(t):
        f = min(1.0, t / max(dur, 1e-6))
        r = (hi[2] - center[2]) + dist * (1.0 + args.push * (0.5 - f))
        eye = target + np.array([math.sin(yaw) * r, 0.04, math.cos(yaw) * r])
        return rf.look_at(eye, target), eye

    box = None
    if args.box:
        size_s, pos_s = args.box.split("@")
        bs = [float(x) for x in size_s.split("x")]
        bp = np.array([float(x) for x in pos_s.split(",")], dtype=np.float64)
        box = {"size": bs, "pos": bp}
        lh = act.joint_names.index("LeftHand")
        rh = act.joint_names.index("RightHand")
        mid = (act.joints[:, lh] + act.joints[:, rh]) / 2
        d = np.linalg.norm(mid - bp, axis=1)
        if args.box_attach == "auto":
            box["attach_t"] = float(np.argmin(d)) / act.fps
            print(f"[actor] box attaches at {box['attach_t']:.2f}s "
                  f"(hands {d.min() * 100:.0f}cm from box)")
        elif args.box_attach == "never":
            box["attach_t"] = float("inf")
        else:
            box["attach_t"] = float(args.box_attach)
        # Constant grip offset, measured at the attach frame, so the box
        # neither snaps nor drifts relative to the hands while carried.
        if math.isfinite(box["attach_t"]):
            fa = min(int(box["attach_t"] * act.fps), act.num_frames - 1)
            box["grip"] = bp - mid[fa]

    ctx = moderngl.create_context(standalone=True, backend="egl")
    ctx.enable(moderngl.DEPTH_TEST)
    prog = ctx.program(vertex_shader=rf.VERT, fragment_shader=rf.FRAG)
    for tk, un in (("hemi_sky", "hemi_sky"), ("hemi_ground", "hemi_ground"),
                   ("key", "key_col"), ("rim", "rim_col"),
                   ("rim2", "rim2_col"), ("emissive", "emissive")):
        prog[un].value = tuple(theme[tk])
    prog["exposure"].value = args.exposure
    prog["glass"].value = 0.0

    vbo_pos = ctx.buffer(reserve=act.nverts * 12, dynamic=True)
    vbo_nrm = ctx.buffer(reserve=act.nverts * 12, dynamic=True)
    vbo_col = ctx.buffer(act.colors)
    actor_vao = ctx.vertex_array(prog, [(vbo_pos, "3f", "in_pos"),
                                        (vbo_nrm, "3f", "in_nrm"),
                                        (vbo_col, "3f", "in_col")],
                                 ctx.buffer(np.ascontiguousarray(
                                     act.faces, dtype=np.uint32)))

    def static_vao(mesh):
        return ctx.vertex_array(prog, [
            (ctx.buffer(mesh["positions"]), "3f", "in_pos"),
            (ctx.buffer(mesh["normals"]), "3f", "in_nrm"),
            (ctx.buffer(mesh["colors"]), "3f", "in_col"),
        ], ctx.buffer(mesh["faces"]))

    wall_z = lo[2] - 1.1
    stage = [static_vao(quad((center[0], 0.0, center[2]), (7.0, 7.0), "y",
                             _col("grille"))),
             static_vao(quad((center[0], 1.4, wall_z), (7.0, 1.6), "z",
                             _col("panel")))]
    box_vao = static_vao(box_mesh(*box["size"], _col("trim"))) if box else None

    samples = 8
    fbo_ms = ctx.framebuffer(
        color_attachments=[ctx.renderbuffer((W, H), samples=samples)],
        depth_attachment=ctx.depth_renderbuffer((W, H), samples=samples))
    fbo = ctx.framebuffer(color_attachments=[ctx.renderbuffer((W, H))])

    post = CrtPost(scanline=args.scanline, grain=args.grain)
    eye3 = np.eye(3, dtype=np.float32)
    ident = np.eye(4, dtype=np.float32)
    bg = theme["bg"]

    captions = json.loads(args.captions) if args.captions else []
    chrome = {}
    headline_layer = None
    if captions or args.headline:
        date_line = args.date_line or "The Daily Prompt · daily AI news"
        chrome[None] = rf.build_chrome(theme, date_line, None, args.source)
        for c in captions:
            chrome[c["text"]] = rf.build_chrome(theme, date_line, c["text"],
                                                args.source)
        if args.headline:
            headline_layer = rf.build_headline(theme, args.headline)

    def render_frame(t):
        tm = t + args.trim_start
        pos = act.pose(tm)
        nrm = rf.vertex_normals(pos, act.faces, act.flat_idx, act.nverts)
        vbo_pos.write(np.ascontiguousarray(pos, dtype=np.float32))
        vbo_nrm.write(np.ascontiguousarray(nrm, dtype=np.float32))

        view, eye = camera_at(t)
        base_mvp = proj @ view
        prog["cam_pos"].value = tuple(eye)

        fbo_ms.use()
        ctx.clear(bg[0] / 255, bg[1] / 255, bg[2] / 255, 1.0)

        prog["mvp"].write(np.ascontiguousarray(base_mvp.T, dtype=np.float32))
        prog["nrm_mat"].write(eye3)
        prog["light_scale"].value = 0.42
        prog["emissive_scale"].value = 0.10
        for vao in stage:
            vao.render()

        prog["light_scale"].value = 1.0
        prog["emissive_scale"].value = 1.0
        actor_vao.render()

        if box_vao is not None:
            m = ident.copy()
            if tm >= box["attach_t"]:
                lh = act.joint_matrix(tm, "LeftHand")[:3, 3]
                rh = act.joint_matrix(tm, "RightHand")[:3, 3]
                m[:3, 3] = (lh + rh) / 2 + box["grip"]
            else:
                m[:3, 3] = box["pos"]
            prog["mvp"].write(np.ascontiguousarray((base_mvp @ m).T,
                                                   dtype=np.float32))
            prog["light_scale"].value = 0.5
            prog["emissive_scale"].value = 0.10
            box_vao.render()

        ctx.copy_framebuffer(fbo, fbo_ms)
        frame = np.frombuffer(fbo.read(components=3), dtype=np.uint8)
        frame = frame.reshape(H, W, 3)[::-1]
        if chrome:
            cap = next((c["text"] for c in captions
                        if c["t0"] - 0.15 <= t <= c["t1"] + 0.35), None)
            out = over(frame, chrome[cap])
            if headline_layer is not None:
                out = over(out, headline_layer)
            frame = out.astype(np.uint8)
        return post(frame)

    if args.still is not None:
        Image.fromarray(render_frame(args.still)).save(args.out)
        print(f"[render] still at t={args.still}s -> {args.out}")
        return

    nframes = int(math.ceil(dur * FPS))
    ff = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", "22",
        "-maxrate", "8M", "-bufsize", "16M",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out,
    ], stdin=subprocess.PIPE)
    for i in range(nframes):
        ff.stdin.write(render_frame(i / FPS).tobytes())
        if i % (FPS * 5) == 0:
            print(f"[render] {i}/{nframes} frames")
    ff.stdin.close()
    ff.wait()
    if ff.returncode:
        sys.exit(f"ffmpeg failed with code {ff.returncode}")
    print(f"[render] done: {nframes} frames, {dur:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
