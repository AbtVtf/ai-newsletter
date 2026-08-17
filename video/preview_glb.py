"""One-still preview of any glb in the house look, no fitting logic.

For judging raw generator output (TRELLIS, Pixal3D, AniGen) before it goes
through prop-specific pipelines like helmet.py.

  ../jarvis-agent/.venv/bin/python video/preview_glb.py --glb m.glb --out m.png
"""

import argparse
import math

import moderngl
import numpy as np
import trimesh
from PIL import Image

import render_face as rf
from design import THEMES, W, H, CrtPost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-faces", type=int, default=400_000)
    ap.add_argument("--yaw", type=float, default=25.0, help="view angle, deg")
    ap.add_argument("--theme", default="phosphor")
    args = ap.parse_args()

    m = trimesh.load(args.glb, force="mesh")
    if len(m.faces) > args.max_faces:
        import fast_simplification
        v, f = fast_simplification.simplify(
            m.vertices.astype(np.float32), m.faces.astype(np.int64),
            target_reduction=1.0 - args.max_faces / len(m.faces))
        m = trimesh.Trimesh(v, f)
    print(f"[preview] {len(m.faces)} faces after decimation")

    pos = m.vertices - m.bounding_box.centroid
    scale = 1.0 / max(m.bounding_box.extents)
    pos = (pos * scale).astype(np.float32)
    faces = np.ascontiguousarray(m.faces, dtype=np.uint32)
    flat = faces.reshape(-1)
    nrm = rf.vertex_normals(pos, faces, flat, len(pos))
    col = np.tile(np.array([0.62, 0.68, 0.63], np.float32), (len(pos), 1))

    theme = THEMES[args.theme]
    ctx = moderngl.create_context(standalone=True, backend="egl")
    ctx.enable(moderngl.DEPTH_TEST)
    prog = ctx.program(vertex_shader=rf.VERT, fragment_shader=rf.FRAG)
    for tk, un in (("hemi_sky", "hemi_sky"), ("hemi_ground", "hemi_ground"),
                   ("key", "key_col"), ("rim", "rim_col"),
                   ("rim2", "rim2_col"), ("emissive", "emissive")):
        prog[un].value = tuple(theme[tk])
    prog["exposure"].value = 1.0
    prog["glass"].value = 0.0
    prog["light_scale"].value = 1.0
    prog["emissive_scale"].value = 1.0

    vao = ctx.vertex_array(prog, [
        (ctx.buffer(np.ascontiguousarray(pos)), "3f", "in_pos"),
        (ctx.buffer(np.ascontiguousarray(nrm)), "3f", "in_nrm"),
        (ctx.buffer(col), "3f", "in_col")], ctx.buffer(faces))

    yaw = math.radians(args.yaw)
    eye = np.array([math.sin(yaw) * 1.9, 0.25, math.cos(yaw) * 1.9])
    proj = rf.perspective(32.0, W / H, 0.05, 20.0)
    view = rf.look_at(eye, np.zeros(3))
    prog["cam_pos"].value = tuple(eye)
    prog["mvp"].write(np.ascontiguousarray((proj @ view).T, dtype=np.float32))
    prog["nrm_mat"].write(np.eye(3, dtype=np.float32))

    samples = 8
    fbo_ms = ctx.framebuffer(
        color_attachments=[ctx.renderbuffer((W, H), samples=samples)],
        depth_attachment=ctx.depth_renderbuffer((W, H), samples=samples))
    fbo = ctx.framebuffer(color_attachments=[ctx.renderbuffer((W, H))])
    fbo_ms.use()
    bg = theme["bg"]
    ctx.clear(bg[0] / 255, bg[1] / 255, bg[2] / 255, 1.0)
    vao.render()
    ctx.copy_framebuffer(fbo, fbo_ms)
    frame = np.frombuffer(fbo.read(components=3), dtype=np.uint8)
    frame = frame.reshape(H, W, 3)[::-1]
    Image.fromarray(CrtPost(scanline=0.10, grain=4.0)(frame)).save(args.out)
    print(f"[preview] -> {args.out}")


if __name__ == "__main__":
    main()
