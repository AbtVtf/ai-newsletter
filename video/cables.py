"""Cables and ridged hoses running from the neck down into the pedestal.

Tubes swept along curves, which is what a cable is. Generated rather than
modelled so the count, sag and thickness are all parameters, and so they can
be rebuilt if the head or pedestal ever moves.

  ../jarvis-agent/.venv/bin/python video/cables.py --out cables.glb
"""

import argparse

import numpy as np
import trimesh

# The model is a bust, not a head: it includes shoulders and chest down to
# y = 0.066, where it ends in a flat underside about 0.098 across. Cables drop
# out of THAT, not out of the neck — a cable emerging from someone's throat
# reads as a mistake.
BUST_BOTTOM = 0.066
BUST_R = 0.058          # inside the 0.098 underside, so they meet solid body
BASE_Y = 0.050
BASE_R = 0.140
Z_CENTER = 0.0271
# The bust floats clear of the pedestal so there is a gap for cables to hang
# in. Must match HEAD_LIFT in render_face.py.
HEAD_LIFT = 0.055

COLORS = {
    "hose": (86, 158, 106),
    "cable": (54, 112, 72),
    "collar": (150, 224, 172),
    # Runs heading to the wall sit further back, so they are darker to match
    # the set dressing rather than the bust.
    "conduit": (40, 88, 56),
}


def bezier(p0, p1, p2, p3, n):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


def sweep(path, radii, sections=14):
    """Sweep a circular cross-section along a path.

    Frames are built from a running reference vector rather than the curve's
    normal: a straight segment has no defined normal, and a cable that goes
    briefly straight would otherwise flip its ring and pinch.
    """
    path = np.asarray(path, dtype=np.float64)
    tangents = np.gradient(path, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)

    ref = np.array([0.0, 0.0, 1.0])
    verts, faces = [], []
    ring_angles = np.linspace(0, 2 * np.pi, sections, endpoint=False)

    for i, (p, tan) in enumerate(zip(path, tangents)):
        if abs(np.dot(ref, tan)) > 0.95:
            ref = np.array([1.0, 0.0, 0.0])
        u = np.cross(tan, ref)
        u /= max(np.linalg.norm(u), 1e-9)
        v = np.cross(tan, u)
        ref = v
        r = radii[i]
        verts.append(p[None, :] + r * (np.cos(ring_angles)[:, None] * u[None, :]
                                       + np.sin(ring_angles)[:, None] * v[None, :]))
        if i:
            a = (i - 1) * sections
            b = i * sections
            for k in range(sections):
                k2 = (k + 1) % sections
                faces.append([a + k, b + k, b + k2])
                faces.append([a + k, b + k2, a + k2])

    return trimesh.Trimesh(vertices=np.vstack(verts),
                           faces=np.array(faces, dtype=np.int64),
                           process=False)


# Where a run can terminate. Mixing these is most of what stops the bundle
# looking like one cable copied around a circle.
WALL_Z = -0.34
# Tube interior, so runs that terminate inside never pierce the glass.
# Must track TUBE_TOP/TUBE_RADIUS and the cap height in set_pieces.py.
TUBE_TOP_INNER = 0.446
TUBE_INNER_R = 0.150
# Where the two back runs plug into the cap: out at the sides, clear of the
# glass. sqrt(x^2 + z^2) must stay under TUBE_INNER_R.
BACK_PORT_X = 0.128
BACK_PORT_Z = -0.040


def run(angle, radius, kind, seed, points=90):
    """One run. `kind` changes its shape, surface and where it terminates.

    hose   thick corrugated, sags into the pedestal
    cable  thin and smooth, deeper sag
    coil   helical, like flex conduit
    taut   nearly straight and tight, no slack
    wall   leaves the bust sideways and terminates on the back wall
    """
    rng = np.random.default_rng(seed)
    ca, sa = np.cos(angle), np.sin(angle)
    start = np.array([BUST_R * ca, BUST_BOTTOM + HEAD_LIFT, Z_CENTER + BUST_R * sa])

    if kind == "back":
        # Anchored to the BACK of the bust, arcing up behind the head and
        # plugging into the underside of the cap. Stays inside the tube:
        # everything below is bounded by TUBE_TOP_INNER in y and by
        # TUBE_INNER_R from the axis, so nothing pierces the glass.
        start = np.array([BUST_R * 0.95 * ca, 0.205 + HEAD_LIFT, Z_CENTER - 0.085])
        # Terminate out at the left and right edges of the cap rather than
        # near its centre, so the two runs read as plugged into opposite sides.
        end = np.array([ca * BACK_PORT_X, TUBE_TOP_INNER, Z_CENTER + BACK_PORT_Z])
        # Length comes from bowing backwards across the head's own height, NOT
        # from dipping toward the shoulders: down there the bust is at its
        # widest, so the run has to sit further back, and radius is the scarce
        # resource here. The glass is at TUBE_INNER_R and the sway displaces
        # the middle outward, so mid-run has to stay well inside it.
        c1 = np.array([ca * 0.030, 0.30, Z_CENTER - 0.110])
        c2 = np.array([ca * 0.070, TUBE_TOP_INNER - rng.uniform(0.05, 0.09),
                       Z_CENTER - 0.095])
    elif kind == "wall":
        # Out sideways and BACK, terminating low. Routing these upward puts a
        # cable straight across the face, which is the one place nothing
        # should cross.
        end = np.array([ca * 0.44, rng.uniform(0.01, 0.10), WALL_Z + 0.09])
        c1 = start + np.array([ca * 0.18, -0.045, sa * 0.18])
        c2 = np.array([ca * 0.36, rng.uniform(-0.02, 0.04),
                       (Z_CENTER + WALL_Z) / 2])
    else:
        end = np.array([BASE_R * ca * 0.82, BASE_Y, Z_CENTER + BASE_R * sa * 0.82])
        # Keep the bulge inside the glass. These used to reach a
        # control point at radius 0.32 against a 0.150 wall, so most
        # runs intersected the tube even before they detached.
        sag = {"hose": 1.25, "cable": 1.45, "coil": 1.15, "taut": 1.0}[kind]
        sag += rng.uniform(-0.12, 0.18)
        drop = -0.055 if kind != "taut" else -0.015
        c1 = start + np.array([BUST_R * ca * 0.4, drop, BUST_R * sa * 0.4])
        c2 = np.array([BASE_R * ca * sag * 0.8,
                       BASE_Y - (0.03 if kind == "cable" else 0.012),
                       Z_CENTER + BASE_R * sa * sag * 0.8])
    path = bezier(start, c1, c2, end, points)

    if kind == "coil":
        # Wrap the path in a small helix so it reads as flexible conduit.
        s = np.linspace(0, 1, points)
        turns = rng.uniform(7, 11)
        amp = 0.011 * np.sin(np.pi * s)          # zero at both fittings
        path = path + np.stack([amp * np.cos(2 * np.pi * turns * s),
                                np.zeros(points),
                                amp * np.sin(2 * np.pi * turns * s)], axis=1)

    s = np.linspace(0, 1, points)
    if kind in ("hose", "wall", "back"):
        radii = radius * (1.0 + 0.22 * np.sin(s * points * 0.9))
    elif kind == "coil":
        radii = radius * (1.0 + 0.10 * np.sin(s * points * 1.6))
    elif kind == "taut":
        radii = radius * (1.0 - 0.10 * np.sin(np.pi * s))   # thins under tension
    else:
        radii = np.full(points, radius)
    # Taper into the sockets at both ends so nothing pokes through.
    radii = radii * np.clip(np.minimum(s * 12, (1 - s) * 12), 0.55, 1.0)
    return sweep(path, radii, sections=16 if kind in ("hose", "coil", "wall", "back") else 10)


def attach_param(mesh, points, sections):
    """Per-vertex position along the cable: 1 at the bust end, 0 at the base.

    The shader uses it to carry the head end with the floating bust and to
    keep both anchored ends still while the slack middle drifts.
    """
    per_ring = np.linspace(1.0, 0.0, points, dtype=np.float32)
    return np.repeat(per_ring, sections)[:len(mesh.vertices)]


def build(seed=4):
    rng = np.random.default_rng(seed)
    meshes = []

    # Deliberately mixed: thicknesses, surfaces, sag and destinations all
    # differ, and the angles are uneven. Ten identical runs spaced evenly
    # around a circle is what made the first version look like a hair plug.
    specs = [(0.42, 0.0180, "hose"), (2.10, 0.0160, "hose"),
             (3.75, 0.0172, "hose"), (5.10, 0.0150, "coil"),
             (1.32, 0.0130, "coil"), (2.62, 0.0082, "cable"),
             (4.30, 0.0074, "cable"), (5.72, 0.0090, "cable"),
             (0.08, 0.0060, "taut"), (3.20, 0.0055, "taut"),
             (1.05, 0.0068, "wall"), (4.02, 0.0072, "wall")]
    palette = {"hose": "hose", "coil": "hose", "cable": "cable",
               "taut": "cable", "wall": "conduit", "back": "hose"}
    # How freely each kind drifts. A taut line barely moves; a long slack run
    # to the wall wanders. Uniform sway is what made them look mechanical.
    sway_by_kind = {"wall": 2.6, "back": 1.3, "cable": 1.7, "hose": 1.0,
                    "coil": 0.7, "taut": 0.25}
    for i, (ang, rad, kind) in enumerate(specs):
        jitter = float(rng.uniform(-0.16, 0.16))
        mesh = run(ang + jitter, rad, kind, seed=seed * 31 + i)
        sections = 16 if kind in ("hose", "coil", "wall", "back") else 10
        mul = sway_by_kind[kind] * float(rng.uniform(0.75, 1.3))
        meshes.append((mesh, COLORS[palette[kind]],
                       attach_param(mesh, 90, sections),
                       float(rng.uniform(0, 6.283)), mul,
                       0.0 if kind == "wall" else 1.0))

    # Collars: one on the bust's underside (rides with it, attach=1) and one
    # on the pedestal (bolted down, attach=0).
    for y, r_in, r_out, h, attach in (
            (BUST_BOTTOM + HEAD_LIFT - 0.005, 0.046, 0.070, 0.013, 1.0),
            (BASE_Y + 0.002, 0.112, 0.146, 0.012, 0.0)):
        ring = trimesh.creation.annulus(r_min=r_in, r_max=r_out, height=h,
                                        sections=64)
        ring.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi / 2, [1, 0, 0]))
        ring.apply_translation([0.0, y, Z_CENTER])
        meshes.append((ring, COLORS["collar"],
                       np.full(len(ring.vertices), attach, dtype=np.float32),
                       0.0, 0.0, 1.0))
    return meshes


def pack_animated(meshes):
    """Flat arrays for the renderer, including the sway attributes."""
    pos, nrm, col, att, pha, swy, ins, fac, off = [], [], [], [], [], [], [], [], 0
    for mesh, rgb, attach, phase, sway_mul, in_tube in meshes:
        v = np.asarray(mesh.vertices, dtype=np.float32)
        pos.append(v)
        nrm.append(np.asarray(mesh.vertex_normals, dtype=np.float32))
        col.append(np.tile(np.array(rgb, np.float32) / 255.0, (len(v), 1)))
        att.append(np.asarray(attach, dtype=np.float32)[:len(v)])
        pha.append(np.full(len(v), phase, dtype=np.float32))
        swy.append(np.full(len(v), sway_mul, dtype=np.float32))
        ins.append(np.full(len(v), in_tube, dtype=np.float32))
        fac.append(np.asarray(mesh.faces, dtype=np.uint32) + off)
        off += len(v)
    return {"positions": np.ascontiguousarray(np.concatenate(pos)),
            "normals": np.ascontiguousarray(np.concatenate(nrm)),
            "colors": np.ascontiguousarray(np.concatenate(col).astype(np.float32)),
            "attach": np.ascontiguousarray(np.concatenate(att)),
            "phase": np.ascontiguousarray(np.concatenate(pha)),
            "sway_mul": np.ascontiguousarray(np.concatenate(swy)),
            "in_tube": np.ascontiguousarray(np.concatenate(ins)),
            "faces": np.ascontiguousarray(np.concatenate(fac))}


def pack(meshes):  # legacy: colour-only pack for GLB export
    pos, nrm, col, fac, off = [], [], [], [], 0
    for mesh, rgb in meshes:
        v = np.asarray(mesh.vertices, dtype=np.float32)
        pos.append(v)
        nrm.append(np.asarray(mesh.vertex_normals, dtype=np.float32))
        col.append(np.tile(np.array(rgb, np.float32) / 255.0, (len(v), 1)))
        fac.append(np.asarray(mesh.faces, dtype=np.uint32) + off)
        off += len(v)
    return {"positions": np.ascontiguousarray(np.concatenate(pos)),
            "normals": np.ascontiguousarray(np.concatenate(nrm)),
            "colors": np.ascontiguousarray(np.concatenate(col).astype(np.float32)),
            "faces": np.ascontiguousarray(np.concatenate(fac))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="path to write a .glb")
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args()

    meshes = build(args.seed)
    scene = trimesh.Scene()
    for i, (mesh, rgb, *_rest) in enumerate(meshes):
        mesh.visual.vertex_colors = np.tile(
            np.array(list(rgb) + [255], np.uint8), (len(mesh.vertices), 1))
        scene.add_geometry(mesh, node_name=f"Cable_{i:02d}")
    scene.export(args.out)
    tris = sum(len(m.faces) for m, *_ in meshes)
    print(f"[cables] {len(meshes)} pieces, {tris} tris -> {args.out}")


if __name__ == "__main__":
    main()
