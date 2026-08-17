"""The visor helmet: a TRELLIS mesh, normalized and fitted to the head.

This is the one prop TRELLIS is genuinely right for. A helmet is a single
organic-ish object photographed from one angle, which is exactly what
image-to-3D does well — unlike a wall that has to tile, or a cable that is a
tube swept along a curve.

The raw mesh arrives arbitrarily scaled, centred and oriented, so everything
here is about pinning it to the head rather than trusting what came out.

  ../jarvis-agent/.venv/bin/python video/helmet.py --check
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh

GLB = Path(__file__).resolve().parent / "assets" / "props" / "helmet4" / "sample.glb"

# Head measurements (world units), from the packed bust:
#   head spans y 0.253 (neck) .. 0.407 (crown), half-width 0.0914
#   z centre 0.027, and the whole bust is lifted by HEAD_LIFT at render time.
HEAD_HALF_W = 0.0914
HEAD_TOP = 0.4071
HEAD_LIFT = 0.055
EYE_Y = 0.3300 + HEAD_LIFT     # visor should land across here
Z_CENTER = 0.0271

# Fitted so the shell clears the skull rather than intersecting it.
WIDTH_SCALE = 1.28             # helmet width relative to the head's
SHELL_HEIGHT = 0.190         # crown to just under the cheekbones; jaw is cut
CROWN_CLEARANCE = 0.012        # shell sits this far above the skull
TARGET_FACES = 45000
COLOR = (118, 196, 132)


def load(path=GLB, rot_x=0.0, rot_y=0.0, rot_z=0.0, target_faces=TARGET_FACES):
    """Load, decimate, orient, then scale and seat it on the head."""
    scene = trimesh.load(str(path), process=False)
    mesh = scene.to_mesh() if hasattr(scene, "to_mesh") else scene

    if target_faces and len(mesh.faces) > target_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
        except Exception as exc:                       # noqa: BLE001
            print(f"[helmet] decimation unavailable ({type(exc).__name__}); "
                  f"keeping {len(mesh.faces)} faces")

    for angle, axis in ((rot_x, [1, 0, 0]), (rot_y, [0, 1, 0]), (rot_z, [0, 0, 1])):
        if angle:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(
                np.radians(angle), axis))

    v = np.asarray(mesh.vertices, dtype=np.float64)
    lo, hi = v.min(axis=0), v.max(axis=0)
    v -= (lo + hi) / 2                                  # centre on its own bbox
    # Fit by WIDTH: a helmet narrower than the skull cannot enclose it, and
    # fitting by height did exactly that — this mesh is a full-face design
    # with a long jaw (width/height 0.83, where a head is about 1.19).
    v *= (HEAD_HALF_W * 2 * WIDTH_SCALE) / max(hi[0] - lo[0], 1e-9)
    mesh.vertices = v

    # Then cut the jaw off. Scaled to fit the skull's width, the full mesh
    # hangs past the neck and onto the shoulders; keeping the upper shell
    # turns it into a visored headset, which is what it needs to be.
    top = v[:, 1].max()
    mesh = trimesh.intersections.slice_mesh_plane(
        mesh, plane_normal=[0, 1, 0], plane_origin=[0, top - SHELL_HEIGHT, 0],
        cap=False)   # no cap needed: the head fills the opening from below

    v = np.asarray(mesh.vertices, dtype=np.float64)
    # Seat by the crown: aligning the top guarantees the shell caps the head
    # whatever proportions the generated mesh happens to have.
    v[:, 1] += (HEAD_TOP + HEAD_LIFT + CROWN_CLEARANCE) - v[:, 1].max()
    v[:, 2] += Z_CENTER
    mesh.vertices = v
    return mesh


def pack(mesh, color=COLOR):
    v = np.asarray(mesh.vertices, dtype=np.float32)
    return {"positions": np.ascontiguousarray(v),
            "normals": np.ascontiguousarray(
                np.asarray(mesh.vertex_normals, dtype=np.float32)),
            "colors": np.ascontiguousarray(
                np.tile(np.array(color, np.float32) / 255.0, (len(v), 1))),
            "faces": np.ascontiguousarray(
                np.asarray(mesh.faces, dtype=np.uint32))}


def wires(mesh, count=5, seed=8, up=1.5):
    """Cables from the helmet's crown climbing out of frame.

    Built here rather than in cables.py because these hang from a prop that
    moves: the renderer applies the helmet's transform to them too.
    """
    import cables

    v = np.asarray(mesh.vertices)
    top = v[:, 1].max()
    cx, cz = 0.0, Z_CENTER
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        a = i * 2 * np.pi / count + float(rng.uniform(-0.3, 0.3))
        r = 0.030 + float(rng.uniform(0, 0.018))
        start = np.array([cx + r * np.cos(a), top - 0.012, cz + r * np.sin(a)])
        end = np.array([cx + r * 2.4 * np.cos(a), top + up,
                        cz + r * 2.4 * np.sin(a)])
        c1 = start + np.array([r * 0.8 * np.cos(a), 0.16, r * 0.8 * np.sin(a)])
        c2 = end - np.array([0.0, up * 0.45, 0.0])
        path = cables.bezier(start, c1, c2, end, 48)
        s = np.linspace(0, 1, 48)
        rad = (0.0062 + 0.0022 * float(rng.random())) * np.clip(
            np.minimum(s * 10, (1 - s) * 40), 0.5, 1.0)
        out.append(cables.sweep(path, rad, sections=10))
    merged = trimesh.util.concatenate(out)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rot-x", type=float, default=0.0)
    ap.add_argument("--rot-y", type=float, default=0.0)
    ap.add_argument("--rot-z", type=float, default=0.0)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    mesh = load(rot_x=args.rot_x, rot_y=args.rot_y, rot_z=args.rot_z)
    v = np.asarray(mesh.vertices)
    print(f"[helmet] {len(mesh.faces)} faces after fitting")
    print("  x %.3f..%.3f   y %.3f..%.3f   z %.3f..%.3f"
          % (v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max(),
             v[:, 2].min(), v[:, 2].max()))
    print(f"  head is x +-{HEAD_HALF_W:.3f}, crown y {HEAD_TOP + HEAD_LIFT:.3f}, "
          f"eyes y {EYE_Y:.3f}")
    if args.check:
        w = wires(mesh)
        print(f"  wires: {len(w.faces)} tris")


if __name__ == "__main__":
    main()
