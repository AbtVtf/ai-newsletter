"""Export the GNM head with the Daily Prompt green-phosphor palette.

Same packed format as jarvis-agent/tools/export_head.py, but written into
this repo (video/assets/) so the Jarvis face stays blue and ours stays
green. Reuses the viseme morph targets Jarvis already solved for.

Run with the jarvis-agent venv (it has gnm + numpy):
  ../jarvis-agent/.venv/bin/python video/export_head.py
"""

import json
import os

import numpy as np
from gnm.shape import gnm_numpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JARVIS = os.path.join(os.path.dirname(ROOT), "jarvis-agent")
ASSETS = os.path.join(ROOT, "video", "assets")

VISEME_ORDER = ["sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn",
                "RR", "aa", "E", "ih", "oh", "ou", "blink",
                "eyes_up", "eyes_side", "pupil_wide"]

# Green CRT phosphor, matching the newsletter's dark theme. Kept desaturated
# on the skin so the lighting rig does the glowing, not the albedo.
PALETTE = {
    "SKIN": [118, 196, 132], "LIP": [92, 170, 110],
    "TEETH": [214, 248, 220], "GUMS": [70, 140, 92],
    "TONGUE": [96, 168, 112], "SCLERA": [206, 250, 214],
    "IRIS": [92, 255, 150], "PUPIL": [10, 30, 18],
    "MOUTH_SOCK": [20, 60, 34],
}


def vertex_colors(gnm):
    colors = np.tile(np.array(PALETTE["SKIN"]), (gnm.num_vertices, 1)).astype(np.float64)

    def paint(group_names, key):
        color = np.array(PALETTE[key], dtype=np.float64)
        names = group_names if isinstance(group_names, list) else [group_names]
        for name in names:
            w = np.clip(np.array(gnm.vertex_group(name)), 0.0, 1.0)[:, None]
            colors[:] = colors * (1 - w) + color[None, :] * w

    paint("mouth_sock", "MOUTH_SOCK")
    paint(["upper_lip", "lower_lip"], "LIP")
    paint("gums", "GUMS")
    paint("teeth", "TEETH")
    paint("tongue", "TONGUE")
    paint("scleras", "SCLERA")
    paint("eye_sockets", "SKIN")
    paint("irises", "IRIS")
    paint("pupils", "PUPIL")
    return np.clip(colors, 0, 255).astype(np.uint8)


def main():
    gnm = gnm_numpy.GNM.from_local(
        version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
    mirror = np.array(gnm.mirror_indices)
    sign = np.array([-1.0, 1.0, 1.0])

    tpl = np.array(gnm.template_vertex_positions, dtype=np.float64)
    tpl = (0.5 * (tpl + tpl[mirror] * sign)).astype(np.float32)

    # Drop the transparent cornea layer, as the Jarvis exporter does.
    tris = np.array(gnm.triangles, dtype=np.uint32)
    cornea = set(gnm.triangle_indices_for_group("eye_exteriors").tolist())
    tris = np.array([t for i, t in enumerate(tris) if i not in cornea], dtype=np.uint32)
    colors = vertex_colors(gnm)

    data = np.load(os.path.join(JARVIS, "build", "visemes.npz"))
    morphs = [data[f"delta_{name}"].astype(np.float32) for name in VISEME_ORDER]

    os.makedirs(ASSETS, exist_ok=True)
    manifest = {"numVertices": int(gnm.num_vertices),
                "numTriangles": int(tris.shape[0]),
                "morphNames": VISEME_ORDER, "blocks": {}}
    offset, blobs = 0, []

    def add(name, arr):
        nonlocal offset
        arr = np.ascontiguousarray(arr)
        manifest["blocks"][name] = {"offset": offset, "dtype": str(arr.dtype),
                                    "shape": list(arr.shape)}
        raw = arr.tobytes()
        pad = (-len(raw)) % 4
        blobs.append(raw + b"\0" * pad)
        offset += len(raw) + pad

    add("positions", tpl)
    add("colors", colors)
    add("triangles", tris)
    for name, delta in zip(VISEME_ORDER, morphs):
        add(f"morph_{name}", delta)

    with open(os.path.join(ASSETS, "head_green.bin"), "wb") as f:
        for b in blobs:
            f.write(b)
    with open(os.path.join(ASSETS, "head_green.json"), "w") as f:
        json.dump(manifest, f)
    print(f"wrote video/assets/head_green.bin ({offset / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
