"""Procedural set for the anchor: a containment tube on a pedestal.

Built from primitives rather than generated, because this is hard-edged
hardware. A generated mesh gives you an unpredictable organic blob; a
cylinder gives you exactly the cylinder you asked for, at the exact radius
the head needs, in the exact palette.

Everything is sized from the head's own bounding box, so it still fits if
the head is ever re-exported at a different scale.

Returns flat float32 arrays ready for the renderer: positions, normals,
colors, and an index buffer.
"""

import math

import numpy as np
import trimesh

# Head bbox (world units): x +-0.128, y 0.066..0.407, z -0.092..0.147.
# The shoulders are the widest part, so they set the tube radius.
TUBE_RADIUS = 0.163
TUBE_BOTTOM = 0.048
TUBE_TOP = 0.462
RING_OUTER = 0.186
RING_HEIGHT = 0.016
BASE_RADIUS = 0.215
BASE_TOP = 0.048
BASE_HEIGHT = 0.052
PLINTH_RADIUS = 0.255
PLINTH_HEIGHT = 0.022
SECTIONS = 96

COLORS = {
    # Desaturated next to the head so the face stays the brightest thing.
    "glass": (108, 190, 132),
    "ring": (150, 224, 172),
    "base": (58, 122, 78),
    "plinth": (44, 96, 62),
    # The wall is deliberately far darker than everything above. It gives
    # depth and context; it must not compete with the face. These albedos look
    # almost black on their own — the key light is aimed at the head and a big
    # flat panel facing it washes out fast, so they have to start very low.
    "panel": (8, 18, 12),
    "rack": (13, 29, 18),
    "grille": (5, 13, 8),
    "trim": (22, 48, 31),
    "conduit": (16, 36, 23),
}

WALL_Z = -0.36          # back wall plane
WALL_W, WALL_H = 2.10, 1.70
RACK_X = 0.33           # rack columns sit just outside the tube
FLOOR_Y = -0.002


def _to_y_axis(mesh, y_center, z_center):
    """trimesh builds cylinders along Z; stand them up and place them."""
    mesh.apply_transform(trimesh.transformations.rotation_matrix(
        np.pi / 2, [1, 0, 0]))
    mesh.apply_translation([0.0, y_center, z_center])
    return mesh


def _cylinder(radius, height, y_center, z_center, sections=SECTIONS):
    return _to_y_axis(trimesh.creation.cylinder(
        radius=radius, height=height, sections=sections), y_center, z_center)


def _annulus(r_min, r_max, height, y_center, z_center, sections=SECTIONS):
    """Ring lying flat, threaded onto the vertical tube."""
    return _to_y_axis(trimesh.creation.annulus(
        r_min=r_min, r_max=r_max, height=height, sections=sections),
        y_center, z_center)


def _annulus_facing(r_min, r_max, thickness, y_center, z_center, sections=SECTIONS):
    """Ring standing upright, facing the camera. trimesh builds annuli in the
    XY plane already, so this one must NOT be rotated — rotating it lays it
    flat and it reads as a shelf across the frame instead of a wall opening."""
    m = trimesh.creation.annulus(r_min=r_min, r_max=r_max, height=thickness,
                                 sections=sections)
    m.apply_translation([0.0, y_center, z_center])
    return m


def _pack(meshes_and_colors):
    positions, normals, colors, faces = [], [], [], []
    offset = 0
    for mesh, rgb in meshes_and_colors:
        v = np.asarray(mesh.vertices, dtype=np.float32)
        # Smooth normals on a cylinder wall look like a curved surface; the
        # face normals trimesh gives per-vertex are already what we want here.
        n = np.asarray(mesh.vertex_normals, dtype=np.float32)
        f = np.asarray(mesh.faces, dtype=np.uint32) + offset
        positions.append(v)
        normals.append(n)
        colors.append(np.tile(np.array(rgb, dtype=np.float32) / 255.0, (len(v), 1)))
        faces.append(f)
        offset += len(v)
    return {
        "positions": np.ascontiguousarray(np.concatenate(positions)),
        "normals": np.ascontiguousarray(np.concatenate(normals)),
        "colors": np.ascontiguousarray(np.concatenate(colors).astype(np.float32)),
        "faces": np.ascontiguousarray(np.concatenate(faces)),
    }


def _box(ex, ey, ez, x, y, z):
    m = trimesh.creation.box(extents=[ex, ey, ez])
    m.apply_translation([x, y, z])
    return m


def build_wall(z_center=0.0271, seed=5):
    """The tube sits in a wall of machinery: back panel with a recessed
    alcove, rack columns either side, grille slats, conduits and a floor.

    Repeated modular boxes on a grid — which is what a rack actually is, and
    what procedural geometry does well. A generated mesh could not tile, could
    not leave a hole for the tube, and could not carry blinking lights.
    """
    rng = np.random.default_rng(seed)
    parts = []

    parts.append((_box(WALL_W, WALL_H, 0.05, 0, WALL_H / 2 - 0.10, WALL_Z),
                  COLORS["panel"]))
    # Recessed collar so the tube reads as set INTO the wall, not in front of
    # it. Inner radius clears the tube; outer stays tight so it frames the
    # opening rather than becoming a slab across the shot.
    parts.append((_annulus_facing(0.185, 0.30, 0.035, 0.255, WALL_Z + 0.04),
                  COLORS["trim"]))

    # Rack columns: stacked modules of varying height, with gaps between.
    for sx in (-1, 1):
        y = FLOOR_Y
        while y < 1.05:
            h = float(rng.uniform(0.045, 0.115))
            depth = float(rng.uniform(0.16, 0.30))
            parts.append((_box(0.30, h, depth, sx * RACK_X, y + h / 2,
                               WALL_Z + 0.05 + depth / 2), COLORS["rack"]))
            # Every few modules gets a vent grille of thin slats.
            if rng.random() < 0.45:
                for k in range(3):
                    parts.append((_box(0.24, 0.006, 0.012, sx * RACK_X,
                                       y + h * (0.3 + 0.2 * k),
                                       WALL_Z + 0.05 + depth + 0.004),
                                  COLORS["grille"]))
            y += h + float(rng.uniform(0.012, 0.03))

    # Vertical conduits running up the wall behind the racks.
    for x in (-0.62, -0.52, 0.52, 0.62):
        parts.append((_cylinder(0.016, 1.4, 0.65, WALL_Z + 0.06,
                                sections=16), COLORS["conduit"]))
        parts[-1][0].apply_translation([x, 0, 0])

    # ---- greebling ----
    # Panel grid, then clusters of small boxes on top. Repetition at one scale
    # with irregular detail at a smaller one is what makes a surface read as
    # machinery rather than as a flat wall.
    for gx in np.arange(-0.98, 0.99, 0.28):
        for gy in np.arange(0.02, 1.60, 0.26):
            if abs(gx) < 0.46 and 0.05 < gy < 0.52:
                continue                    # keep the tube opening clear
            if rng.random() < 0.22:
                continue                    # gaps stop it looking tiled
            w = 0.26 - float(rng.uniform(0.0, 0.05))
            h = 0.24 - float(rng.uniform(0.0, 0.05))
            d = float(rng.uniform(0.012, 0.05))
            parts.append((_box(w, h, d, float(gx), float(gy), WALL_Z + 0.025 + d / 2),
                          COLORS["panel"]))
            # Greebles: a few small blocks scattered on the panel face.
            for _ in range(int(rng.integers(0, 5))):
                bw = float(rng.uniform(0.015, 0.07))
                bh = float(rng.uniform(0.010, 0.045))
                bd = float(rng.uniform(0.008, 0.028))
                parts.append((_box(bw, bh, bd,
                                   float(gx) + float(rng.uniform(-0.09, 0.09)),
                                   float(gy) + float(rng.uniform(-0.08, 0.08)),
                                   WALL_Z + 0.025 + d + bd / 2), COLORS["rack"]))
            # Occasional screen: brighter, so a few points glow in the dark.
            if rng.random() < 0.16:
                parts.append((_box(float(rng.uniform(0.09, 0.17)),
                                   float(rng.uniform(0.06, 0.11)), 0.006,
                                   float(gx), float(gy),
                                   WALL_Z + 0.025 + d + 0.004), COLORS["trim"]))

    # Sensor dishes: shallow cylinders on stubby mounts.
    for x, y, r in ((-0.86, 1.18, 0.085), (0.88, 1.02, 0.070),
                    (-0.74, 0.20, 0.055), (0.80, 0.38, 0.062)):
        dish = _cylinder(r, 0.022, y, WALL_Z + 0.10, sections=32)
        dish.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi / 2, [1, 0, 0], [0, y, WALL_Z + 0.10]))
        dish.apply_translation([x, 0, 0])
        parts.append((dish, COLORS["trim"]))
        parts.append((_box(0.03, 0.03, 0.07, x, y, WALL_Z + 0.055),
                      COLORS["conduit"]))

    # Pipe runs crossing the wall, breaking up the grid.
    for y in (0.63, 1.34):
        pipe = trimesh.creation.cylinder(radius=0.021, height=1.95, sections=16)
        pipe.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi / 2, [0, 1, 0]))
        pipe.apply_translation([0.0, y, WALL_Z + 0.075])
        parts.append((pipe, COLORS["conduit"]))
        for x in (-0.70, -0.20, 0.30, 0.78):
            parts.append((_box(0.05, 0.055, 0.05, x, y, WALL_Z + 0.075),
                          COLORS["trim"]))

    # Floor slab under the whole set.
    parts.append((_box(WALL_W, 0.05, 1.20, 0, FLOOR_Y - 0.025, WALL_Z + 0.62),
                  COLORS["panel"]))
    parts.append((_box(WALL_W, 0.008, 0.02, 0, FLOOR_Y + 0.004, WALL_Z + 1.21),
                  COLORS["trim"]))
    return _pack(parts)


def screen_texture(width=384, height=1536, seed=21):
    """A tall strip of fake terminal output, sampled and scrolled by the
    screens. Tall so it wraps seamlessly, and generated so it tiles without
    anyone having to author it.
    """
    from PIL import Image, ImageDraw

    from design import font as design_font

    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (width, height), (4, 10, 6))
    d = ImageDraw.Draw(img)
    fnt = design_font("vt323", 22)
    words = ["SYS", "LINK", "SCAN", "CORE", "NODE", "BUS", "MEM", "IO",
             "SYNC", "TRACE", "POOL", "TASK", "NET", "DISK", "PIPE"]
    tags = ["OK", "RUN", "WAIT", "DONE", "IDLE", "BUSY", "PASS"]

    y = 4
    while y < height:
        roll = rng.random()
        if roll < 0.42:
            line = (f"{rng.integers(0, 0xFFFF):04X}  "
                    + " ".join(f"{v:02X}" for v in rng.integers(0, 256, 6)))
        elif roll < 0.72:
            line = (f"{words[rng.integers(len(words))]}."
                    f"{words[rng.integers(len(words))].lower()} "
                    f"{tags[rng.integers(len(tags))]}")
        elif roll < 0.86:
            line = f"[{rng.integers(0, 100):02d}%] {'#' * int(rng.integers(1, 14))}"
        else:
            line = ""
        # A few lines burn brighter, like fresh output.
        shade = 235 if rng.random() < 0.18 else int(rng.integers(90, 165))
        d.text((8, y), line, font=fnt, fill=(int(shade * 0.5), shade,
                                             int(shade * 0.62)))
        y += 24
    return img


def build_screens(seed=21):
    """Emissive terminal panels on the wall.

    These solve the flat-wall problem as much as they decorate it: a screen
    makes its own light, so unlike the greebles it does not depend on spill
    from the key light aimed at the head.
    """
    rng = np.random.default_rng(seed)
    # (x, y, w, h). Two constraints fix the placement: the tube and its cap
    # occupy |x| < 0.19, and at the widest framing only about |x| < 0.55 of
    # the wall is on screen at all — anything further out is never seen.
    specs = [(-0.37, 1.00, 0.32, 0.25), (0.39, 1.10, 0.29, 0.22),
             (-0.41, 0.56, 0.27, 0.32), (0.40, 0.64, 0.29, 0.26),
             (-0.34, 0.14, 0.26, 0.21), (0.36, 0.20, 0.28, 0.22),
             (-0.28, 1.34, 0.30, 0.20), (0.31, 1.38, 0.27, 0.19),
             (-0.48, 0.88, 0.20, 0.16), (0.49, 0.38, 0.19, 0.16)]
    # In front of every greeble. Panels sit as far forward as WALL_Z + 0.075,
    # so anything shallower than that gets covered by the wall detail.
    z = WALL_Z + 0.10
    verts, uvs, scrolls, faces = [], [], [], []
    for i, (x, y, w, h) in enumerate(specs):
        hw, hh = w / 2, h / 2
        verts.append(np.array([[x - hw, y - hh, z], [x + hw, y - hh, z],
                               [x + hw, y + hh, z], [x - hw, y + hh, z]],
                              dtype=np.float32))
        # V spans a slice of the tall strip, so each screen shows its own text.
        v0 = float(rng.uniform(0, 1))
        uvs.append(np.array([[0, v0 + h * 1.6], [1, v0 + h * 1.6],
                             [1, v0], [0, v0]], dtype=np.float32))
        scrolls.append(np.tile(np.array(
            [rng.uniform(0.02, 0.09), rng.uniform(0, 1)], np.float32), (4, 1)))
        b = i * 4
        faces.append(np.array([[b, b + 1, b + 2], [b, b + 2, b + 3]],
                              dtype=np.uint32))
    return {"positions": np.ascontiguousarray(np.concatenate(verts)),
            "uvs": np.ascontiguousarray(np.concatenate(uvs)),
            "scroll": np.ascontiguousarray(np.concatenate(scrolls)),
            "faces": np.ascontiguousarray(np.concatenate(faces))}


def build_lights(seed=9):
    """Indicator LEDs as camera-facing quads, each with its own blink phase.

    Returned with a per-vertex phase attribute; the renderer animates them in
    a small unlit shader, so the wall has motion without anything moving.
    """
    rng = np.random.default_rng(seed)
    verts, cols, phases, faces = [], [], [], []
    n = 0
    for sx in (-1, 1):
        y = FLOOR_Y + 0.05
        while y < 1.0:
            if rng.random() < 0.62:
                for k in range(int(rng.integers(2, 6))):
                    x = sx * RACK_X + (-0.11 + 0.05 * k)
                    z = WALL_Z + 0.05 + float(rng.uniform(0.17, 0.29))
                    w, h = 0.009, 0.006
                    quad = np.array([[x - w, y - h, z], [x + w, y - h, z],
                                     [x + w, y + h, z], [x - w, y + h, z]],
                                    dtype=np.float32)
                    verts.append(quad)
                    bright = COLORS["ring"] if rng.random() < 0.75 else (255, 214, 120)
                    cols.append(np.tile(np.array(bright, np.float32) / 255.0, (4, 1)))
                    phases.append(np.full(4, rng.random() * 6.283, dtype=np.float32))
                    faces.append(np.array([[n, n + 1, n + 2], [n, n + 2, n + 3]],
                                          dtype=np.uint32))
                    n += 4
            y += float(rng.uniform(0.055, 0.13))
    return {
        "positions": np.ascontiguousarray(np.concatenate(verts)),
        "colors": np.ascontiguousarray(np.concatenate(cols).astype(np.float32)),
        "phases": np.ascontiguousarray(np.concatenate(phases)),
        "faces": np.ascontiguousarray(np.concatenate(faces)),
    }


def load_glb(path, glass_keyword="glass"):
    """Load a set built in Blender, replacing the procedural one.

    This is the round trip that makes editing in Blender worth doing: Blender
    is the scene editor, .glb is the interchange, the renderer consumes it.

    Convention: any object whose name contains "glass" is drawn in the
    transparent pass. Leave the head out of the file — it is animated here,
    not there — and keep the world origin and scale, since the camera is
    framed from the head's own bounding box.
    """
    scene = trimesh.load(path, process=False)
    groups = {"opaque": [], "glass": []}
    geoms = ([(n, scene.geometry[scene.graph[n][1]].copy(), scene.graph[n][0])
              for n in scene.graph.nodes_geometry]
             if isinstance(scene, trimesh.Scene) else [("mesh", scene, None)])

    for name, mesh, transform in geoms:
        if transform is not None:
            mesh.apply_transform(transform)
        try:
            rgb = np.asarray(mesh.visual.to_color().vertex_colors)[:, :3]
        except Exception:
            rgb = np.tile(np.array(COLORS["rack"], dtype=np.uint8), (len(mesh.vertices), 1))
        key = "glass" if glass_keyword in name.lower() else "opaque"
        groups[key].append((mesh, rgb))

    def pack(items):
        if not items:
            return {"positions": np.zeros((0, 3), np.float32),
                    "normals": np.zeros((0, 3), np.float32),
                    "colors": np.zeros((0, 3), np.float32),
                    "faces": np.zeros((0, 3), np.uint32)}
        pos, nrm, col, fac, off = [], [], [], [], 0
        for mesh, rgb in items:
            v = np.asarray(mesh.vertices, dtype=np.float32)
            pos.append(v)
            nrm.append(np.asarray(mesh.vertex_normals, dtype=np.float32))
            col.append((np.asarray(rgb, dtype=np.float32) / 255.0))
            fac.append(np.asarray(mesh.faces, dtype=np.uint32) + off)
            off += len(v)
        return {"positions": np.ascontiguousarray(np.concatenate(pos)),
                "normals": np.ascontiguousarray(np.concatenate(nrm)),
                "colors": np.ascontiguousarray(np.concatenate(col).astype(np.float32)),
                "faces": np.ascontiguousarray(np.concatenate(fac))}

    return pack(groups["opaque"]), pack(groups["glass"])


CAP_TOP = 0.566          # top of the cap's fitting, where the gripper lands


def build_arm(z_center=0.0271):
    """Servicing arm: shaft, housing, pistons and a three-finger gripper.

    Modelled with the gripper mouth at the local origin and the shaft running
    up +Y, so the renderer positions it by dropping that origin onto the cap
    and never has to reason about the arm's own length.
    """
    parts = []
    parts.append((_cylinder(0.030, 1.60, 0.80, 0.0, sections=24), COLORS["rack"]))
    parts.append((_cylinder(0.052, 0.075, 0.130, 0.0, sections=24), COLORS["trim"]))
    parts.append((_annulus(0.052, 0.070, 0.014, 0.176, 0.0, sections=32),
                  COLORS["ring"]))
    # Pistons flanking the shaft — the detail that reads as machinery.
    for i in range(3):
        a = i * 2 * math.pi / 3
        rod = _cylinder(0.009, 0.34, 0.30, 0.0, sections=12)
        rod.apply_translation([0.046 * math.cos(a), 0.0, 0.046 * math.sin(a)])
        parts.append((rod, COLORS["conduit"]))
    # Three fingers reaching down and inward.
    for i in range(3):
        a = i * 2 * math.pi / 3
        finger = _box(0.020, 0.088, 0.030,
                      0.056 * math.cos(a), 0.048, 0.056 * math.sin(a))
        parts.append((finger, COLORS["ring"]))
        tip = _box(0.026, 0.024, 0.034,
                   0.048 * math.cos(a), 0.010, 0.048 * math.sin(a))
        parts.append((tip, COLORS["trim"]))
    packed = _pack(parts)
    packed["positions"][:, 2] += z_center
    return packed


def build_water(z_center=0.0271, sections=64):
    """A disc for the fluid surface, drawn in the transparent pass and moved
    down as the tube drains. Without it the level is invisible — bubbles just
    stop appearing, which reads as a bug rather than as draining."""
    disc = trimesh.creation.cylinder(radius=TUBE_RADIUS - 0.004, height=0.004,
                                     sections=sections)
    disc.apply_transform(trimesh.transformations.rotation_matrix(
        np.pi / 2, [1, 0, 0]))
    disc.apply_translation([0.0, 0.0, z_center])
    return _pack([(disc, COLORS["glass"])])


def build_groups(z_center=0.0271):
    """The set split by what moves: the wall is bolted to the world, the rig
    and glass ride the opening spin, and the cap later leaves on its own."""
    opaque, glass = build(z_center)
    return {"wall": build_wall(z_center), "rig": opaque, "cap": build_cap(z_center),
            "glass": glass, "arm": build_arm(z_center), "water": build_water(z_center)}


def build_cap(z_center=0.0271):
    """The cap alone, so it can be unscrewed and lifted away."""
    cap_lo = TUBE_TOP - 0.012
    parts = [(_cylinder(RING_OUTER, 0.030, cap_lo + 0.015, z_center), COLORS["ring"]),
             (_cylinder(0.140, 0.046, cap_lo + 0.053, z_center), COLORS["base"]),
             (_cylinder(0.062, 0.040, cap_lo + 0.096, z_center), COLORS["plinth"]),
             (_annulus(0.062, 0.104, 0.014, cap_lo + 0.083, z_center), COLORS["ring"])]
    for i in range(10):
        a = i * 2 * math.pi / 10
        b = _cylinder(0.010, 0.016, cap_lo + 0.036, z_center, sections=12)
        b.apply_translation([0.166 * math.cos(a), 0.0, 0.166 * math.sin(a)])
        parts.append((b, COLORS["ring"]))
    return _pack(parts)


def build(z_center=0.0271):
    """Opaque set dressing and the transparent tube, packed separately so the
    renderer can draw the glass last with blending."""
    ring_lo = _annulus(TUBE_RADIUS - 0.004, RING_OUTER, RING_HEIGHT,
                       TUBE_BOTTOM + RING_HEIGHT / 2, z_center)
    ring_hi = _annulus(TUBE_RADIUS - 0.004, RING_OUTER, RING_HEIGHT,
                       TUBE_TOP - RING_HEIGHT / 2, z_center)
    base = _cylinder(BASE_RADIUS, BASE_HEIGHT, BASE_TOP - BASE_HEIGHT / 2, z_center)
    plinth = _cylinder(PLINTH_RADIUS, PLINTH_HEIGHT,
                       BASE_TOP - BASE_HEIGHT - PLINTH_HEIGHT / 2, z_center)

    # Rig only. The cap is build_cap() and the wall is build_wall(), because
    # they move independently; the cables have their own animated program.
    opaque = _pack([(plinth, COLORS["plinth"]), (base, COLORS["base"]),
                    (ring_lo, COLORS["ring"]), (ring_hi, COLORS["ring"])])

    # Open-ended wall only: caps would fog the face behind them.
    wall = trimesh.creation.cylinder(
        radius=TUBE_RADIUS, height=TUBE_TOP - TUBE_BOTTOM, sections=SECTIONS)
    wall = _to_y_axis(wall, (TUBE_TOP + TUBE_BOTTOM) / 2, z_center)
    keep = np.abs(wall.face_normals[:, 1]) < 0.5
    wall.update_faces(keep)
    wall.remove_unreferenced_vertices()
    glass = _pack([(wall, COLORS["glass"])])
    return opaque, glass


if __name__ == "__main__":
    o, g = build()
    print(f"opaque: {len(o['positions'])} verts, {len(o['faces'])} tris")
    print(f"glass:  {len(g['positions'])} verts, {len(g['faces'])} tris")
