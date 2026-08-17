"""Rebuild the Daily Prompt set inside Blender as editable objects.

Run from Blender's Scripting tab, or:
  blender --python video/blender/build_scene.py

Everything is native Blender geometry, not an import, so it is all editable.
The head is NOT included: it is animated from morph targets in the renderer.
A placeholder cylinder marks where it lives so you can frame around it —
delete it before exporting, or leave it, since only names matter on export.

Export with File > Export > glTF 2.0 (.glb), then render with:
  render_face.py --set-glb /path/to/set.glb

Rules for the export:
  - Keep the world origin and scale. The camera is framed from the head's
    bounding box in these same units, so moving the origin moves the shot.
  - Name anything transparent with "glass" in it; those go to the blended
    pass. Everything else is drawn opaque.
"""

import random

import bpy

# Same numbers as video/set_pieces.py — change them there too if you retune.
TUBE_RADIUS = 0.163
TUBE_BOTTOM = 0.048
TUBE_TOP = 0.462
Z_CENTER = 0.0271
WALL_Z = -0.36
RACK_X = 0.33

PALETTE = {
    "glass": (0.42, 0.75, 0.52, 1.0),
    "ring": (0.59, 0.88, 0.67, 1.0),
    "base": (0.23, 0.48, 0.31, 1.0),
    "panel": (0.03, 0.07, 0.05, 1.0),
    "rack": (0.05, 0.11, 0.07, 1.0),
    "trim": (0.09, 0.19, 0.12, 1.0),
    "head": (0.46, 0.86, 0.55, 1.0),
}


def material(name, rgba):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = rgba
    # Exported vertex/base colour is what the renderer reads back.
    mat.diffuse_color = rgba
    return mat


def add(obj, name, palette_key, collection):
    obj.name = name
    obj.data.materials.append(material(palette_key, PALETTE[palette_key]))
    for c in obj.users_collection:
        c.objects.unlink(obj)
    collection.objects.link(obj)
    return obj


def by(renderer_z):
    """Renderer depth -> Blender Y.

    The glTF +Y-up exporter maps Blender (x, y, z) to (x, z, -y), so Blender's
    Y becomes the renderer's Z *negated*. Placing something at Blender
    Y = -0.36 puts it at renderer Z = +0.36 — in front of the camera instead
    of behind the head, which hides the entire scene.
    """
    return -renderer_z


def cylinder(name, key, radius, depth, z, coll, vertices=64):
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth,
                                        location=(0, by(Z_CENTER), z),
                                        vertices=vertices)
    return add(bpy.context.object, name, key, coll)


def cube(name, key, sx, sy, sz, x, y, z, coll):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, z))
    obj = bpy.context.object
    obj.scale = (sx, sy, sz)
    return add(obj, name, key, coll)


def clear_startup_cube():
    """Remove Blender's default cube.

    It is 2m across and sits at the origin, so a plain File > Export bakes a
    giant box straight into the shot — and because it surrounds the head it is
    not obvious in the viewport that anything is wrong.
    """
    obj = bpy.data.objects.get("Cube")
    if obj and obj.type == 'MESH' and len(obj.data.vertices) == 8:
        bpy.data.objects.remove(obj, do_unlink=True)
        print("[scene] removed the default startup cube")


def warn_about_strays(coll):
    strays = [o.name for o in bpy.data.objects
              if o.type == 'MESH' and coll not in o.users_collection]
    if strays:
        print(f"[scene] WARNING: {len(strays)} mesh(es) outside 'DailyPrompt': "
              f"{', '.join(strays[:6])}")
        print("[scene] they WILL be exported unless you limit the export to "
              "the DailyPrompt collection")


def main():
    # Blender is Z-up; the renderer is Y-up. Build in Blender's axes and let
    # the glTF exporter convert — its default +Y up setting does exactly this.
    scene = bpy.context.scene
    clear_startup_cube()
    coll = bpy.data.collections.get("DailyPrompt")
    if coll:
        for obj in list(coll.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
    else:
        coll = bpy.data.collections.new("DailyPrompt")
        scene.collection.children.link(coll)

    h = TUBE_TOP - TUBE_BOTTOM
    cylinder("Tube_glass", "glass", TUBE_RADIUS, h, TUBE_BOTTOM + h / 2, coll)
    cylinder("Ring_lower", "ring", 0.186, 0.016, TUBE_BOTTOM + 0.008, coll)
    cylinder("Ring_upper", "ring", 0.186, 0.016, TUBE_TOP - 0.008, coll)
    cylinder("Pedestal", "base", 0.215, 0.052, 0.022, coll)
    cylinder("Plinth", "base", 0.255, 0.022, -0.007, coll)

    cube("Wall_back", "panel", 2.10, 0.05, 1.70, 0, by(WALL_Z), 0.75, coll)
    for side, sx in (("L", -1), ("R", 1)):
        for i in range(9):
            z = 0.05 + i * 0.12
            cube(f"Rack_{side}_{i:02d}", "rack", 0.30, 0.24, 0.09,
                 sx * RACK_X, by(WALL_Z + 0.17), z, coll)
    cube("Floor", "panel", 2.10, 1.20, 0.05, 0, by(WALL_Z + 0.62), -0.025, coll)

    # Greeble grid: panels with small blocks on them. Seeded so re-running
    # gives the same wall, and so it matches what set_pieces.py renders.
    rng = random.Random(5)
    gx = -0.98
    while gx < 0.99:
        gy = 0.02
        while gy < 1.60:
            if not (abs(gx) < 0.46 and 0.05 < gy < 0.52) and rng.random() > 0.22:
                d = rng.uniform(0.012, 0.05)
                cube(f"Panel_{gx:+.2f}_{gy:.2f}", "panel", 0.26 - rng.uniform(0, 0.05),
                     0.24 - rng.uniform(0, 0.05), d,
                     gx, by(WALL_Z + 0.025 + d / 2), gy, coll)
                for k in range(rng.randint(0, 4)):
                    bd = rng.uniform(0.008, 0.028)
                    cube(f"Greeble_{gx:+.2f}_{gy:.2f}_{k}", "rack",
                         rng.uniform(0.015, 0.07), rng.uniform(0.010, 0.045), bd,
                         gx + rng.uniform(-0.09, 0.09),
                         by(WALL_Z + 0.025 + d + bd / 2),
                         gy + rng.uniform(-0.08, 0.08), coll)
            gy += 0.26
        gx += 0.28

    # Cables and bubbles are generated, not modelled — import
    # video/assets/props/cables.glb if you want them visible while you work.
    print("[scene] cables: File > Import > glTF 2.0 > video/assets/props/cables.glb")

    # Placeholder only — the real head is animated in the renderer.
    ph = cylinder("HEAD_PLACEHOLDER_delete_me", "head", 0.10, 0.34, 0.24, coll)
    ph.display_type = 'WIRE'

    warn_about_strays(coll)
    print(f"[scene] built {len(coll.objects)} objects in collection 'DailyPrompt'")
    print("[scene] export: File > Export > glTF 2.0, +Y up, and set")
    print("[scene]         Include > Limit to > Visible Objects (or select the")
    print("[scene]         DailyPrompt collection) so nothing else sneaks in")
    print("[scene] then:   render_face.py --set-glb <file>.glb")


if __name__ == "__main__":
    main()
