"""Daily Prompt — prop generator panel for Blender.

Type a prompt in the sidebar, get a mesh in the scene. Generation runs as a
background process and is polled from a modal timer, so Blender stays usable
for the ~5 minutes TRELLIS takes instead of locking up.

Install: Edit > Preferences > Add-ons > Install..., pick this file, enable it.
The panel appears in the 3D viewport sidebar (press N) under "Daily Prompt".
"""

import os
import subprocess
import tempfile
import time

import bpy

bl_info = {
    "name": "Daily Prompt — Prop Generator",
    "author": "The Daily Prompt",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar (N) > Daily Prompt",
    "description": "Generate props from a text prompt via TRELLIS and import them",
    "category": "Add Mesh",
}

RUNNER = "/home/mafuu/Documents/GitHub/ai-newsletter/video/blender/run_prop_gen.sh"


class TDPProps(bpy.types.PropertyGroup):
    prompt: bpy.props.StringProperty(
        name="Prompt",
        description="What to generate. One object works far better than a scene",
        default="a rack-mounted server unit with cables",
    )
    target_faces: bpy.props.IntProperty(
        name="Target faces",
        description="TRELLIS returns ~1M faces. Decimate on import to keep the "
                    "viewport usable; 0 imports the raw mesh",
        default=40000, min=0, max=2000000,
    )
    shade_smooth: bpy.props.BoolProperty(name="Shade smooth", default=True)
    status: bpy.props.StringProperty(default="")


class TDP_OT_generate(bpy.types.Operator):
    bl_idname = "tdp.generate_prop"
    bl_label = "Generate Prop"
    bl_description = "Generate a 3D prop from the prompt and import it"

    _proc = None
    _timer = None
    _out_dir = ""
    _t0 = 0.0

    def execute(self, context):
        props = context.scene.tdp_props
        if not props.prompt.strip():
            self.report({'ERROR'}, "Enter a prompt first")
            return {'CANCELLED'}
        if not os.path.exists(RUNNER):
            self.report({'ERROR'}, f"Runner not found: {RUNNER}")
            return {'CANCELLED'}

        self._out_dir = tempfile.mkdtemp(prefix="tdp_prop_")
        self._t0 = time.time()

        # Blender exports PYTHONHOME/PYTHONPATH/LD_LIBRARY_PATH for its own
        # bundled interpreter. A subprocess inherits them and conda's python
        # then loads Blender's stdlib and dies. Strip them.
        env = {k: v for k, v in os.environ.items()
               if k not in ("PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE",
                            "LD_LIBRARY_PATH", "LD_PRELOAD")}

        # Log to a file, not a pipe: TRELLIS prints progress for ~5 minutes and
        # a full pipe buffer would deadlock the process we are polling.
        self._log_path = os.path.join(self._out_dir, "run.log")
        self._log = open(self._log_path, "w")
        self._proc = subprocess.Popen(
            [RUNNER, "--prompt", props.prompt, "--out-dir", self._out_dir],
            stdout=self._log, stderr=subprocess.STDOUT, text=True, env=env)

        props.status = "generating…"
        self._timer = context.window_manager.event_timer_add(1.0, window=context.window)
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, "Generating — this takes about 5 minutes")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        props = context.scene.tdp_props
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        if self._proc.poll() is None:
            props.status = f"generating… {int(time.time() - self._t0)}s"
            # Nudge the sidebar so the elapsed time actually updates on screen.
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
            return {'PASS_THROUGH'}

        self._log.close()
        output = open(self._log_path).read()
        self._cleanup(context)

        if self._proc.returncode != 0:
            props.status = "failed — see log"
            tail = "\n".join(output.strip().splitlines()[-3:])
            print(f"[tdp] generation failed, full log: {self._log_path}\n{output}")
            self.report({'ERROR'}, f"Failed: {tail[:180]} (log: {self._log_path})")
            return {'CANCELLED'}

        glb = ""
        for line in output.splitlines():
            if line.startswith("GLB_PATH="):
                glb = line.split("=", 1)[1].strip()
        if not glb or not os.path.exists(glb):
            props.status = "no mesh returned"
            self.report({'ERROR'}, "Generator produced no .glb")
            return {'CANCELLED'}

        self._import(context, glb)
        props.status = f"done in {int(time.time() - self._t0)}s"
        return {'FINISHED'}

    def _import(self, context, glb):
        props = context.scene.tdp_props
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=glb)
        new = [o for o in bpy.data.objects if o not in before and o.type == 'MESH']

        for obj in new:
            obj.location = context.scene.cursor.location
            if props.target_faces and len(obj.data.polygons) > props.target_faces:
                mod = obj.modifiers.new("TDP Decimate", 'DECIMATE')
                mod.ratio = max(0.001, props.target_faces / len(obj.data.polygons))
            if props.shade_smooth:
                for poly in obj.data.polygons:
                    poly.use_smooth = True
        if new:
            context.view_layer.objects.active = new[0]

    def _cleanup(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

    def cancel(self, context):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self._cleanup(context)
        context.scene.tdp_props.status = "cancelled"


class TDP_PT_panel(bpy.types.Panel):
    bl_label = "Daily Prompt"
    bl_idname = "TDP_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Daily Prompt"

    def draw(self, context):
        layout = self.layout
        props = context.scene.tdp_props
        layout.prop(props, "prompt")
        col = layout.column(align=True)
        col.prop(props, "target_faces")
        col.prop(props, "shade_smooth")
        layout.separator()
        layout.operator("tdp.generate_prop", icon='SHADERFX')
        if props.status:
            layout.label(text=props.status)
        box = layout.box()
        box.scale_y = 0.7
        box.label(text="One object beats a scene.", icon='INFO')
        box.label(text="Imports at the 3D cursor.")


CLASSES = (TDPProps, TDP_OT_generate, TDP_PT_panel)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.tdp_props = bpy.props.PointerProperty(type=TDPProps)


def unregister():
    del bpy.types.Scene.tdp_props
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
