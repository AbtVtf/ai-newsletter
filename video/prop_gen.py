"""prompt -> image -> TRELLIS -> .glb, as one command.

Runs inside the `trellis2` conda env (see run_prop_gen.sh), so it can import
the pipeline directly instead of going through the MCP server. The MCP server
is stdio and owned by whichever client launched it; a Blender addon cannot
attach to it, but it can run this.

  ./video/blender/run_prop_gen.sh --prompt "a server rack" --out-dir /tmp/rack
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import requests

TRELLIS_ROOT = Path("/home/mafuu/Documents/GitHub/3d-pipeline/TRELLIS.2")
HDRI = TRELLIS_ROOT / "assets" / "hdri" / "forest.exr"
NEWSLETTER = Path(__file__).resolve().parent.parent
IMAGE_MODEL = "google/gemini-2.5-flash-image"

# TRELLIS wants one clean object, centred, on plain white. Saying so every
# time matters more than the wording of the object itself.
IMAGE_SUFFIX = (" Single object, centered, front three-quarter view, plain "
                "pure white background, even studio lighting, no shadows, no "
                "text, no watermark, whole object visible inside the frame.")


def api_key():
    for line in (NEWSLETTER / ".env").read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("OPENROUTER_API_KEY not found in ai-newsletter/.env")


def make_image(prompt, out_path):
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key()}",
                 "Content-Type": "application/json"},
        json={"model": IMAGE_MODEL,
              "messages": [{"role": "user", "content": prompt + IMAGE_SUFFIX}],
              "modalities": ["image", "text"]},
        timeout=240,
    )
    body = resp.json()
    if "choices" not in body:
        raise SystemExit(f"image generation failed: {json.dumps(body)[:400]}")
    for img in body["choices"][0]["message"].get("images") or []:
        url = img.get("image_url", {}).get("url", "")
        if url.startswith("data:"):
            out_path.write_bytes(base64.b64decode(url.split(",", 1)[1]))
            return out_path
    raise SystemExit("model returned no image")


def make_mesh(image_path, out_dir, decimation_target=200000):
    """Run TRELLIS and write a .glb.

    This mirrors mcp-trellis2/server.py exactly. run() takes only the image —
    the envmap belongs to the turntable-video path, not to generation — and
    the mesh has to go through o_voxel.postprocess.to_glb() to become a glb;
    the pipeline does not hand one back.
    """
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    sys.path.insert(0, str(TRELLIS_ROOT))
    import o_voxel
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipe.cuda()

    mesh = pipe.run(Image.open(image_path))[0]
    mesh.simplify(16777216)                       # nvdiffrast hard cap

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target, texture_size=2048,
        remesh=True, remesh_band=1, remesh_project=0, verbose=False)

    path = Path(out_dir) / "sample.glb"
    glb.export(str(path), extension_webp=True)
    print(f"[prop] {mesh.faces.shape[0]} faces before decimation", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--image-only", action="store_true")
    ap.add_argument("--image", default="",
                    help="use an existing image instead of generating one")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.image:
        img = Path(args.image)
        print(f"[prop] using existing image {img}", flush=True)
    else:
        img = out_dir / "prompt.png"
        print("[prop] generating image...", flush=True)
        make_image(args.prompt, img)
        print(f"[prop] image -> {img}", flush=True)
    if args.image_only:
        return

    print("[prop] running TRELLIS (this takes ~4-5 min)...", flush=True)
    glb = make_mesh(img, out_dir)
    print(f"[prop] done in {time.time() - t0:.0f}s -> {glb}", flush=True)
    # The addon watches for this line to know where to import from.
    print(f"GLB_PATH={glb}", flush=True)


if __name__ == "__main__":
    main()
