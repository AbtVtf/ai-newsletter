"""Cut actor inserts into a finished narrator short.

Video-only overlay at given windows; the mixed audio track of the base short
runs untouched underneath, so narration, music and sfx stay continuous across
the cuts.

  .venv/bin/python video/splice_inserts.py --base short.mp4 \
      --insert "pound.mp4@2.1-6.8" --insert "carry.mp4@7.0-10.2" \
      --out v2.mp4
"""

import argparse
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="narrator short with audio")
    ap.add_argument("--insert", action="append", default=[],
                    help="'clip.mp4@T0-T1' in base-video seconds; repeatable")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not args.insert:
        sys.exit("pass at least one --insert clip.mp4@T0-T1")

    inputs = ["-i", args.base]
    chains, prev = [], "0:v"
    for n, spec in enumerate(args.insert):
        path, window = spec.rsplit("@", 1)
        t0, t1 = (float(x) for x in window.split("-"))
        inputs += ["-i", path]
        # Shift the insert onto the base clock, then let it own the frame for
        # its window. eof_action=pass hands the frame back if it runs short.
        chains.append(f"[{n + 1}:v]setpts=PTS+{t0:.3f}/TB[i{n}]")
        chains.append(f"[{prev}][i{n}]overlay=enable='between(t,{t0:.3f},"
                      f"{t1:.3f})':eof_action=pass[o{n}]")
        prev = f"o{n}"

    cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs,
           "-filter_complex", ";".join(chains),
           "-map", f"[{prev}]", "-map", "0:a",
           "-c:v", "libx264", "-preset", "medium", "-crf", "22",
           "-maxrate", "8M", "-bufsize", "16M",
           "-pix_fmt", "yuv420p", "-c:a", "copy",
           "-movflags", "+faststart", args.out]
    if subprocess.run(cmd).returncode:
        sys.exit("ffmpeg failed")
    print(f"[splice] {len(args.insert)} inserts -> {args.out}")


if __name__ == "__main__":
    main()
