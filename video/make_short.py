"""One command: edition article -> finished vertical short.

Chains the three stages, each in the venv that has its dependencies:
  1. script_writer  (this venv)          article -> script.txt + script.json
  2. voice_track    (jarvis-agent venv)  script -> audio.wav + timeline.json
  3. render_face    (jarvis-agent venv)  timeline -> short.mp4

  .venv/bin/python -m video.make_short --date 2026-08-08 --story story-1
  .venv/bin/python -m video.make_short --date 2026-08-08 --all

Everything lands in output/<date>/shorts/<slug>/. Stages are skipped when
their output already exists, so --force is how you redo one.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JARVIS_PY = ROOT.parent / "jarvis-agent" / ".venv" / "bin" / "python"
VIDEO = Path(__file__).resolve().parent


def run(cmd):
    print("+ " + " ".join(str(c) for c in cmd))
    proc = subprocess.run([str(c) for c in cmd])
    if proc.returncode:
        sys.exit(f"stage failed: {cmd[1]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--story", help="slug like story-1")
    ap.add_argument("--all", action="store_true", help="every article in the edition")
    ap.add_argument("--force", action="store_true", help="redo stages already done")
    ap.add_argument("--min-words", type=int, default=48)
    ap.add_argument("--max-words", type=int, default=62)
    ap.add_argument("--exaggeration", type=float, default=0.35)
    ap.add_argument("--cfg", type=float, default=0.45)
    ap.add_argument("--fx-depth", type=float, default=0.0)
    ap.add_argument("--head-height", type=float, default=0.215)
    ap.add_argument("--head-y", type=float, default=0.50)
    ap.add_argument("--speed", type=float, default=1.15)
    ap.add_argument("--theme", default="phosphor")
    ap.add_argument("--music", default=str(VIDEO / "assets" / "daily-prompt-bed.wav"),
                    help='music bed under the narration; "" for none')
    ap.add_argument("--music-gain", type=float, default=-19.0)
    ap.add_argument("--music-fade-in", type=float, default=1.5)
    ap.add_argument("--grain", type=float, default=6.0)
    ap.add_argument("--glitch", type=float, default=1.0,
                    help="glitch burst frequency multiplier; 0 disables")
    ap.add_argument("--pulse", type=float, default=3.6,
                    help="headline breathing period in seconds; 0 disables")
    ap.add_argument("--fx", default="clean",
                    help="voice processing preset: clean, machine, overlord, swarm")
    ap.add_argument("--crt-intro", default="none", choices=("flash", "full", "none"),
                    help="'full' opens from black — costly on an autoplay feed")
    ap.add_argument("--cut-every", type=int, default=2,
                    help="cut to a new framing every N sentences")
    ap.add_argument("--title-card", type=float, default=0.0,
                    help="seconds of masthead-only card before the scene")
    args = ap.parse_args()

    if not args.story and not args.all:
        sys.exit("pass --story story-1 or --all")
    if not JARVIS_PY.exists():
        sys.exit(f"jarvis venv not found at {JARVIS_PY}")

    edition = ROOT / "output" / args.date
    data = json.loads((edition / "articles.json").read_text())
    slugs = [a["slug"] for a in data["articles"]] if args.all else [args.story]
    date_line = f"{data['date_display']} · No. {data['issue_no']}"

    for slug in slugs:
        work = edition / "shorts" / slug
        print(f"\n=== {slug} ===")

        if args.force or not (work / "script.txt").exists():
            run([sys.executable, "-m", "video.script_writer", "--date", args.date,
                 "--story", slug, "--min-words", args.min_words,
                 "--max-words", args.max_words])

        # Re-synthesize when the script is newer than the audio. Without this
        # a rewritten script silently keeps the old narration, and the video
        # ships saying something the script no longer says.
        timeline = work / "timeline.json"
        stale = timeline.exists() and \
            (work / "script.txt").stat().st_mtime > timeline.stat().st_mtime
        if stale:
            print("  script changed since the audio was made — re-synthesizing")
        if args.force or not timeline.exists() or stale:
            run([JARVIS_PY, VIDEO / "voice_track.py",
                 "--script", work / "script.txt", "--out-dir", work,
                 "--exaggeration", args.exaggeration, "--cfg", args.cfg,
                 "--fx-depth", args.fx_depth])

        # Voice FX run on the finished narration, not inside TTS, so trying a
        # different preset costs seconds instead of a full re-synthesis.
        narration = work / "audio.wav"
        if args.fx != "clean":
            # Always re-run: it takes seconds, and this is the stage you tune,
            # so caching it just means edits to a preset silently do nothing.
            narration = work / f"audio-{args.fx}.wav"
            run([JARVIS_PY, VIDEO / "voice_fx.py", "--in", work / "audio.wav",
                 "--preset", args.fx, "--out", narration])

        meta = json.loads((work / "script.json").read_text())
        cmd = [JARVIS_PY, VIDEO / "render_face.py", "--work-dir", work,
               "--audio", narration,
               "--headline", meta["overlay"], "--date-line", date_line,
               "--hook", meta.get("hook", ""), "--crt-intro", args.crt_intro,
               "--cut-every", args.cut_every,
               "--source", meta["source"], "--theme", args.theme,
               "--speed", args.speed, "--grain", args.grain,
               "--glitch", args.glitch, "--pulse", args.pulse,
               "--head-height", args.head_height, "--head-y", args.head_y,
               "--title-card", args.title_card,
               "--out", work / "short.mp4"]
        if args.music and Path(args.music).is_file():
            cmd += ["--music", args.music, "--music-gain", args.music_gain,
                    "--music-fade-in", args.music_fade_in]
        run(cmd)

        run([sys.executable, VIDEO / "make_thumb.py",
             "--hook", meta.get("hook", meta["overlay"]),
             "--kicker", meta.get("kicker", ""), "--date-line", date_line,
             "--source", meta["source"], "--theme", args.theme,
             "--grain", args.grain, "--out", work / "thumb.png"])

        dur = json.loads((work / "timeline.json").read_text())["duration"] / args.speed
        print(f"{slug}: {dur:.1f}s -> {work / 'short.mp4'}")
        print(f"  title: {meta['title']}")
        print(f"  hook:  {meta.get('hook', '')}")
        print(f"  files: short.mp4, thumb.png, description.txt")


if __name__ == "__main__":
    main()
