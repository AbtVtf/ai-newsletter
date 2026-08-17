"""Narration script -> audio.wav + viseme timeline.json.

Reuses the Jarvis stack: Chatterbox TTS (voice cloned from the reference
recording) and MMS_FA forced alignment for lip sync. Must run with the
jarvis-agent venv:

  ../jarvis-agent/.venv/bin/python video/voice_track.py \
      --script work/script.txt --out-dir work/

Outputs into --out-dir:
  audio.wav      24 kHz mono narration, with lead-in/out padding
  timeline.json  {"duration", "segments": [{v,t0,t1,w}], "sentences": [{text,t0,t1}]}
"""

import argparse
import json
import re
import sys
from pathlib import Path

JARVIS = Path(__file__).resolve().parent.parent.parent / "jarvis-agent"
sys.path.insert(0, str(JARVIS))

import torch  # noqa: E402
import torchaudio  # noqa: E402

from server.lipsync import LipSync  # noqa: E402
from server.tts import clean_for_tts, robotize  # noqa: E402

BOUNDARY = re.compile(r"(?<=[.!?…])[\s\"')\]]*\s+")
DEFAULT_VOICE = JARVIS / "data" / "voice" / "reference.wav"


def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    parts, rest = [], text
    while rest:
        m = BOUNDARY.search(rest)
        if not m:
            parts.append(rest.strip())
            break
        parts.append(rest[:m.start() + 1].strip())
        rest = rest[m.end():]
    return [p for p in parts if re.search(r"[a-zA-Z0-9]", p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="text file with the narration")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--voice", default=str(DEFAULT_VOICE))
    ap.add_argument("--exaggeration", type=float, default=0.35)
    ap.add_argument("--cfg", type=float, default=0.45)
    ap.add_argument("--fx-depth", type=float, default=0.0,
                    help="0 = clean voice, >0 layers the Jarvis digital FX")
    ap.add_argument("--gap", type=float, default=0.28, help="silence between sentences (s)")
    ap.add_argument("--pad", type=float, default=0.5, help="lead-in/out silence (s)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = Path(args.script).read_text()
    parts = split_sentences(text)
    if not parts:
        sys.exit("script has no speakable sentences")

    print(f"[voice] loading Chatterbox + aligner ({len(parts)} sentences)...")
    from chatterbox.tts import ChatterboxTTS
    tts = ChatterboxTTS.from_pretrained(device="cuda")
    lipsync = LipSync()
    sr = tts.sr

    kwargs = {"exaggeration": args.exaggeration, "cfg_weight": args.cfg}
    if args.voice and Path(args.voice).is_file():
        kwargs["audio_prompt_path"] = args.voice
        print(f"[voice] cloning voice from {args.voice}")

    gap = torch.zeros(1, int(args.gap * sr))
    pad = torch.zeros(1, int(args.pad * sr))
    chunks, segments, sentences = [pad], [], []
    t = args.pad
    for i, sentence in enumerate(parts):
        clean = clean_for_tts(sentence)
        wav = tts.generate(clean, **kwargs).cpu()
        if args.fx_depth > 0:
            wav = robotize(wav, sr, args.fx_depth)
        chunk_path = out_dir / f"_chunk_{i}.wav"
        torchaudio.save(str(chunk_path), wav, sr)
        for seg in lipsync.timeline(str(chunk_path), clean):
            segments.append({"v": seg["v"], "t0": round(seg["t0"] + t, 4),
                             "t1": round(seg["t1"] + t, 4), "w": seg["w"]})
        dur = wav.shape[-1] / sr
        sentences.append({"text": sentence, "t0": round(t, 3), "t1": round(t + dur, 3)})
        chunk_path.unlink()
        chunks.append(wav)
        chunks.append(gap)
        t += dur + args.gap
        print(f"[voice] {i + 1}/{len(parts)}  {dur:5.2f}s  {sentence[:60]}")

    chunks[-1] = pad  # trailing pad instead of a mid-gap
    audio = torch.cat(chunks, dim=-1)
    torchaudio.save(str(out_dir / "audio.wav"), audio, sr)
    duration = audio.shape[-1] / sr
    (out_dir / "timeline.json").write_text(json.dumps({
        "duration": round(duration, 3),
        "sample_rate": sr,
        "segments": segments,
        "sentences": sentences,
    }, indent=1))
    print(f"[voice] done: {duration:.1f}s -> {out_dir}/audio.wav + timeline.json")


if __name__ == "__main__":
    main()
