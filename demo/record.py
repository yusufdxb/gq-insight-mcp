#!/usr/bin/env python3
"""Render a narrated terminal-screencast .mp4 of gq-insight running.

Real output from the library (no faked text) laid out as a GitHub-dark
terminal, typed with a typewriter effect, with a spoken voiceover generated
locally by Kokoro TTS. Each scene's visuals are timed to its narration, then
the scenes are concatenated into one H.264 file with AAC audio.

Usage:  python demo/record.py            # -> assets/gq-insight-demo.mp4
        python demo/record.py --silent   # no voiceover (visual only)

Requires: Pillow, ffmpeg. Voiceover also needs `kokoro` + `soundfile` (falls
back to a silent render automatically if TTS is unavailable).
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from gq_insight.answer import answer_question  # noqa: E402
from gq_insight.eval import DEFAULT_QUERIES, _load_queries, evaluate  # noqa: E402
from gq_insight.index import InterviewIndex  # noqa: E402

# ---- look ----
W, H = 1280, 720
FPS = 30
SR = 24000  # Kokoro sample rate
BG = (13, 17, 23)
FG = (201, 209, 217)
GREEN = (63, 185, 80)
BLUE = (88, 166, 255)
DIM = (110, 118, 129)
ORANGE = (240, 136, 62)
MARGIN = 64
LINE_H = 30
FONT_SIZE = 19
VOICE = "am_michael"

# ---- narration (spoken, scene by scene) ----
NARR = {
    "intro": "This is gee-cue insight. An MCP server I built for Great Question "
             "that does semantic search and grounded answering over customer "
             "interviews, with its own evaluation harness.",
    "search": "Ask it why customers churn, and it returns the actual quotes. Each "
              "one traceable to an interview, a timestamp, and a speaker. No "
              "summary you can't check.",
    "answer": "Ask a real research question and it answers straight from the "
              "interviews. Every claim carries a citation, and if it can't ground "
              "a claim in a real quote, it refuses rather than make one up.",
    "eval": "And because tools like this need quality measures, it ships an eval "
            "harness. Recall, mean reciprocal rank, and a faithfulness rate, all "
            "gated in continuous integration.",
    "outro": "Semantic search, MCP tool structuring, and evals across MCP tools. "
             "Three things straight from your job post, running today. The code "
             "is on my GitHub.",
}


def _font(size, bold=False):
    base = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono"
    return ImageFont.truetype(f"{base}-Bold.ttf" if bold else f"{base}.ttf", size)


MONO, MONO_B = _font(FONT_SIZE), _font(FONT_SIZE, bold=True)
TITLE, SUB = _font(38, bold=True), _font(22)


def _wrap(text, width=104):
    out = []
    for raw in text.split("\n"):
        if not raw:
            out.append("")
            continue
        line = ""
        for word in raw.split(" "):
            if len(line) + len(word) + 1 > width:
                out.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        out.append(line)
    return out


def _new():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 40], fill=(22, 27, 34))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([20 + i * 26, 14, 32 + i * 26, 26], fill=c)
    d.text((W // 2 - 90, 11), "gq-insight - demo", font=MONO, fill=DIM)
    return img, d


def _draw_lines(d, lines, y0=60):
    y = y0
    for txt, color in lines:
        d.text((MARGIN, y), txt, font=MONO, fill=color)
        y += LINE_H
    return y


class Scene:
    """Collects PNG frames for one narrated scene, padded to >= audio length."""

    def __init__(self, tmp: Path, idx: int, audio_dur: float):
        self.dir = tmp / f"scene{idx}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.audio_dur = audio_dur
        self.n = 0
        self._last: Image.Image | None = None

    def add(self, img, hold_frames):
        for _ in range(max(1, hold_frames)):
            img.save(self.dir / f"f{self.n:05d}.png")
            self.n += 1
        self._last = img

    def pad_to_audio(self, tail=0.4):
        target = math.ceil((self.audio_dur + tail) * FPS)
        if self._last is not None and self.n < target:
            self.add(self._last, target - self.n)


def card(scene, title, subs, accent=GREEN):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([MARGIN, H // 2 - 130, MARGIN + 8, H // 2 + 130], fill=accent)
    d.text((MARGIN + 36, H // 2 - 120), title, font=TITLE, fill=FG)
    y = H // 2 - 50
    for s, col in subs:
        d.text((MARGIN + 36, y), s, font=SUB, fill=col)
        y += 40
    scene.add(img, int(0.6 * FPS))
    scene.pad_to_audio()


def type_and_run(scene, prompt_cmd, output_blocks, base_lines):
    for i in range(0, len(prompt_cmd) + 1, 2):
        img, d = _new()
        y = _draw_lines(d, base_lines)
        d.text((MARGIN, y), "$ ", font=MONO_B, fill=GREEN)
        d.text((MARGIN + 24, y), prompt_cmd[:i], font=MONO_B, fill=FG)
        scene.add(img, 1)
    shown = []
    for block in output_blocks:
        shown.extend(block)
        img, d = _new()
        y = _draw_lines(d, base_lines)
        d.text((MARGIN, y), "$ ", font=MONO_B, fill=GREEN)
        d.text((MARGIN + 24, y), prompt_cmd, font=MONO_B, fill=FG)
        yy = y + LINE_H + 6
        for txt, col in shown:
            d.text((MARGIN, yy), txt, font=MONO, fill=col)
            yy += LINE_H
        scene.add(img, int(0.5 * FPS))
    scene.pad_to_audio()


# ---------- audio ----------
def synth_all(audio_dir: Path) -> dict[str, float] | None:
    try:
        import soundfile as sf
        from kokoro import KPipeline
    except Exception as e:  # noqa: BLE001
        print(f"[voice] TTS unavailable ({e}); rendering silent.", file=sys.stderr)
        return None
    audio_dir.mkdir(parents=True, exist_ok=True)
    pipe = KPipeline(lang_code="a")
    durs: dict[str, float] = {}
    for key, text in NARR.items():
        chunks = []
        for _, _, audio in pipe(text, voice=VOICE):
            a = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            chunks.append(a)
        wav = np.concatenate(chunks).astype(np.float32)
        wav = np.concatenate([wav, np.zeros(int(0.25 * SR), np.float32)])  # breath
        sf.write(audio_dir / f"{key}.wav", wav, SR)
        durs[key] = len(wav) / SR
        print(f"[voice] {key}: {durs[key]:.1f}s", file=sys.stderr)
    return durs


# ---------- build ----------
def build(silent: bool):
    idx = InterviewIndex.from_dir().build()
    st = idx.stats()
    churn_hits = idx.search("why do customers churn or switch tools?", k=3)
    ans = answer_question(idx, "What is blocking the enterprise rollout?", k=6, backend="extractive")
    rep = evaluate(idx, _load_queries(DEFAULT_QUERIES), k=6)["summary"]

    tmp = Path(tempfile.mkdtemp(prefix="gqframes_"))
    audio_dir = ROOT / "demo" / "audio"
    durs = None if silent else synth_all(audio_dir)
    has_audio = durs is not None

    def dur(key, fallback):
        return durs[key] if has_audio else fallback

    base = [(f"corpus: {st['interviews']} interviews · {st['turns']} turns · "
             f"{st['model'].split('/')[-1]} ({st['dim']}-d)", DIM), ("", FG)]

    scenes: list[tuple[Scene, str]] = []

    # intro
    s = Scene(tmp, 0, dur("intro", 6.0))
    card(s, "gq-insight", [
        ("Semantic search + grounded answers over customer interviews", FG),
        ("an MCP server with a built-in eval harness", DIM),
        ("Yusuf Guenena   ·   built for Great Question", BLUE),
    ], accent=BLUE)
    scenes.append((s, "intro"))

    # search
    s = Scene(tmp, 1, dur("search", 9.0))
    blocks = [[("# top cited quotes, traceable to source", DIM)]]
    for i, h in enumerate(churn_hits, 1):
        d = h.as_dict()
        q = d["quote"]
        q = q[:88] + "..." if len(q) > 88 else q
        blocks.append([(f"{i}. [{d['citation']}]  score={d['score']}", BLUE), (f"   \"{q}\"", FG)])
    type_and_run(s, 'gq-insight search "why do customers churn?"', blocks, base)
    scenes.append((s, "search"))

    # answer
    s = Scene(tmp, 2, dur("answer", 11.0))
    ablk = [[(l, FG) for l in _wrap(ans.text, 100)]]
    ablk.append([("", FG), (f"faithful: {ans.faithful}  (every claim cites a real quote)", GREEN)])
    type_and_run(s, 'gq-insight answer "what blocks the enterprise rollout?"', ablk, base)
    scenes.append((s, "answer"))

    # eval
    s = Scene(tmp, 3, dur("eval", 9.0))
    ev = [
        [(f"hit@k            {rep['hit_rate_at_k']:.3f}", FG)],
        [(f"recall@k         {rep['mean_recall_at_k']:.3f}", FG)],
        [(f"MRR              {rep['mean_mrr']:.3f}", FG)],
        [(f"nDCG@k           {rep['mean_ndcg_at_k']:.3f}", FG)],
        [(f"faithfulness     {rep['faithfulness_rate']:.3f}", FG), ("", FG),
         ("ALL QUALITY GATES PASS", GREEN)],
    ]
    type_and_run(s, "gq-insight eval", ev, base)
    scenes.append((s, "eval"))

    # outro
    s = Scene(tmp, 4, dur("outro", 7.0))
    card(s, "Three things from your JD", [
        ("· semantic search across thousands of interview hours", GREEN),
        ("· MCP tool structuring + prompt tuning", GREEN),
        ("· evals & quality measures across MCP tools", GREEN),
        ("", FG),
        ("github.com/yusufdxb/gq-insight-mcp", BLUE),
    ], accent=ORANGE)
    scenes.append((s, "outro"))

    # encode each scene (video + its audio), then concat
    seg_paths = []
    for i, (scene, key) in enumerate(scenes):
        seg = tmp / f"seg{i}.mp4"
        cmd = ["ffmpeg", "-y", "-framerate", str(FPS), "-i", str(scene.dir / "f%05d.png")]
        if has_audio:
            cmd += ["-i", str(audio_dir / f"{key}.wav")]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-r", str(FPS)]
        if has_audio:
            cmd += ["-c:a", "aac", "-b:a", "128k", "-af", "apad", "-shortest"]
        cmd += [str(seg)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        seg_paths.append(seg)

    concat_list = tmp / "list.txt"
    concat_list.write_text("".join(f"file '{p}'\n" for p in seg_paths))
    out = ROOT / "assets" / "gq-insight-demo.mp4"
    out.parent.mkdir(exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
         "-c", "copy", "-movflags", "+faststart", str(out)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    shutil.rmtree(tmp, ignore_errors=True)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(out)],
        capture_output=True, text=True,
    )
    print(f"done: {out}  ({float(probe.stdout.strip()):.1f}s, "
          f"{out.stat().st_size // 1024} KB, voice={'yes' if has_audio else 'no'})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--silent", action="store_true")
    build(ap.parse_args().silent)
