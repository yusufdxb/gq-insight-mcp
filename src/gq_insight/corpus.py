"""Load customer-research interview transcripts into citable turns.

Each transcript is a plain-text file with a small header block and a body of
timestamped speaker turns:

    # Interview INT001
    participant: P-1042 - Operations Lead ...
    date: 2026-02-03
    ---
    [00:38] Interviewer: Walk me through ...
    [00:52] P-1042: Honestly the first week ...

A *turn* is the atomic unit of retrieval and the thing we cite. Citing a whole
30-minute transcript is useless to a researcher; citing "INT002 @ 02:11,
P-2210" is a quote they can drop into a deck.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# [mm:ss] Speaker: text
_TURN_RE = re.compile(r"^\[(\d{1,2}:\d{2})\]\s+([^:]+?):\s*(.*)$")
_HEADER_KEY_RE = re.compile(r"^([a-zA-Z_]+):\s*(.*)$")

DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "transcripts"


@dataclass(frozen=True)
class Turn:
    """One speaker turn — the retrieval and citation unit."""

    interview_id: str
    index: int  # 0-based position within the interview
    timestamp: str  # "mm:ss"
    speaker: str
    text: str
    participant: str  # interview-level participant label, for context
    date: str

    @property
    def turn_id(self) -> str:
        return f"{self.interview_id}#{self.index}"

    @property
    def citation(self) -> str:
        return f"{self.interview_id} @ {self.timestamp} ({self.speaker})"


@dataclass
class Interview:
    interview_id: str
    participant: str
    date: str
    product: str
    source_path: Path
    turns: list[Turn] = field(default_factory=list)


def _parse_header(lines: list[str]) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in lines:
        line = line.strip()
        if line.startswith("# Interview"):
            meta["interview_id"] = line.split("Interview", 1)[1].strip()
            continue
        m = _HEADER_KEY_RE.match(line)
        if m:
            meta[m.group(1).lower()] = m.group(2).strip()
    return meta


def parse_transcript(path: Path) -> Interview:
    raw = path.read_text(encoding="utf-8")
    if "---" not in raw:
        raise ValueError(f"{path.name}: missing '---' separator between header and body")
    header_block, body_block = raw.split("---", 1)

    meta = _parse_header(header_block.splitlines())
    interview_id = meta.get("interview_id") or path.stem
    participant = meta.get("participant", "unknown")
    date = meta.get("date", "unknown")
    product = meta.get("product", "unknown")

    interview = Interview(
        interview_id=interview_id,
        participant=participant,
        date=date,
        product=product,
        source_path=path,
    )

    idx = 0
    for line in body_block.splitlines():
        m = _TURN_RE.match(line.strip())
        if not m:
            continue
        timestamp, speaker, text = m.group(1), m.group(2).strip(), m.group(3).strip()
        if not text:
            continue
        interview.turns.append(
            Turn(
                interview_id=interview_id,
                index=idx,
                timestamp=timestamp,
                speaker=speaker,
                text=text,
                participant=participant,
                date=date,
            )
        )
        idx += 1

    if not interview.turns:
        raise ValueError(f"{path.name}: no speaker turns parsed")
    return interview


def load_corpus(corpus_dir: Path | str = DEFAULT_CORPUS_DIR) -> list[Interview]:
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"corpus dir not found: {corpus_dir}")
    interviews = [parse_transcript(p) for p in sorted(corpus_dir.glob("*.txt"))]
    if not interviews:
        raise ValueError(f"no transcripts (*.txt) found in {corpus_dir}")
    return interviews


def all_turns(interviews: list[Interview]) -> list[Turn]:
    return [t for iv in interviews for t in iv.turns]
