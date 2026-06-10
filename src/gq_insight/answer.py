"""Quote-grounded answering over retrieved interview turns.

Design rule: an answer may only assert what a retrieved quote supports, and
every claim carries an inline citation like ``[INT002 @ 02:11]``. This is the
property the eval harness checks (faithfulness): zero uncited sentences, zero
citations that point outside the retrieved set. No grounding, no answer.

Two backends, same contract:
  * ``extractive`` (default, zero-dependency): stitches the top quotes into a
    cited summary. Deterministic, always faithful, reviewer can run it offline.
  * ``ollama`` (optional): asks a local model to synthesize, under a prompt that
    forbids uncited claims, then *verifies* the output and falls back to
    extractive if the model invents a citation. Prompt tuning lives here.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

from .index import InterviewIndex, SearchHit

_CITATION_RE = re.compile(r"\[([A-Z]+\d+)\s*@\s*(\d{1,2}:\d{2})\]")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_DEFAULT_MODEL = "llama3.1:8b"

_SYSTEM_PROMPT = """You are a customer-research analyst. Answer the question \
using ONLY the numbered interview excerpts provided. Rules:
- Every sentence that makes a claim MUST end with a citation in the exact form [INTxxx @ mm:ss].
- Use ONLY citations that appear in the excerpts below. Never invent one.
- If the excerpts do not answer the question, say so in one sentence. Do not speculate.
- Be concise: 2-4 sentences. No preamble."""


@dataclass
class Answer:
    question: str
    text: str
    backend: str
    supporting: list[SearchHit]
    faithful: bool
    faithfulness_note: str

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.text,
            "backend": self.backend,
            "faithful": self.faithful,
            "faithfulness_note": self.faithfulness_note,
            "citations": [h.turn.citation for h in self.supporting],
            "supporting_quotes": [h.as_dict() for h in self.supporting],
        }


def _allowed_citations(hits: list[SearchHit]) -> set[tuple[str, str]]:
    return {(h.turn.interview_id, h.turn.timestamp) for h in hits}


def check_faithfulness(text: str, hits: list[SearchHit]) -> tuple[bool, str]:
    """An answer is faithful iff it cites, and every cited (interview, ts) was retrieved."""
    found = _CITATION_RE.findall(text)
    if not found:
        return False, "no citations in answer"
    allowed = _allowed_citations(hits)
    bad = [f"[{iid} @ {ts}]" for (iid, ts) in found if (iid, ts) not in allowed]
    if bad:
        return False, f"cites quotes not in retrieved set: {', '.join(sorted(set(bad)))}"
    return True, f"{len(found)} citation(s), all grounded in retrieved quotes"


def _extractive(question: str, hits: list[SearchHit], max_quotes: int) -> str:
    picked = hits[:max_quotes]
    parts = [
        f'"{h.turn.text.rstrip(".")}." [{h.turn.interview_id} @ {h.turn.timestamp}]'
        for h in picked
    ]
    return " ".join(parts)


def _ollama_synthesize(question: str, hits: list[SearchHit], model: str, timeout: float) -> str | None:
    excerpts = "\n".join(
        f"[{h.turn.interview_id} @ {h.turn.timestamp}] {h.turn.participant}: {h.turn.text}"
        for h in hits
    )
    prompt = f"{_SYSTEM_PROMPT}\n\nExcerpts:\n{excerpts}\n\nQuestion: {question}\n\nAnswer:"
    payload = json.dumps(
        {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.0}}
    ).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read()).get("response", "").strip() or None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def answer_question(
    index: InterviewIndex,
    question: str,
    k: int = 6,
    backend: str = "extractive",
    max_quotes: int = 3,
    ollama_model: str = OLLAMA_DEFAULT_MODEL,
    ollama_timeout: float = 30.0,
) -> Answer:
    hits = index.search(question, k=k)

    if backend == "ollama":
        synth = _ollama_synthesize(question, hits, ollama_model, ollama_timeout)
        if synth:
            faithful, note = check_faithfulness(synth, hits)
            if faithful:
                return Answer(question, synth, "ollama", hits, True, note)
            # Model hallucinated a citation -> fall back rather than ship it.
            text = _extractive(question, hits, max_quotes)
            ok, note2 = check_faithfulness(text, hits)
            return Answer(
                question, text, "extractive(fallback)", hits, ok,
                f"ollama rejected ({note}); used extractive: {note2}",
            )
        # Ollama unavailable -> graceful extractive.
        text = _extractive(question, hits, max_quotes)
        ok, note = check_faithfulness(text, hits)
        return Answer(question, text, "extractive(fallback)", hits, ok, f"ollama unavailable; {note}")

    text = _extractive(question, hits, max_quotes)
    ok, note = check_faithfulness(text, hits)
    return Answer(question, text, "extractive", hits, ok, note)
