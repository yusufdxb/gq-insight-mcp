"""Dense semantic index over interview turns.

Embeddings are computed once with a small sentence-transformer
(`all-MiniLM-L6-v2`, 384-dim, CPU-friendly) and searched with cosine
similarity. On this corpus that is exact and instant; the same interface
scales to an approximate index (FAISS/HNSW) when you cross "tens of thousands
of interview hours" — the search contract does not change, only the backend.

The model is loaded lazily so importing this module (e.g. in tests that only
exercise the parser) does not pay the model-load cost.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .corpus import Interview, Turn, all_turns, load_corpus

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_model_lock = threading.Lock()


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    # Imported here so the dependency is only required when actually embedding.
    from sentence_transformers import SentenceTransformer

    with _model_lock:
        return SentenceTransformer(model_name)


@dataclass
class SearchHit:
    turn: Turn
    score: float

    def as_dict(self) -> dict:
        return {
            "interview_id": self.turn.interview_id,
            "timestamp": self.turn.timestamp,
            "speaker": self.turn.speaker,
            "participant": self.turn.participant,
            "citation": self.turn.citation,
            "quote": self.turn.text,
            "score": round(float(self.score), 4),
        }


class InterviewIndex:
    """Embed every turn once; answer top-k cosine queries against it."""

    def __init__(self, interviews: list[Interview], model_name: str = DEFAULT_MODEL):
        if not interviews:
            raise ValueError("InterviewIndex requires at least one interview")
        self.model_name = model_name
        self.interviews = interviews
        self.turns: list[Turn] = all_turns(interviews)
        # Researchers want what the customer said, not the moderator's prompts.
        # Interviewer turns are indexed for context but excluded from results
        # by default.
        self._is_participant = np.array(
            [t.speaker.strip().lower() != "interviewer" for t in self.turns]
        )
        self._embeddings: np.ndarray | None = None

    @classmethod
    def from_dir(cls, corpus_dir: Path | str | None = None, model_name: str = DEFAULT_MODEL) -> "InterviewIndex":
        interviews = load_corpus(corpus_dir) if corpus_dir else load_corpus()
        return cls(interviews, model_name=model_name)

    @property
    def embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            self.build()
        assert self._embeddings is not None
        return self._embeddings

    def build(self) -> "InterviewIndex":
        model = _load_model(self.model_name)
        texts = [t.text for t in self.turns]
        emb = model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine == dot product
            show_progress_bar=False,
        )
        self._embeddings = emb.astype(np.float32)
        return self

    def search(self, query: str, k: int = 5, participant_only: bool = True) -> list[SearchHit]:
        if not query or not query.strip():
            raise ValueError("query must be non-empty")
        model = _load_model(self.model_name)
        q = model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )[0].astype(np.float32)
        scores = self.embeddings @ q  # cosine, since both are L2-normalized
        if participant_only:
            scores = np.where(self._is_participant, scores, -np.inf)
            pool = int(self._is_participant.sum())
        else:
            pool = len(self.turns)
        k = max(1, min(k, pool))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [SearchHit(turn=self.turns[i], score=float(scores[i])) for i in top]

    def stats(self) -> dict:
        return {
            "interviews": len(self.interviews),
            "turns": len(self.turns),
            "model": self.model_name,
            "dim": int(self.embeddings.shape[1]),
        }
