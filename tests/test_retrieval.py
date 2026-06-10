"""Retrieval + eval tests. These load the embedding model once (session-scoped
fixture) and assert the semantic index actually finds the right interviews."""

import json
from pathlib import Path

import pytest

from gq_insight.answer import answer_question
from gq_insight.eval import evaluate
from gq_insight.index import InterviewIndex

QUERIES = Path(__file__).resolve().parents[1] / "evals" / "queries.jsonl"


@pytest.fixture(scope="session")
def index():
    return InterviewIndex.from_dir().build()


def test_search_finds_pricing_interview(index):
    hits = index.search("unexpected bill, charged per person", k=3)
    assert any(h.turn.interview_id == "INT002" for h in hits)
    assert hits[0].score > hits[-1].score  # ranked


def test_search_finds_security_interview(index):
    hits = index.search("we need SAML single sign-on for compliance", k=3)
    assert any(h.turn.interview_id == "INT008" for h in hits)


def test_excludes_interviewer_by_default(index):
    hits = index.search("what is blocking the rollout?", k=6)
    assert all(h.turn.speaker.lower() != "interviewer" for h in hits)
    # but interviewer turns are still reachable when explicitly requested
    incl = index.search("walk me through the first week", k=10, participant_only=False)
    assert any(h.turn.speaker.lower() == "interviewer" for h in incl)


def test_empty_query_rejected(index):
    with pytest.raises(ValueError):
        index.search("   ", k=3)


def test_extractive_answer_is_faithful(index):
    ans = answer_question(index, "Why did customers churn?", k=6, backend="extractive")
    assert ans.faithful
    assert ans.supporting
    assert "[INT" in ans.text


def test_eval_clears_gates(index):
    queries = [json.loads(l) for l in QUERIES.read_text().splitlines() if l.strip()]
    report = evaluate(index, queries, k=6)
    s = report["summary"]
    assert s["mean_recall_at_k"] >= 0.80
    assert s["mean_mrr"] >= 0.70
    assert s["faithfulness_rate"] == 1.0
    assert report["all_gates_pass"]
