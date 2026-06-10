"""Faithfulness logic is pure string/citation checking — no model needed,
so these run fast and deterministically."""

from gq_insight.answer import check_faithfulness
from gq_insight.corpus import Turn
from gq_insight.index import SearchHit


def _hit(iid, ts, text="x"):
    return SearchHit(
        turn=Turn(iid, 0, ts, "P", text, "P", "2026-01-01"), score=1.0
    )


def test_uncited_answer_is_unfaithful():
    hits = [_hit("INT001", "00:52")]
    ok, note = check_faithfulness("Customers struggled with setup.", hits)
    assert ok is False
    assert "no citations" in note


def test_grounded_citation_is_faithful():
    hits = [_hit("INT001", "00:52"), _hit("INT003", "04:40")]
    text = "Setup was slow [INT001 @ 00:52] and mobile assumed a desk [INT003 @ 04:40]."
    ok, note = check_faithfulness(text, hits)
    assert ok is True
    assert "all grounded" in note


def test_invented_citation_is_caught():
    hits = [_hit("INT001", "00:52")]
    text = "Customers loved it [INT099 @ 99:99]."
    ok, note = check_faithfulness(text, hits)
    assert ok is False
    assert "not in retrieved set" in note
