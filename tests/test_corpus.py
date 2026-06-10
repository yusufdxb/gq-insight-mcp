from pathlib import Path

import pytest

from gq_insight.corpus import all_turns, load_corpus, parse_transcript

CORPUS = Path(__file__).resolve().parents[1] / "data" / "transcripts"


def test_loads_all_interviews():
    interviews = load_corpus(CORPUS)
    assert len(interviews) == 8
    ids = {iv.interview_id for iv in interviews}
    assert {"INT001", "INT005", "INT008"} <= ids


def test_turns_are_citable():
    interviews = load_corpus(CORPUS)
    turns = all_turns(interviews)
    assert len(turns) > 50
    t = turns[0]
    assert t.turn_id.startswith("INT")
    assert "@" in t.citation and t.timestamp in t.citation


def test_header_parsed():
    iv = parse_transcript(CORPUS / "INT002_pricing.txt")
    assert iv.interview_id == "INT002"
    assert "Finance Manager" in iv.participant
    assert iv.date == "2026-02-05"


def test_rejects_missing_separator(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("# Interview INTX\n[00:01] A: hi\n")
    with pytest.raises(ValueError):
        parse_transcript(bad)


def test_rejects_empty_body(tmp_path):
    bad = tmp_path / "empty.txt"
    bad.write_text("# Interview INTY\ndate: 2026-01-01\n---\n\n")
    with pytest.raises(ValueError):
        parse_transcript(bad)
