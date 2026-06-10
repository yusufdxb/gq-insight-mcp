"""gq-insight: an MCP server for semantic search + grounded answering over
customer-research interview transcripts, with a built-in eval harness."""

from .answer import Answer, answer_question, check_faithfulness
from .corpus import Interview, Turn, load_corpus
from .index import InterviewIndex, SearchHit

__all__ = [
    "Answer",
    "Interview",
    "InterviewIndex",
    "SearchHit",
    "Turn",
    "answer_question",
    "check_faithfulness",
    "load_corpus",
]
