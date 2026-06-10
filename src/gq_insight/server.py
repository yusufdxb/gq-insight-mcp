"""MCP server exposing the interview corpus to any LLM agent.

Tools:
  * search_interviews(query, k)        -> ranked, cited quotes
  * answer_with_citations(question, k) -> grounded answer + faithfulness flag
  * list_themes()                      -> corpus map (interviews + participants)
  * run_eval(k)                        -> live quality scorecard for these tools

Run:  python -m gq_insight.server         (stdio transport, for Claude/agents)

The index is built once at process start and reused across tool calls, so the
embedding cost is paid a single time per session.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .answer import answer_question
from .eval import DEFAULT_QUERIES, _load_queries, evaluate
from .index import InterviewIndex

mcp = FastMCP("gq-insight")

_index: InterviewIndex | None = None


def get_index() -> InterviewIndex:
    global _index
    if _index is None:
        _index = InterviewIndex.from_dir().build()
    return _index


@mcp.tool()
def search_interviews(query: str, k: int = 5) -> dict:
    """Semantic search over customer-interview transcripts.

    Returns the k most relevant verbatim quotes, each with a citation
    (interview id, timestamp, speaker) so findings stay traceable to source.
    """
    hits = get_index().search(query, k=k)
    return {"query": query, "k": k, "results": [h.as_dict() for h in hits]}


@mcp.tool()
def answer_with_citations(question: str, k: int = 6, backend: str = "extractive") -> dict:
    """Answer a research question grounded in the interviews.

    Every claim is backed by a quote and an inline citation. The response
    includes a `faithful` flag: an answer that cannot be grounded is refused
    rather than fabricated. `backend` may be 'extractive' (offline, default) or
    'ollama' (local LLM synthesis, verified before return).
    """
    return answer_question(get_index(), question, k=k, backend=backend).as_dict()


@mcp.tool()
def list_themes() -> dict:
    """Map the corpus: every interview, its participant, and turn count."""
    idx = get_index()
    return {
        "stats": idx.stats(),
        "interviews": [
            {
                "interview_id": iv.interview_id,
                "participant": iv.participant,
                "date": iv.date,
                "turns": len(iv.turns),
            }
            for iv in idx.interviews
        ],
    }


@mcp.tool()
def run_eval(k: int = 6) -> dict:
    """Score the retrieval + answer tools against the labeled eval set.

    Returns hit@k, recall@k, precision@k, MRR, nDCG@k, and answer-faithfulness,
    plus pass/fail against the CI quality gates.
    """
    report = evaluate(get_index(), _load_queries(DEFAULT_QUERIES), k=k)
    return {"summary": report["summary"], "gate_pass": report["gate_pass"],
            "all_gates_pass": report["all_gates_pass"]}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
