"""Thin CLI over the same functions the MCP tools expose. Every command shown
in the demo video is runnable here, verbatim.

    gq-insight search "why do customers churn?"
    gq-insight answer "what blocks an enterprise rollout?"
    gq-insight eval
    gq-insight themes
"""

from __future__ import annotations

import argparse
import json

from .answer import answer_question
from .eval import DEFAULT_QUERIES, _load_queries, evaluate
from .index import InterviewIndex


def _cmd_search(idx, args):
    for i, h in enumerate(idx.search(args.query, k=args.k), 1):
        d = h.as_dict()
        print(f"{i}. [{d['citation']}]  score={d['score']}")
        print(f"   \"{d['quote']}\"\n")


def _cmd_answer(idx, args):
    a = answer_question(idx, args.question, k=args.k, backend=args.backend)
    print(a.text)
    print(f"\n  faithful: {a.faithful}  ({a.faithfulness_note})  [backend: {a.backend}]")


def _cmd_eval(idx, args):
    rep = evaluate(idx, _load_queries(DEFAULT_QUERIES), k=args.k)
    print(json.dumps(rep["summary"], indent=2))
    print("all_gates_pass:", rep["all_gates_pass"])


def _cmd_themes(idx, args):
    for iv in idx.interviews:
        print(f"{iv.interview_id}  {len(iv.turns):>2} turns  {iv.participant}")


def main() -> int:
    ap = argparse.ArgumentParser(prog="gq-insight", description="customer-interview insight CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="semantic search for cited quotes")
    s.add_argument("query"); s.add_argument("-k", type=int, default=3); s.set_defaults(fn=_cmd_search)

    a = sub.add_parser("answer", help="grounded, cited answer to a question")
    a.add_argument("question"); a.add_argument("-k", type=int, default=6)
    a.add_argument("--backend", choices=["extractive", "ollama"], default="extractive")
    a.set_defaults(fn=_cmd_answer)

    e = sub.add_parser("eval", help="quality scorecard for the tools")
    e.add_argument("-k", type=int, default=6); e.set_defaults(fn=_cmd_eval)

    t = sub.add_parser("themes", help="list the interview corpus"); t.set_defaults(fn=_cmd_themes)

    args = ap.parse_args()
    idx = InterviewIndex.from_dir().build()
    args.fn(idx, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
