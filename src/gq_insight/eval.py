"""Evaluation harness for the retrieval + answering MCP tools.

Two quality gates, both threshold-checked so this can run in CI:

  1. Retrieval quality against a labeled query set
     (data/queries.jsonl -> relevant interview ids):
        hit@k, recall@k, precision@k, MRR, nDCG@k.
  2. Answer faithfulness: the fraction of answers whose every claim is
     grounded in a retrieved quote (no invented citations).

Run:  python -m gq_insight.eval            # extractive (offline, deterministic)
      python -m gq_insight.eval --backend ollama
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .answer import answer_question
from .index import InterviewIndex

DEFAULT_QUERIES = Path(__file__).resolve().parents[2] / "evals" / "queries.jsonl"

# CI gates are conservative floors, deliberately set below measured performance
# (recall 0.90, MRR 0.79, nDCG 0.84 with all-MiniLM-L6-v2) so the gate catches
# regressions without being gamed up to a flattering number. Q1 (onboarding) and
# Q3 (integrations) rank the right interview only 4th-5th -- a real limitation of
# a small embedder on abstract queries over concrete transcript language -- and
# are kept in the set so the gate stays honest.
GATES = {"mean_recall_at_k": 0.80, "mean_mrr": 0.70, "faithfulness_rate": 1.0}


@dataclass
class QueryResult:
    id: str
    query: str
    relevant: list[str]
    retrieved: list[str]  # interview ids of top-k chunks, in rank order (with dups)
    hit: int
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    faithful: bool


def _load_queries(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _ndcg(rels: list[int]) -> float:
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    ideal = sorted(rels, reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(
    index: InterviewIndex,
    queries: list[dict],
    k: int = 6,
    backend: str = "extractive",
) -> dict:
    results: list[QueryResult] = []
    for q in queries:
        gold = set(q["relevant"])
        hits = index.search(q["query"], k=k)
        retrieved_ids = [h.turn.interview_id for h in hits]
        rels = [1 if iid in gold else 0 for iid in retrieved_ids]

        found_unique = {iid for iid in retrieved_ids if iid in gold}
        recall = len(found_unique) / len(gold) if gold else 0.0
        precision = sum(rels) / len(rels) if rels else 0.0
        rank = next((i + 1 for i, r in enumerate(rels) if r), 0)
        mrr = 1.0 / rank if rank else 0.0
        ndcg = _ndcg(rels)

        ans = answer_question(index, q["query"], k=k, backend=backend)

        results.append(
            QueryResult(
                id=q["id"], query=q["query"], relevant=sorted(gold),
                retrieved=retrieved_ids, hit=int(bool(rank)),
                recall_at_k=round(recall, 4), precision_at_k=round(precision, 4),
                mrr=round(mrr, 4), ndcg_at_k=round(ndcg, 4), faithful=ans.faithful,
            )
        )

    n = len(results)
    summary = {
        "queries": n,
        "k": k,
        "backend": backend,
        "hit_rate_at_k": round(sum(r.hit for r in results) / n, 4),
        "mean_recall_at_k": round(sum(r.recall_at_k for r in results) / n, 4),
        "mean_precision_at_k": round(sum(r.precision_at_k for r in results) / n, 4),
        "mean_mrr": round(sum(r.mrr for r in results) / n, 4),
        "mean_ndcg_at_k": round(sum(r.ndcg_at_k for r in results) / n, 4),
        "faithfulness_rate": round(sum(r.faithful for r in results) / n, 4),
    }
    gate_status = {g: summary[g] >= thr for g, thr in GATES.items()}
    return {"summary": summary, "gates": GATES, "gate_pass": gate_status,
            "all_gates_pass": all(gate_status.values()),
            "per_query": [asdict(r) for r in results]}


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval the gq-insight retrieval + answer tools")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--backend", choices=["extractive", "ollama"], default="extractive")
    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    ap.add_argument("--json", action="store_true", help="emit full JSON report")
    args = ap.parse_args()

    index = InterviewIndex.from_dir().build()
    report = evaluate(index, _load_queries(args.queries), k=args.k, backend=args.backend)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["all_gates_pass"] else 1

    s = report["summary"]
    print(f"\n  gq-insight eval  ({s['queries']} queries, k={s['k']}, backend={s['backend']})")
    print("  " + "-" * 46)
    for key in ["hit_rate_at_k", "mean_recall_at_k", "mean_precision_at_k",
                "mean_mrr", "mean_ndcg_at_k", "faithfulness_rate"]:
        print(f"  {key:<22} {s[key]:.3f}")
    print("  " + "-" * 46)
    for g, passed in report["gate_pass"].items():
        mark = "PASS" if passed else "FAIL"
        print(f"  gate {g:<20} >= {GATES[g]:<5} [{mark}]")
    verdict = "ALL GATES PASS" if report["all_gates_pass"] else "GATES FAILED"
    print(f"\n  {verdict}\n")
    return 0 if report["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
