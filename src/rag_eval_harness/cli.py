from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .metrics import evaluate
from .retrieval import BM25


def run(corpus: dict, golden: list[dict], synonyms: dict | None, k: int) -> dict:
    bm = BM25(corpus)
    result = {"baseline": evaluate(golden, lambda q, n: bm.search(q, n), k)}
    if synonyms:
        result["expanded"] = evaluate(golden, lambda q, n: bm.search(q, n, synonyms), k)
    return result


def render(result: dict) -> str:
    lines = ["# Retrieval report", ""]
    for name, r in result.items():
        key = f"recall@{r['k']}"
        lines.append(f"- **{name}**: {key} = {r[key]:.2f}, MRR = {r['mrr']:.2f} ({r['n']} questions)")
    last = list(result.values())[-1]
    lines += ["", f"## Misses ({list(result)[-1]})", ""]
    if not last["misses"]:
        lines.append("None.")
    for m in last["misses"]:
        lines.append(f"- `{m['id']}` {m['question']!r}: expected {m['expected']}, got {m['got']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rag-eval", description=__doc__)
    p.add_argument("dataset", type=Path, help="JSON file with corpus, golden, optional synonyms")
    p.add_argument("-k", type=int, default=3)
    p.add_argument("--min-recall", type=float, help="exit 1 if final Recall@k is below this")
    a = p.parse_args(argv)
    data = json.loads(a.dataset.read_text(encoding="utf-8"))
    result = run(data["corpus"], data["golden"], data.get("synonyms"), a.k)
    sys.stdout.write(render(result))
    final = list(result.values())[-1]
    if a.min_recall is not None and final[f"recall@{a.k}"] < a.min_recall:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
