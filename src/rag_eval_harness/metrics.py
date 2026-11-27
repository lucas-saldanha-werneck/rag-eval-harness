from __future__ import annotations

from typing import Callable

Retriever = Callable[[str, int], list[str]]


def evaluate(golden: list[dict], retrieve: Retriever, k: int = 3) -> dict:
    """golden: [{"id", "question", "relevant": [doc ids]}]. Returns metrics and misses."""
    if not golden:
        raise ValueError("golden set is empty")
    hits, rr_sum, misses = 0, 0.0, []
    for q in golden:
        relevant = set(q["relevant"])
        got = retrieve(q["question"], k)
        rank = next((i + 1 for i, d in enumerate(got) if d in relevant), None)
        if rank is not None:
            hits += 1
            rr_sum += 1 / rank
        else:
            misses.append({"id": q["id"], "question": q["question"],
                           "expected": sorted(relevant), "got": got})
    n = len(golden)
    return {"n": n, "k": k, f"recall@{k}": hits / n, "mrr": rr_sum / n, "misses": misses}
