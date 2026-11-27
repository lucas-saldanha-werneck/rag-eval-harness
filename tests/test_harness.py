import json
from pathlib import Path

import pytest

from rag_eval_harness.cli import main, render, run
from rag_eval_harness.metrics import evaluate
from rag_eval_harness.retrieval import BM25, tokenize

SAMPLE = Path(__file__).parent.parent / "examples" / "sample.json"


def load():
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


def test_tokenize_drops_stopwords_and_case():
    assert tokenize("How do I Reset the Password?") == ["reset", "password"]


def test_bm25_ranks_exact_term_first():
    bm = BM25({"a": "cats purr", "b": "dogs bark loudly", "c": "birds sing"})
    assert bm.search("dogs", 3) == ["b"]


def test_bm25_empty_corpus_rejected():
    with pytest.raises(ValueError):
        BM25({})


def test_no_match_returns_nothing():
    assert BM25({"a": "cats purr"}).search("zzz", 3) == []


def test_metrics_rank_math():
    golden = [{"id": "x", "question": "q1", "relevant": ["a"]},
              {"id": "y", "question": "q2", "relevant": ["b"]},
              {"id": "z", "question": "q3", "relevant": ["c"]}]
    answers = {"q1": ["a", "b"], "q2": ["c", "b"], "q3": ["d"]}
    r = evaluate(golden, lambda q, k: answers[q], k=2)
    assert r["recall@2"] == pytest.approx(2 / 3)
    assert r["mrr"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert [m["id"] for m in r["misses"]] == ["z"]


def test_empty_golden_rejected():
    with pytest.raises(ValueError):
        evaluate([], lambda q, k: [])


def test_sample_expansion_beats_baseline():
    d = load()
    res = run(d["corpus"], d["golden"], d["synonyms"], 3)
    assert res["expanded"]["recall@3"] > res["baseline"]["recall@3"]
    assert res["expanded"]["mrr"] >= res["baseline"]["mrr"]


def test_render_lists_misses():
    d = load()
    text = render(run(d["corpus"], d["golden"], None, 3))
    assert "Recall" not in text and "recall@3" in text
    assert "Misses (baseline)" in text


def test_cli_min_recall_gate(capsys):
    assert main([str(SAMPLE), "--min-recall", "0.1"]) == 0
    assert main([str(SAMPLE), "--min-recall", "1.01"]) == 1
    assert "recall@3" in capsys.readouterr().out
