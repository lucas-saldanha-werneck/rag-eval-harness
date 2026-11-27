# rag-eval-harness

Independent demo, not affiliated with FullStack (cliente não diz).

A small offline tool that scores a retriever against a golden question set and lists the questions it got wrong. I built it because the first thing I want to know on a RAG project is a number: how often does the right document show up in the top results? Without that number, every prompt or index change is a guess.

It uses only the Python standard library. The retriever is plain BM25 with an optional synonym map for query expansion. The metrics are Recall@k and MRR.

## Run it

```bash
uv sync
uv run rag-eval examples/sample.json            # report
uv run rag-eval examples/sample.json -k 5 --min-recall 0.8   # exit 1 if below
uv run pytest -q
```

The dataset is one JSON file with `corpus` (id to text), `golden` (question plus relevant doc ids) and optional `synonyms`. `examples/sample.json` is invented example data: ten short support-article snippets and ten questions. It is not from any real company.

## Sample output

```
- baseline: recall@3 = 0.50, MRR = 0.50 (10 questions)
- expanded: recall@3 = 0.90, MRR = 0.90 (10 questions)
Misses (expanded): q2 'Where can I download my invoice?' expected d2, got d5
```

One miss is worth reading. Question q2 says "invoice" and the document says "Invoices". The tokenizer does no stemming, so the two words never match. Plain BM25 on this sample fails on plurals, and the report shows it.

## What was tested

- 9 pytest tests, all offline: tokenizer, BM25 ranking, empty-input errors, the Recall@k and MRR arithmetic on a hand-computed case, the sample dataset, report text, and the `--min-recall` exit code.
- The same tests run in GitHub Actions on every push.

## What was not tested

- Real data. The corpus is ten invented snippets, so the numbers say nothing about any production system.
- The synonym map was written by me while looking at the same ten questions. The jump from 0.50 to 0.90 is therefore optimistic. A fair test needs held-out questions.
- No embeddings, no reranker, no LLM judge. Those need models or network access and were out of scope for a few hundred lines.
- Big corpora. The scorer loops over every document, which is fine for a demo and slow for real indexes.
- Python versions other than the one in CI and my machine.
