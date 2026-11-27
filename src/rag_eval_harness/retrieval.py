"""A small BM25 retriever with optional query expansion. Pure standard library."""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an the of to in on for and or is are do does how can i my it we you what when "
    "with be by at from this that".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        if not docs:
            raise ValueError("corpus is empty")
        self.k1, self.b = k1, b
        self.ids = list(docs)
        self.tf = [Counter(tokenize(docs[i])) for i in self.ids]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.len) / len(self.len) or 1.0
        df: Counter[str] = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(self.ids)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def score(self, query_tokens: list[str]) -> list[tuple[str, float]]:
        out = []
        for i, doc_id in enumerate(self.ids):
            s = 0.0
            for t in query_tokens:
                f = self.tf[i].get(t, 0)
                if not f:
                    continue
                norm = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                s += self.idf[t] * f * (self.k1 + 1) / norm
            out.append((doc_id, s))
        # Stable: ties keep corpus order, so results are reproducible.
        return sorted(out, key=lambda x: -x[1])

    def search(self, query: str, k: int = 5, synonyms: dict[str, list[str]] | None = None) -> list[str]:
        tokens = tokenize(query)
        if synonyms:
            extra = []
            for t in tokens:
                extra.extend(tokenize(" ".join(synonyms.get(t, []))))
            tokens = tokens + extra
        return [d for d, s in self.score(tokens)[:k] if s > 0]
