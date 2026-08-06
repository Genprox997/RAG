"""
Offline, pluggable reranker for the hybrid retrieval pipeline.

Two backends:
- "embedding" (default): reuse the local dense encoder (bge) to score each
  (query, document) pair by cosine similarity. Fully offline, no new downloads,
  and it complements RRF with a single consistent relevance signal.
- "cross-encoder": if `sentence_transformers` is installed, use a real
  cross-encoder (default bge-reranker-v2-m3). Falls back to the embedding
  backend when the dependency or model is unavailable.

The cloud providers (jina / cohere) are handled separately in retrieval.py.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


def cosine(a, b) -> float:
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class LocalReranker:
    backend: str = "embedding"  # "embedding" | "cross-encoder"
    model_name: str = "BAAI/bge-reranker-v2-m3"
    scorer: Optional[Callable[[str, str], float]] = None  # injectable (tests/custom)
    _ce = None  # cached cross-encoder instance

    def _load_cross_encoder(self):
        if self._ce is not None:
            return self._ce
        try:
            from sentence_transformers import CrossEncoder
        except Exception:
            return None
        try:
            self._ce = CrossEncoder(self.model_name)
            return self._ce
        except Exception:
            return None

    def _default_scorer(self) -> Callable[[str, str], float]:
        """Build the scoring function for the active backend.

        Falls back to the embedding backend when cross-encoder is unavailable.
        """
        if self.backend == "cross-encoder":
            ce = self._load_cross_encoder()
            if ce is not None:
                def _ce_score(query: str, doc: str) -> float:
                    return float(ce.predict([(query, doc)])[0])

                return _ce_score
        # default: embedding backend (reuse locally cached dense encoder)
        from src import llm

        def _emb_score(query: str, doc: str) -> float:
            qv = llm.embed([query], task="retrieval.query")[0]
            dv = llm.embed([doc], task="retrieval.passage")[0]
            return cosine(qv, dv)

        return _emb_score

    def get_scorer(self) -> Callable[[str, str], float]:
        if self.scorer is not None:
            return self.scorer
        return self._default_scorer()

    def rerank(
        self, query: str, documents: list[str], top_n: Optional[int] = None
    ) -> tuple[list[str], list[float]]:
        score_fn = self.get_scorer()
        scored = [(doc, score_fn(query, doc)) for doc in documents]
        scored.sort(key=lambda x: x[1], reverse=True)
        if top_n is not None:
            scored = scored[:top_n]
        ranked = [doc for doc, _ in scored]
        scores = [float(s) for _, s in scored]
        return ranked, scores
