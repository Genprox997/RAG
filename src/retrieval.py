"""
Hybrid retrieval: dense vector search (FAISS) + sparse keyword search (BM25),
fused with Reciprocal Rank Fusion (RRF), optionally re-ranked by a cloud
cross-encoder (Jina / Cohere).
"""
import json
import os
import time
import urllib.request
from dataclasses import dataclass

import faiss
import numpy as np

import config
from src import llm
from src.tokenize import tokenize


@dataclass
class Hit:
    chunk_id: int
    source: str
    text: str
    score: float
    vector_score: float = 0.0
    bm25_score: float = 0.0


class HybridIndex:
    def __init__(self, index_dir: str | None = None):
        s = config.get_settings()
        self.index_dir = index_dir or s.index_dir
        self.index = faiss.read_index(os.path.join(self.index_dir, "vectors.faiss"))
        import pickle

        with open(os.path.join(self.index_dir, "bm25.pkl"), "rb") as f:
            self.bm25 = pickle.load(f)
        with open(os.path.join(self.index_dir, "chunks.json"), "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.s = s
        self.meta = self._load_meta()
        self._validate()
        # per-call observability: last retrieve()'s sub-step timings (ms)
        self.timings: dict = {}

    def _load_meta(self) -> dict | None:
        path = os.path.join(self.index_dir, "index_meta.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _validate(self) -> None:
        """Validate index compatibility against recorded metadata + current config.

        Prevents the cryptic FAISS dimension-mismatch crash that used to happen when
        EMBED_PROVIDER (local 512-dim vs cloud 1536-dim) was switched without rebuilding.
        """
        if self.meta is None:
            return
        # 1) structural consistency: FAISS dim must equal recorded embed_dim
        if self.meta.get("embed_dim") != self.index.d:
            raise RuntimeError(
                f"Index corruption: meta embed_dim={self.meta.get('embed_dim')} "
                f"but FAISS index dim={self.index.d}. Rebuild the index."
            )
        # 2) provider/dim drift: current config would produce incompatible query vectors
        if self.s.embed_provider != self.meta.get("embed_provider") or (
            self.s.embed_provider == "cloud"
            and int(self.s.embed_dim) != self.meta.get("embed_dim")
        ):
            print(
                f"[warn] Index built with embed_provider={self.meta.get('embed_provider')} "
                f"model={self.meta.get('embed_model')} dim={self.meta.get('embed_dim')}, "
                f"but current config is embed_provider={self.s.embed_provider} "
                f"dim={self.s.embed_dim}. Rebuild the index (`python -m src.ingestion`) "
                f"to avoid runtime dimension mismatch."
            )

    # ---- vector search ----
    def _vector_search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = llm.embed([query], task="retrieval.query")[0]
        q = np.asarray([q], dtype="float32")
        if q.shape[1] != self.index.d:
            raise RuntimeError(
                f"Query embedding dim {q.shape[1]} != index dim {self.index.d}. "
                f"The index was built with a different embedder/provider. "
                f"Rebuild it with `python -m src.ingestion`."
            )
        faiss.normalize_L2(q)
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(q, k)
        return list(zip(ids[0].tolist(), scores[0].tolist()))

    # ---- bm25 search ----
    def _bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        toks = tokenize(query)
        if not toks:
            return []
        scores = self.bm25.get_scores(toks)
        order = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    # ---- RRF fusion ----
    @staticmethod
    def _rrf(ranked_lists: list[list[tuple[int, float]]], k: int) -> dict[int, float]:
        fused: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, (cid, _) in enumerate(ranked):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
        return fused

    def retrieve(self, query: str, top_k_final: int | None = None) -> list[Hit]:
        top_k_final = top_k_final or self.s.top_k_final
        t0 = time.perf_counter()
        vec = self._vector_search(query, self.s.top_k_vector)
        t1 = time.perf_counter()
        bm = self._bm25_search(query, self.s.top_k_bm25)
        t2 = time.perf_counter()
        fused = self._rrf([vec, bm], self.s.rrf_k)
        t3 = time.perf_counter()

        # carry individual scores for transparency
        vec_scores = {cid: sc for cid, sc in vec}
        bm_scores = {cid: sc for cid, sc in bm}

        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        candidates = ordered[: max(top_k_final * 4, 12)]

        # optional rerank
        if self.s.rerank_provider == "local":
            candidates = self._local_rerank(query, candidates)
        elif self.s.rerank_provider in ("jina", "cohere") and self.s.rerank_api_key:
            candidates = self._rerank(query, candidates)
        t4 = time.perf_counter()

        # record sub-step timings for observability (ms)
        self.timings = {
            "vector_ms": round((t1 - t0) * 1000, 3),
            "bm25_ms": round((t2 - t1) * 1000, 3),
            "rrf_ms": round((t3 - t2) * 1000, 3),
            "rerank_ms": round((t4 - t3) * 1000, 3),
        }

        hits: list[Hit] = []
        for cid, sc in candidates[:top_k_final]:
            c = self.chunks[cid]
            hits.append(
                Hit(
                    chunk_id=cid,
                    source=c["source"],
                    text=c["text"],
                    score=sc,
                    vector_score=round(vec_scores.get(cid, 0.0), 4),
                    bm25_score=round(bm_scores.get(cid, 0.0), 4),
                )
            )
        return hits

    # ---- cloud rerank ----
    def _rerank(self, query: str, candidates: list[tuple[int, float]]) -> list[tuple[int, float]]:
        docs = [self.chunks[cid]["text"] for cid, _ in candidates]
        try:
            scores = _cloud_rerank(
                provider=self.s.rerank_provider,
                api_key=self.s.rerank_api_key,
                model=self.s.rerank_model,
                query=query,
                documents=docs,
            )
            reranked = sorted(
                zip((cid for cid, _ in candidates), scores),
                key=lambda x: x[1],
                reverse=True,
            )
            return reranked
        except Exception as e:
            print(f"[warn] rerank failed, falling back to RRF: {e}")
            return candidates

    # ---- local offline rerank ----
    def _local_rerank(self, query: str, candidates: list[tuple[int, float]]) -> list[tuple[int, float]]:
        from collections import defaultdict

        from src.rerank import LocalReranker

        docs = [self.chunks[cid]["text"] for cid, _ in candidates]
        try:
            reranker = LocalReranker(backend=self.s.local_reranker)
            ranked_docs, scores = reranker.rerank(query, docs)
            # map reordered docs back to (chunk_id, score) by occurrence position,
            # so even duplicate chunk texts map to distinct candidates.
            occ: dict[int, list[int]] = defaultdict(list)
            for i, d in enumerate(docs):
                occ[id(d)].append(i)

            def _orig_idx(d: str) -> int:
                return occ[id(d)].pop(0)

            reranked = [
                (candidates[_orig_idx(d)][0], float(sc))
                for d, sc in zip(ranked_docs, scores)
            ]
            return reranked
        except Exception as e:
            print(f"[warn] local rerank failed, falling back to RRF: {e}")
            return candidates


def _cloud_rerank(provider: str, api_key: str, model: str, query: str, documents: list[str]) -> list[float]:
    if provider == "jina":
        url = "https://api.jina.ai/v1/rerank"
        payload = {"model": model, "query": query, "documents": documents, "top_n": len(documents)}
    elif provider == "cohere":
        url = "https://api.cohere.com/v1/rerank"
        payload = {"model": model, "query": query, "documents": documents, "top_n": len(documents)}
    else:
        raise ValueError(f"unknown rerank provider {provider}")

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    # both return a list with "index" + "relevance_score"
    out = [0.0] * len(documents)
    for r in data.get("results", data.get("data", [])):
        out[r["index"]] = float(r["relevance_score"])
    return out
