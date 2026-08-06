"""
Offline smoke check: exercises chunking -> hybrid retrieval -> agentic loop ->
generator prompt without any network / cloud API.
It monkeypatches the LLM embedder & chat with deterministic stubs so the
entire pipeline can be validated locally.

Run it as a script:  python tests/smoke_check.py

NOTE: this file is deliberately NOT named `smoke_test.py` / `test_smoke.py`.
It patches `src.llm.embed` / `src.llm.chat` at *module level*, so if pytest
collected it those stubs would leak into the whole session -- which is exactly
what happened: `tests/test_p0_improvements.py` then captured a stub as the
"real" llm.embed and the cache tests broke under `pytest tests/`. Keeping the
name outside pytest's `test_*.py` / `*_test.py` patterns keeps the suite clean.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import src.llm as llm_mod
from src import tokenize, ingestion
from src.retrieval import HybridIndex
from src.agent import RAGAgent
from src.generator import build_messages


# ---- deterministic fake embedder (bag-of-tokens, L2-normalized) ----
def fake_embed(texts, task=None):
    # `task` mirrors the real llm.embed signature (retrieval.query/passage).
    dim = 64
    out = []
    for t in texts:
        v = np.zeros(dim, dtype="float32")
        for tok in tokenize.tokenize(t):
            v[hash(tok) % dim] += 1.0
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        out.append(v.tolist())
    return out


llm_mod.embed = fake_embed


def fake_chat(messages, **kwargs):
    content = messages[-1]["content"]
    if "retrieval planner" in content:
        return '{"search_query":"rag reranking self reflection agent","sub_queries":[]}'
    if "evaluating whether retrieved context" in content:
        return '{"sufficient": true, "gap":"", "next_query":""}'
    return "This is a stub answer [1]."


llm_mod.chat = fake_chat
llm_mod.stream_chat = lambda m, **k: iter(["This ", "is ", "a ", "stub."])


def main():
    # 1) tokenize
    toks = tokenize.tokenize("RAG 检索增强生成 2024")
    assert any(t == "rag" for t in toks) and any(t == "检" for t in toks)

    # 2) chunk
    docs = ingestion.load_documents("data/docs")
    assert docs, "sample doc should load"
    chunks = ingestion.build_chunks(docs)
    assert len(chunks) > 1, "doc should be split into multiple chunks"

    # 3) build index (fake embed) in temp dir
    tmp = tempfile.mkdtemp()
    info = ingestion.build_and_save(chunks, tmp)
    assert info["n_chunks"] == len(chunks)

    # 4) hybrid retrieval
    idx = HybridIndex(tmp)
    hits = idx.retrieve("reranking 重排 cross-encoder")
    assert hits, "retrieval should return hits"
    assert hits[0].score > 0

    # 5) agentic loop
    agent = RAGAgent(idx)
    res = agent.run("Agentic RAG 的自省循环包含哪些步骤？")
    assert res.context, "agent should collect context"
    assert res.trace, "agent should record a trace"

    # 6) generator prompt
    msgs = build_messages("test question?", res.context)
    assert msgs[0]["role"] == "system" and "[1]" in msgs[0]["content"]

    print("SMOKE TEST OK")
    print(f"  chunks={len(chunks)}  hits={len(hits)}  "
          f"agent_iters={res.iterations}  ctx={len(res.context)}  trace={len(res.trace)}")


if __name__ == "__main__":
    main()
