"""
P0 improvement self-tests (run with the project's managed venv python + pytest).
Each test imports the code it needs locally so the file can be extended per task
without breaking collection of unrelated tests.

Run a subset:  pytest tests/test_p0_improvements.py -k pdf
"""
import glob
import json
import os

# Capture the genuine llm functions at import time so cache tests can restore
# them even if an upstream test leaked a patched LLM.chat / LLM.embed.
import src.llm as _LLM_MODULE

_REAL_EMBED = _LLM_MODULE.embed
_REAL_CHAT = _LLM_MODULE.chat



def _first_text_pdf(root: str = "data/docs"):
    """Return a path to a PDF under root whose pypdf extraction is non-trivial."""
    from src.ingestion import _pdf_text_pypdf

    for path in sorted(glob.glob(os.path.join(root, "**", "*.pdf"), recursive=True)):
        try:
            if len((_pdf_text_pypdf(path) or "").strip()) > 100:
                return path
        except Exception:
            continue
    return None


# ----------------------------- P0-1: PDF extraction -----------------------------
def test_pdf_loader_nonempty():
    from src.ingestion import _load_pdf

    path = _first_text_pdf()
    assert path is not None, "no extractable PDF found in data/docs for testing"
    text = _load_pdf(path)
    assert isinstance(text, str) and len(text.strip()) > 0, f"fitz loader returned empty for {path}"


def test_pdf_loader_ge_pypdf():
    from src.ingestion import _load_pdf, _pdf_text_pypdf

    path = _first_text_pdf()
    assert path is not None
    fitz_len = len(_load_pdf(path).strip())
    pypdf_len = len(_pdf_text_pypdf(path).strip())
    # PyMuPDF should recover at least as much text as pypdf on a text-based PDF.
    assert fitz_len >= pypdf_len, (
        f"fitz extracted fewer chars ({fitz_len}) than pypdf ({pypdf_len}) for {path}"
    )


# ------------------- P0-2: index metadata + dimension validation -------------------
def _write_synthetic_index(tmp, dim, meta_dim=None, provider="local"):
    """Write a minimal valid index (FAISS+BM25+chunks+meta) into tmp dir."""
    import faiss
    import numpy as np
    import pickle

    from rank_bm25 import BM25Okapi

    arr = np.random.rand(5, dim).astype("float32")
    faiss.normalize_L2(arr)
    idx = faiss.IndexFlatIP(dim)
    idx.add(arr)
    faiss.write_index(idx, os.path.join(tmp, "vectors.faiss"))

    bm = BM25Okapi([["a", "b"]] * 5)
    with open(os.path.join(tmp, "bm25.pkl"), "wb") as f:
        pickle.dump(bm, f)

    chunks = [{"chunk_id": i, "text": f"unique chunk text {i} 光学设计", "source": f"d{i}", "n_tokens": 1} for i in range(5)]
    with open(os.path.join(tmp, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    meta = {
        "version": 1,
        "embed_provider": provider,
        "embed_model": "m",
        "embed_dim": meta_dim if meta_dim is not None else dim,
        "faiss_metric": "IP",
        "index_type": "IndexFlatIP",
        "tokenizer": "cjk_bigram",
        "chunk_count": 5,
    }
    with open(os.path.join(tmp, "index_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    return chunks


def test_build_meta_fields():
    from src.ingestion import Chunk, _build_meta

    chunks = [Chunk(chunk_id=i, text="t", source=f"s{i}", n_tokens=3) for i in range(4)]
    meta = _build_meta(chunks, dim=512)
    assert meta["embed_dim"] == 512
    assert meta["chunk_count"] == 4
    assert meta["tokenizer"] == "cjk_bigram"
    assert meta["index_type"] == "IndexFlatIP"


def test_index_meta_validation_match_ok(tmp_path):
    from src.retrieval import HybridIndex

    _write_synthetic_index(str(tmp_path), dim=8, provider="local")
    hi = HybridIndex(str(tmp_path))  # must not raise
    assert hi.index.d == 8
    assert hi.meta["embed_dim"] == 8


def test_index_meta_validation_mismatch_raises(tmp_path):
    import pytest

    from src.retrieval import HybridIndex

    # meta claims dim 16 but the FAISS index is actually dim 8 -> corruption guard
    _write_synthetic_index(str(tmp_path), dim=8, meta_dim=16, provider="local")
    with pytest.raises(RuntimeError):
        HybridIndex(str(tmp_path))


def test_vector_search_dim_guard(monkeypatch, tmp_path):
    import pytest

    import src.retrieval as R
    from src.retrieval import HybridIndex

    _write_synthetic_index(str(tmp_path), dim=8, provider="local")
    hi = HybridIndex(str(tmp_path))
    # simulate a query embedding whose dim (4) != index dim (8)
    monkeypatch.setattr(R.llm, "embed", lambda texts, task=None: [[0.0] * 4])
    with pytest.raises(RuntimeError):
        hi._vector_search("anything", 3)


# ------------------- P0-3: Chinese BM25 tokenizer (CJK bigram) -------------------
def test_tokenize_bigram_emits_phrase_tokens():
    from src.tokenize import tokenize

    toks = tokenize("检索增强生成")
    assert "检索" in toks and "增强" in toks, "CJK bigrams must be produced"
    # unigrams still present
    assert {"检", "索", "增", "强", "生", "成"}.issubset(set(toks))


def test_tokenize_phrase_overlap():
    from src.tokenize import tokenize

    doc = tokenize("混合检索提升召回率")
    q = tokenize("什么是混合检索")
    overlap = set(doc) & set(q)
    # the bigram "检索" should be shared, giving BM25 a phrase-level signal
    assert "检索" in overlap


def test_tokenize_english_and_mixed():
    from src.tokenize import tokenize

    toks = tokenize("CODE V 光学设计 MicroLED 2024")
    assert "code" in toks and "v" in toks and "microled" in toks and "2024" in toks
    # mixed CJK term yields bigram
    assert "设计" in toks


def test_tokenize_bm25_recall_improves():
    from rank_bm25 import BM25Okapi

    from src.tokenize import tokenize

    # 3-doc corpus so singleton query terms get positive IDF (with only 2 docs,
    # a term appearing once yields IDF=log(1)=0 and a zero score).
    corpus = [
        tokenize("混合检索提升召回率"),
        tokenize("图像处理与滤波技术"),
        tokenize("天气预报与温度变化"),
    ]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize("混合检索方法"))
    assert scores[0] > 0, "on-topic doc must score > 0"
    assert scores[0] > scores[1] and scores[0] > scores[2], (
        "bigram tokenizer should rank the on-topic doc first"
    )


# ------------------- P0-4: retrieval-level evaluation metrics -------------------
def test_recall_at_k_basic():
    from src.metrics import recall_at_k

    rel = ["A"]
    retrieved = ["B", "A", "C"]
    assert recall_at_k(rel, retrieved, 1) == 0.0
    assert recall_at_k(rel, retrieved, 3) == 1.0


def test_mrr_basic():
    from src.metrics import mrr

    assert mrr(["A"], ["B", "A", "C"]) == 0.5
    assert mrr(["A"], ["B", "C", "D"]) == 0.0


def test_hit_rate_basic():
    from src.metrics import hit_rate

    rel = ["A"]
    assert hit_rate(rel, ["B", "C", "A"], 2) == 0.0
    assert hit_rate(rel, ["B", "C", "A"], 5) == 1.0


def test_empty_relevant_returns_zero_or_empty():
    from src.metrics import recall_at_k, retrieval_metrics

    assert recall_at_k([], ["A", "B"], 3) == 0.0
    assert retrieval_metrics([], ["A", "B"]) == {}


def test_retrieval_metrics_bundle():
    from src.metrics import retrieval_metrics

    out = retrieval_metrics(["A"], ["B", "A", "C"], ks=[1, 3, 5])
    assert out["recall@1"] == 0.0
    assert out["recall@3"] == 1.0
    assert out["recall@5"] == 1.0
    assert out["mrr"] == 0.5
    assert out["hit_rate@5"] == 1.0


def test_eval_wiring_with_fake_hits():
    """Mirror evaluate_item's retrieval-metric wiring without the LLM/index."""
    from types import SimpleNamespace

    from src.metrics import retrieval_metrics

    # fake retrieved chunks (only .source is used, as in eval.py)
    hits = [SimpleNamespace(source=s) for s in ["B", "A", "C"]]
    retrieved_sources = [h.source for h in hits]
    ks = sorted(set([1, 3, 5, 6]))
    metrics = retrieval_metrics(["A"], retrieved_sources, ks=ks)
    assert metrics["recall@1"] == 0.0
    assert metrics["recall@3"] == 1.0
    assert metrics["mrr"] == 0.5


# ------------------- P1-1: local offline reranker -------------------
def test_rerank_ordering_with_injected_scorer():
    from src.rerank import LocalReranker

    docs = ["关于苹果公司的财报", "足球比赛的战术分析", "苹果手机的最新评测"]
    reranker = LocalReranker(scorer=lambda q, d: 1.0 if "苹果" in d else 0.0)
    ranked, scores = reranker.rerank("苹果", docs)
    assert ranked[0] in ("关于苹果公司的财报", "苹果手机的最新评测")
    assert "足球" not in ranked[0]
    assert scores[0] >= scores[-1]


def test_rerank_cross_encoder_missing_falls_back(monkeypatch):
    import sys

    import src.llm as LLM
    from src.rerank import LocalReranker

    # Force sentence_transformers to look "missing" so cross-encoder backend
    # must fall back to the embedding backend (fully offline).
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    def fake_embed(texts, task=None):
        return [[1.0, 0.0] if "苹果" in t else [0.0, 1.0] for t in texts]

    monkeypatch.setattr(LLM, "embed", fake_embed)
    reranker = LocalReranker(backend="cross-encoder")
    docs = ["关于苹果公司的财报", "足球比赛的战术分析", "苹果手机的最新评测"]
    ranked, _ = reranker.rerank("苹果", docs)
    assert ranked[0] in ("关于苹果公司的财报", "苹果手机的最新评测")


def test_retrieve_routes_to_local_rerank(monkeypatch, tmp_path):
    import src.rerank as RR
    import src.retrieval as R
    from src.retrieval import HybridIndex

    _write_synthetic_index(str(tmp_path), dim=8, provider="local")
    monkeypatch.setattr(R.llm, "embed", lambda texts, task=None: [[1.0] * 8 for _ in texts])
    hi = HybridIndex(str(tmp_path))

    # baseline order under RRF only
    hi.s.rerank_provider = "none"
    base = [h.source for h in hi.retrieve("q")]

    # route to local rerank via a fake reranker that reverses candidate order
    class _FakeReranker:
        def __init__(self, *a, **k):
            pass

        def rerank(self, query, documents, top_n=None):
            rev = list(reversed(documents))
            return rev, [float(len(rev) - i) for i in range(len(rev))]

    monkeypatch.setattr(RR, "LocalReranker", _FakeReranker)
    hi.s.rerank_provider = "local"
    reranked = [h.source for h in hi.retrieve("q")]
    assert reranked == list(reversed(base)), (reranked, base)


# ------------------- P1-2: robust agentic self-reflection -------------------
class _ScriptedChat:
    """Replay a list of responses for src.llm.chat (records call count)."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def __call__(self, messages, json_mode=False):
        self.calls += 1
        return self._responses.pop(0)


def _make_agent(tmp_path, chat):
    import src.llm as LLM
    import src.retrieval as R
    from src.agent import RAGAgent
    from src.retrieval import HybridIndex

    _write_synthetic_index(str(tmp_path), dim=8, provider="local")
    # retrieval needs embeddings; provide deterministic vectors (no network)
    LLM.embed = lambda texts, task=None: [[1.0] * 8 for _ in texts]
    R.llm.embed = LLM.embed
    LLM.chat = chat
    hi = HybridIndex(str(tmp_path))
    hi.s.reflect_confidence_threshold = 0.8
    hi.s.max_sub_queries = 4
    return RAGAgent(hi), hi


def test_agent_dynamic_sub_queries_count(monkeypatch, tmp_path):
    from src.agent import RAGAgent

    plan = '{"search_query":"光学设计","sub_queries":["像差校正","MTF评价","CODE V优化","第四个保留","第五个应被截断"]}'
    refl = '{"sufficient":true,"confidence":0.95,"gap":"","next_query":""}'
    chat = _ScriptedChat([plan, refl])
    agent, _ = _make_agent(tmp_path, chat)
    res = agent.run("光学设计基础")
    # LLM emitted 4 sub-queries; cap is 4 so all kept, the 5th would be dropped
    q0 = res.trace[0].query
    for sub in ["光学设计", "像差校正", "MTF评价", "CODE V优化"]:
        assert sub in q0, q0
    assert "第五个应被截断" not in q0, "sub_queries must be capped at max_sub_queries"


def test_agent_confidence_threshold_early_stop(monkeypatch, tmp_path):
    plan = '{"search_query":"q","sub_queries":[]}'
    refl = '{"sufficient":false,"confidence":0.9,"gap":"需要更多细节","next_query":"补充查询"}'
    chat = _ScriptedChat([plan, refl])
    agent, _ = _make_agent(tmp_path, chat)
    res = agent.run("某问题")
    # confidence 0.9 >= threshold 0.8 -> stop after first reflection
    assert res.iterations == 1, res.trace
    assert res.trace[0].reflection.startswith("insufficient(conf=0.90)")


def test_agent_parse_failure_retry_then_fallback(monkeypatch, tmp_path):
    plan = '{"search_query":"q","sub_queries":[]}'
    garbage1 = "I cannot output JSON here."          # initial reflection
    garbage2 = "still not json"                        # strict retry
    ok = '{"sufficient":true,"confidence":1.0,"gap":"","next_query":""}'
    chat = _ScriptedChat([plan, garbage1, garbage2, ok])
    agent, _ = _make_agent(tmp_path, chat)
    res = agent.run("某问题")  # must not raise / must not prematurely stop
    assert res.iterations == 2, res.trace
    # both initial + strict-retry reflection calls happened (plan + 2 fails + ok)
    assert chat.calls >= 4
    assert len(res.context) > 0


def test_agent_empty_retrieval_graceful(monkeypatch, tmp_path):
    plan = '{"search_query":"q","sub_queries":[]}'
    chat = _ScriptedChat([plan])
    agent, hi = _make_agent(tmp_path, chat)
    hi.retrieve = lambda q: []  # force empty retrieval
    res = agent.run("某问题")     # must not raise
    assert res.empty_retrieval is True
    assert len(res.context) == 0
    assert res.trace[-1].reflection == "no chunks retrieved"
    assert res.iterations == 0


# ------------------- P1-3: answer-level critic (hallucination / omission) -----
def test_critic_detect_unsupported_number():
    from src.critic import detect_unsupported_numbers

    ctx = "实验显示准确率为 87%，样本量 1200。"
    # answer invents a different number -> should be flagged
    bad = "实验显示准确率为 99.5%，样本量 1200。"
    flagged = detect_unsupported_numbers(bad, ctx)
    assert "99.5" in flagged
    assert "1200" not in flagged  # grounded number is fine
    # fully grounded answer -> nothing flagged
    ok = "准确率是 87%。"
    assert detect_unsupported_numbers(ok, ctx) == []


def test_critic_llm_wiring_finds_hallucination_and_omission():
    from src.critic import critic

    ctx = "光学系统用 MTF 评价成像质量；CODE V 用于优化设计。"
    ans = "MTF 可评价成像质量（上下文有），但声称系统成本为 500 万元（编造）。"

    def fake_chat(messages, json_mode=False):
        # ignore the prompt, return a scripted judgment
        return (
            '{"faithful": false, '
            '"hallucinated": ["系统成本为 500 万元"], '
            '"missing": ["CODE V 用于优化设计这一事实未被引用"], '
            '"issues": ["answer invents a cost not in context"]}'
        )

    r = critic("光学系统如何评价", ans, ctx, chat_fn=fake_chat)
    assert r.faithful is False
    assert any("500" in h for h in r.hallucinated)
    assert any("CODE V" in m for m in r.missing)


def test_eval_includes_critic_fields(monkeypatch, tmp_path):
    import src.llm as LLM
    import src.retrieval as R
    from src.agent import RAGAgent
    from src.eval import evaluate_item
    from src.retrieval import HybridIndex

    _write_synthetic_index(str(tmp_path), dim=8, provider="local")
    LLM.embed = lambda texts, task=None: [[1.0] * 8 for _ in texts]
    R.llm.embed = LLM.embed

    # scripted chat: plan, reflection(sufficient), answer, critic
    script = [
        '{"search_query":"光学","sub_queries":[]}',
        '{"sufficient":true,"confidence":0.95,"gap":"","next_query":""}',
        "MTF 用于评价成像质量。",  # generated answer
        '{"faithful": true, "hallucinated": [], "missing": [], "issues": []}',  # critic
    ]
    calls = {"n": 0}

    def fake_chat(messages, json_mode=False):
        i = calls["n"]
        calls["n"] += 1
        return script[i]

    LLM.chat = fake_chat

    hi = HybridIndex(str(tmp_path))
    agent = RAGAgent(hi)
    row = evaluate_item(agent, "什么是 MTF", relevant_sources=["d0"])
    assert "critic_faithful" in row
    assert row["critic_faithful"] is True
    assert row["critic_hallucinated"] == []
    assert "retrieval" in row and "recall@1" in row["retrieval"]


# ------------------- P1-4: citation validation + deterministic abstain ---------
def test_validate_citations_flags_out_of_range():
    from src.generator import validate_citations

    v = validate_citations("根据[1]与[5]可知结论。", n_chunks=3)
    assert v["has_citation"] is True
    assert v["invalid"] == [5]
    assert v["valid"] is False
    # all-in-range answer is valid
    ok = validate_citations("见[1]与[2]。", n_chunks=3)
    assert ok["valid"] is True and ok["invalid"] == []


def test_needs_abstain_cases():
    from src.generator import needs_abstain, validate_citations

    # empty context -> must abstain
    assert needs_abstain("任何内容", n_chunks=0) is True
    # no citation at all with non-empty context -> abstain
    assert needs_abstain("光学系统用MTF评价。", n_chunks=2) is True
    # valid citation present -> do not abstain
    assert needs_abstain("MTF评价成像质量[1]。", n_chunks=2) is False
    # out-of-range citation still counts as "has citation" (not abstain, but invalid)
    assert validate_citations("结论[9]。", n_chunks=2)["has_citation"] is True


def test_safe_answer_abstains_and_annotates(monkeypatch):
    import src.generator as G

    # case A: model returns an uncited answer -> deterministic abstain
    monkeypatch.setattr(G, "generate", lambda q, ctx: "光学系统用MTF评价成像质量。")
    abstained = G.safe_answer("什么是MTF", [object()])
    assert abstained == G.ABSTAIN_MESSAGE

    # case B: valid citation -> returned as-is
    monkeypatch.setattr(G, "generate", lambda q, ctx: "MTF评价成像质量[1]。")
    cited = G.safe_answer("什么是MTF", [object()])
    assert cited == "MTF评价成像质量[1]。"

    # case C: out-of-range citation -> warning annotated, not abstained
    monkeypatch.setattr(G, "generate", lambda q, ctx: "结论[9]。")
    annotated = G.safe_answer("结论是什么", [object()])
    assert "[9]" in annotated and "超出上下文范围" in annotated

    # case D: empty context -> abstain even if generate would say something
    monkeypatch.setattr(G, "generate", lambda q, ctx: "凭空编造[1]。")
    assert G.safe_answer("x", []) == G.ABSTAIN_MESSAGE


# ------------------- P2-1: local response cache (embeddings + LLM) -------------------
def test_cache_roundtrip(tmp_path):
    from src.cache import ResponseCache

    c = ResponseCache(cache_dir=str(tmp_path), enabled=True)
    assert c.get("embed", (["a"],)) is None
    c.set("embed", (["a"],), [[0.1, 0.2]])
    assert c.get("embed", (["a"],)) == [[0.1, 0.2]]


def test_cache_miss_then_hit_counts_compute(tmp_path):
    from src.cache import ResponseCache

    c = ResponseCache(cache_dir=str(tmp_path), enabled=True)
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return "result"

    assert c.cached("chat", ("same-prompt",), compute) == "result"
    assert c.cached("chat", ("same-prompt",), compute) == "result"
    assert calls["n"] == 1, "second call must hit cache, not recompute"
    # different key -> recompute
    assert c.cached("chat", ("other-prompt",), compute) == "result"
    assert calls["n"] == 2


def test_cache_disabled_never_stores(tmp_path):
    from src.cache import ResponseCache

    c = ResponseCache(cache_dir=str(tmp_path), enabled=False)
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return "x"

    assert c.cached("chat", ("k",), compute) == "x"
    assert c.cached("chat", ("k",), compute) == "x"
    assert calls["n"] == 2, "disabled cache must always recompute"
    assert c.get("chat", ("k",)) is None


def test_llm_embed_is_cached(monkeypatch, tmp_path):
    import src.llm as LLM
    from src.cache import ResponseCache

    # neutralize any leaked LLM.chat/embed from upstream tests, then isolate cache
    monkeypatch.setattr(LLM, "embed", _REAL_EMBED)
    monkeypatch.setattr(LLM, "chat", _REAL_CHAT)
    cache = ResponseCache(cache_dir=str(tmp_path), enabled=True)
    monkeypatch.setattr(LLM, "default_cache", lambda: cache)

    calls = {"n": 0}

    def fake_embed_impl(texts, task=None):
        calls["n"] += 1
        return [[0.0] * 4 for _ in texts]

    monkeypatch.setattr(LLM, "_embed_impl", fake_embed_impl)
    texts = ["混合检索", "向量检索"]
    # two identical cached calls -> only one underlying embed
    LLM.embed(texts, task="retrieval.passage")
    LLM.embed(texts, task="retrieval.passage")
    assert calls["n"] == 1, "repeated identical embed must hit cache"
    # _cache=False bypasses cache
    LLM.embed(texts, task="retrieval.passage", _cache=False)
    assert calls["n"] == 2


def test_llm_chat_is_cached(monkeypatch, tmp_path):
    import src.llm as LLM
    from src.cache import ResponseCache

    # neutralize any leaked LLM.chat/embed from upstream tests, then isolate cache
    monkeypatch.setattr(LLM, "embed", _REAL_EMBED)
    monkeypatch.setattr(LLM, "chat", _REAL_CHAT)
    cache = ResponseCache(cache_dir=str(tmp_path), enabled=True)
    monkeypatch.setattr(LLM, "default_cache", lambda: cache)

    calls = {"n": 0}

    def fake_chat_impl(messages, *, temperature=0.0, max_tokens=None, json_mode=False):
        calls["n"] += 1
        return "answer"

    monkeypatch.setattr(LLM, "_chat_impl", fake_chat_impl)
    msgs = [{"role": "user", "content": "hi"}]
    LLM.chat(msgs, json_mode=True)
    LLM.chat(msgs, json_mode=True)
    assert calls["n"] == 1, "repeated identical chat must hit cache"
    # different json_mode -> different key -> recompute
    LLM.chat(msgs, json_mode=False)
    assert calls["n"] == 2


