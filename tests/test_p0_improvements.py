"""
P0 improvement self-tests (run with the project's managed venv python + pytest).
Each test imports the code it needs locally so the file can be extended per task
without breaking collection of unrelated tests.

Run a subset:  pytest tests/test_p0_improvements.py -k pdf
"""
import glob
import json
import os


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

    chunks = [{"chunk_id": i, "text": "x", "source": f"d{i}", "n_tokens": 1} for i in range(5)]
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


