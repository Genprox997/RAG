"""
Document ingestion: load -> chunk -> embed -> build hybrid index (FAISS + BM25).
"""
import datetime
import json
import os
import pickle
import re
from dataclasses import dataclass, asdict

import faiss
import numpy as np

import config
from src import llm
from src.tokenize import tokenize

# ---------- token counting (tiktoken with fallback) ----------
try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except Exception:  # pragma: no cover
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)


@dataclass
class Chunk:
    chunk_id: int
    text: str
    source: str
    n_tokens: int


# ---------- loaders ----------
def _load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ---------- PDF text extraction (quality-first, with fallbacks) ----------
# PyMuPDF (fitz) gives the best text recovery for complex / multi-column /
# CJK-heavy scientific PDFs; pdfplumber and pypdf are fallbacks; an optional
# OCR path (paddleocr) handles scanned documents when installed.
def _is_sparse(text: str, threshold: int = 30) -> bool:
    return len((text or "").strip()) < threshold


def _pdf_text_fitz(path: str) -> str:
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    try:
        return "\n".join(page.get_text("text") for page in doc)
    finally:
        doc.close()


def _pdf_text_pdfplumber(path: str) -> str:
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        return "\n".join((p.extract_text() or "") for p in pdf.pages)


def _pdf_text_pypdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        raise RuntimeError("pypdf not installed; `pip install pypdf` to ingest PDFs.")
    reader = PdfReader(path)
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def _pdf_text_ocr(path: str) -> str:
    """Optional OCR fallback for scanned PDFs. Needs paddleocr; silent no-op otherwise."""
    try:
        from paddleocr import PaddleOCR
    except Exception:
        return ""
    import numpy as np

    ocr = PaddleOCR(use_angle_cls=True, lang="ch")
    doc = fitz.open(path)
    try:
        pages = []
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:  # RGBA -> RGB
                img = img[..., :3]
            result = ocr.ocr(img, cls=True) or []
            lines = [txt for block in result for _, (txt, _) in block]
            pages.append("\n".join(lines))
        return "\n".join(pages)
    finally:
        doc.close()


def _load_pdf(path: str) -> str:
    """Extract PDF text: fitz -> pdfplumber -> pypdf -> OCR (scanned only)."""
    # 1) primary: PyMuPDF — best for complex layouts / CJK / multi-column
    try:
        text = _pdf_text_fitz(path)
        if not _is_sparse(text):
            return text
    except Exception as e:  # fitz import/runtime failure
        print(f"[warn] fitz failed {path}: {e}")
        text = ""

    # 2) pdfplumber fallback
    try:
        pl_text = _pdf_text_pdfplumber(path)
        if not _is_sparse(pl_text):
            return pl_text
    except Exception:
        pass

    # 3) original pypdf fallback
    try:
        py_text = _pdf_text_pypdf(path)
        if not _is_sparse(py_text):
            return py_text
    except Exception:
        pass

    # 4) last resort: OCR for scanned docs (no-op if paddleocr missing)
    return _pdf_text_ocr(path)


def load_documents(docs_dir: str) -> list[dict]:
    """Return list of {'source', 'text'} from .txt/.md/.pdf under docs_dir."""
    docs: list[dict] = []
    if not os.path.isdir(docs_dir):
        return docs
    for root, _, files in os.walk(docs_dir):
        for fn in sorted(files):
            path = os.path.join(root, fn)
            lower = fn.lower()
            try:
                if lower.endswith((".txt", ".md", ".markdown")):
                    text = _load_txt(path)
                elif lower.endswith(".pdf"):
                    text = _load_pdf(path)
                else:
                    continue
                if text.strip():
                    docs.append({"source": fn, "text": text})
            except Exception as e:  # skip bad files, keep pipeline alive
                print(f"[warn] skip {path}: {e}")
    return docs


# ---------- chunking (recursive, token-aware, with overlap) ----------
_SEPARATORS = ["\n\n", "\n", "。", "；", "；", ". ", "? ", "! ", " ", ""]


def _recursive_split(text: str, max_tokens: int, seps: list[str]) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text] if text.strip() else []
    if not seps:
        # hard split by characters
        step = max(1, max_tokens * 3)
        return [text[i : i + step] for i in range(0, len(text), step)]
    sep = seps[0]
    parts = text.split(sep) if sep else list(text)
    out: list[str] = []
    buf = ""
    for p in parts:
        cand = (buf + sep + p) if buf else p
        if count_tokens(cand) <= max_tokens:
            buf = cand
        else:
            if buf:
                out.extend(_recursive_split(buf, max_tokens, seps[1:]))
            # the piece itself may be too big -> recurse
            out.extend(_recursive_split(p, max_tokens, seps[1:]))
            buf = ""
    if buf:
        out.extend(_recursive_split(buf, max_tokens, seps[1:]))
    return out


def chunk_text(text: str, max_tokens: int = 400, overlap: int = 80) -> list[str]:
    pieces = _recursive_split(text, max_tokens, _SEPARATORS)
    if overlap <= 0 or len(pieces) <= 1:
        return [p for p in pieces if p.strip()]
    merged: list[str] = []
    for p in pieces:
        if merged:
            prev = merged[-1]
            # append tail of previous chunk as overlap context
            prev_tokens = count_tokens(prev)
            if prev_tokens > overlap:
                tail = prev[-overlap * 4 :]
                p = tail + "\n" + p
        merged.append(p)
    return [m.strip() for m in merged if m.strip()]


def build_chunks(docs: list[dict], max_tokens: int = 400, overlap: int = 80) -> list[Chunk]:
    chunks: list[Chunk] = []
    for d in docs:
        for piece in chunk_text(d["text"], max_tokens, overlap):
            chunks.append(
                Chunk(
                    chunk_id=len(chunks),
                    text=piece,
                    source=d["source"],
                    n_tokens=count_tokens(piece),
                )
            )
    return chunks


# ---------- index build & persist ----------
def _build_meta(chunks: list[Chunk], dim: int) -> dict:
    """Build index metadata so loading can validate compatibility."""
    s = config.get_settings()
    return {
        "version": 1,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "embed_provider": s.embed_provider,
        "embed_model": s.embed_model,
        "embed_dim": dim,
        "faiss_metric": "IP",
        "index_type": "IndexFlatIP",
        "tokenizer": "cjk_bigram",
        "chunk_count": len(chunks),
    }


def build_and_save(chunks: list[Chunk], index_dir: str | None = None) -> dict:
    index_dir = index_dir or config.get_settings().index_dir
    os.makedirs(index_dir, exist_ok=True)

    texts = [c.text for c in chunks]
    embeddings = llm.embed(texts, task="retrieval.passage")
    arr = np.asarray(embeddings, dtype="float32")
    # L2-normalize for cosine (inner product on normalized vectors == cosine)
    faiss.normalize_L2(arr)

    dim = arr.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(arr)

    # BM25
    from rank_bm25 import BM25Okapi

    corpus_tokens = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(corpus_tokens)

    # persist
    faiss.write_index(index, os.path.join(index_dir, "vectors.faiss"))
    with open(os.path.join(index_dir, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)
    with open(os.path.join(index_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)
    with open(os.path.join(index_dir, "index_meta.json"), "w", encoding="utf-8") as f:
        json.dump(_build_meta(chunks, dim), f, ensure_ascii=False, indent=2)

    return {
        "n_chunks": len(chunks),
        "dim": dim,
        "index_dir": index_dir,
    }


def ingest(docs_dir: str | None = None, index_dir: str | None = None) -> dict:
    docs_dir = docs_dir or config.get_settings().docs_dir
    docs = load_documents(docs_dir)
    if not docs:
        raise RuntimeError(f"No documents found in {docs_dir}")
    chunks = build_chunks(docs)
    return build_and_save(chunks, index_dir)


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass
    print(ingest())
