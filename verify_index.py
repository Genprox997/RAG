"""索引校验 + 真实混合检索验证。"""
import sys, json, os
from collections import Counter

sys.path.insert(0, "D:/code/rag")
from dotenv import load_dotenv

load_dotenv("D:/code/rag/.env")

import faiss
from src.retrieval import HybridIndex

INDEX = "D:/code/rag/data/index"

# ---------- 1. 块数 / 来源 / 向量统计 ----------
with open(os.path.join(INDEX, "chunks.json"), "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("========== 索引校验 ==========")
print("总块数 (chunks):", len(chunks))
srcs = Counter(c.get("source", "?") for c in chunks)
print("来源文档数 (去重):", len(srcs))

idx = faiss.read_index(os.path.join(INDEX, "vectors.faiss"))
print("FAISS 向量数:", idx.ntotal, "| 维度:", idx.d)

# ---------- 2. 索引文件大小 ----------
print("\n--- 索引文件大小 ---")
total = 0
for fn in ["vectors.faiss", "bm25.pkl", "chunks.json"]:
    p = os.path.join(INDEX, fn)
    sz = os.path.getsize(p)
    total += sz
    print(f"  {fn:14s} {sz/1024/1024:7.2f} MB")
print(f"  合计        {total/1024/1024:7.2f} MB")

# ---------- 3. 真实检索验证 ----------
hi = HybridIndex(index_dir=INDEX)
queries = [
    "显微镜的成像原理",
    "CODE V 中如何进行优化设计",
    "GB/T 12085 标准对显微镜的要求",
]
for q in queries:
    print("\n========== 检索:", q, "==========")
    hits = hi.retrieve(q, top_k_final=5)
    if not hits:
        print("  (无召回)")
        continue
    for i, h in enumerate(hits, 1):
        snip = " ".join(h.text.split())[:110]
        print(f"{i}. 融合分={h.score:.4f}  向量={h.vector_score}  BM25={h.bm25_score}")
        print(f"   来源: {h.source}")
        print(f"   片段: {snip}")
