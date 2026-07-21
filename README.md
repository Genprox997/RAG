# 🧠 Agentic RAG — Hybrid Retrieval + Self-Reflective Agent

一个用于 **简历 / 求职 AI·大模型岗位** 的检索增强生成（RAG）项目。
覆盖从工程实现到原理理解的完整链路：**混合检索 + 重排 + Agentic 自省 + 引用溯源 + 自动评测**。

> 全程手写模块化实现（不依赖 LangChain 等重型框架），便于展示对 RAG 内部机制的掌握。

---

## ✨ 核心特性

| 模块 | 技术点 | 简历亮点 |
|------|--------|----------|
| **切分** | 递归切分 + 重叠 + token 感知 | Chunking 策略意识 |
| **向量检索** | FAISS (IndexFlatIP, cosine) | 向量库底层原理 |
| **关键词检索** | BM25（中英混合分词） | 稀疏检索 + 中文支持 |
| **融合** | Reciprocal Rank Fusion (RRF) | 混合检索工程实现 |
| **重排** | 云端 Cross-Encoder（Jina/Cohere） | 两阶段检索范式 |
| **Agentic** | Query 改写/分解 → 多步检索 → LLM 自省 | Self-RAG 思路 |
| **生成** | 流式输出 + 强制引用 [n] | 引用溯源 / 可控生成 |
| **评测** | Faithfulness / Answer Relevancy / Context Relevance | RAGAS 式自动评测 |
| **界面** | Streamlit + 可展开检索轨迹 | 可演示 Demo |

---

## 🏗️ 架构

```
用户问题
   │
   ▼
┌─────────────────── Agentic Loop (src/agent.py) ───────────────────┐
│ 1. Query 改写 / 分解        (LLM → search_query + sub_queries)     │
│ 2. 多步混合检索             (对每个 query 走下面 retrieval 管道)     │
│ 3. 自省 Self-Reflection     (LLM 判断上下文是否足够)               │
│ 4. 不足则生成补充查询再检索  (直到充分 / 达到 MAX_ITERATIONS)        │
└───────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
        ┌─────── Hybrid Retrieval (src/retrieval.py) ───────┐
        │  Dense: FAISS 向量检索  ─┐                        │
        │  Sparse: BM25 关键词检索 ─┤→ RRF 融合 → (可选重排) │
        └──────────────────────────────────────────────────┘
                                   │ context (带引用)
                                   ▼
                         Generator (src/generator.py) → 流式答案 + 引用
                                   │
                                   ▼
                        Eval (src/eval.py) → 指标报告
```

数据流向：`data/docs/*` → 切分 → 云端 Embedding → `data/index/`（FAISS + BM25 + chunks.json）

---

## 🚀 快速开始

```bash
# 1. 准备环境
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. 配置 API（任意 OpenAI 兼容端点：OpenAI / DeepSeek / Qwen ...）
cp .env.example .env
# 编辑 .env 填入 LLM_API_KEY 等

# 3. 建立索引（读取 data/docs 下样例文档）
python -m src.ingestion

# 4. 命令行提问
python run.py "Agentic RAG 的自省循环包含哪些步骤？"

# 5. 启动 Web Demo
streamlit run app.py
```

可选：开启重排 — 在 `.env` 设置 `RERANK_PROVIDER=jina` 并填入 `RERANK_API_KEY`。

> **离线 Embedding（推荐，免 API 费用）**：默认走云端 Embedding（需 `EMBED_API_KEY`）。
> 也可在 `.env` 设置 `EMBED_PROVIDER=local`，改用本地 `fastembed` 模型（`BAAI/bge-small-zh-v1.5`，512 维，中文优化），无需联网即可建库。
> 注意：切换到 local 后 `EMBED_DIM` 需与模型一致（如 512）。

---

## 📊 评测

基于 `evaluation/golden_set.json` 的 golden set，跑 LLM-as-Judge 评测：

```bash
python -m src.eval
```
输出各问题及聚合指标（Faithfulness / Answer Relevancy / Context Relevance，均 0–1）。

---

## 📁 目录结构

```
.
├── app.py                 # Streamlit UI
├── run.py                 # 命令行入口
├── config.py              # 配置（读 .env）
├── requirements.txt
├── .env.example
├── src/
│   ├── llm.py             # LLM / Embedding 客户端（OpenAI 兼容 + 流式）
│   ├── tokenize.py        # 中英混合分词（BM25 用）
│   ├── ingestion.py       # 加载 / 切分 / 入库 / 持久化
│   ├── retrieval.py       # 混合检索 + RRF + 云端重排
│   ├── agent.py           # Agentic 自省循环
│   ├── generator.py       # 带引用的生成
│   └── eval.py            # RAGAS 式评测
├── data/
│   ├── docs/              # 待检索文档（含样例）
│   └── index/             # 生成的索引（运行时产生）
└── evaluation/
    └── golden_set.json    # 评测集
```

---


## 🔧 可扩展点（加分项）

- 接本地模型：用 Ollama / vLLM 替换云端端点实现离线 RAG。
- 多模态：用 layout 检测把 PDF 图表/表格纳入检索。
- 高级切分：按标题层级 + 表格感知切分。
- 评测深化：接入 RAGAS / 人工标注集做回归。

