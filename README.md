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
| **重排** | 云端 Cross-Encoder（Jina/Cohere）/ **本地离线重排** | 两阶段检索范式 |
| **Agentic** | Query 改写/分解 → 多步检索 → LLM 自省（解析失败重试 / confidence 阈值早停 / 空检索降级） | Self-RAG 思路 |
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

数据流向：`data/docs/*` → 切分 → Embedding（本地 fastembed / 云端）→ `data/index/`（FAISS + BM25 + chunks.json + **index_meta.json**）；PDF 优先用 **PyMuPDF** 抽取，稀疏时回退 pdfplumber / pypdf，扫描件可选 OCR。

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

**缓存（降本提速，离线可用）**：`src/cache.py` 对 embedding 与 LLM 补全按内容哈希做本地缓存，默认走零依赖的文件缓存（`data/cache/`），若安装 `diskcache` 可设 `CACHE_BACKEND=diskcache` 提速。重复查询 / 重新入库会自动命中缓存，跳过昂贵调用。可用 `ENABLE_CACHE=false` 关闭。

**分阶段可观测**：`AgentResult.stats` 记录各阶段耗时（plan / retrieve / rerank / reflect / total，单位 ms）与 LLM 调用次数、prompt/completion token 估算（tiktoken，含回退）；`HybridIndex.timings` 记录单次检索内 vector / bm25 / rrf / rerank 子阶段耗时。`run.py` 会打印这些指标，便于定位瓶颈、写进简历「优化点」。

可选：开启重排：
- **云端**：在 `.env` 设置 `RERANK_PROVIDER=jina`（或 `cohere`）并填入 `RERANK_API_KEY`。
- **本地离线（推荐，全离线、零额外下载）**：设置 `RERANK_PROVIDER=local`，默认用已缓存的 bge 编码器对候选做 query-doc 余弦重排（`LOCAL_RERANKER=embedding`）；若已安装 `sentence_transformers`，可设 `LOCAL_RERANKER=cross-encoder` 启用 `bge-reranker-v2-m3` 真·交叉编码器（不可用时自动回退到 embedding 后端）。

> **离线 Embedding（推荐，免 API 费用）**：默认走云端 Embedding（需 `EMBED_API_KEY`）。
> 也可在 `.env` 设置 `EMBED_PROVIDER=local`，改用本地 `fastembed` 模型（`BAAI/bge-small-zh-v1.5`，512 维，中文优化），无需联网即可建库。
> 注意：切换到 local 后 `EMBED_DIM` 需与模型一致（如 512）。

> **Agentic 自省调参（`.env`）**：`MAX_ITERATIONS`（默认 3）控制多步检索上限；`REFLECT_CONFIDENCE_THRESHOLD`（默认 0.8）为连续置信度阈值，达到即早停；`MAX_SUB_QUERIES`（默认 4）为子查询数量安全上限（由 LLM 自行决定数量，截断保护）。反射 JSON 解析失败会自动重试一次，仍失败则保守补充检索而非过早停止；首轮检索为空时优雅降级返回空上下文而不崩溃。

---

## 📊 评测

基于 `evaluation/golden_set.json` 的 golden set，跑 LLM-as-Judge 评测：

```bash
python -m src.eval
```
输出各问题及聚合指标：
- **LLM-judged**（均 0–1）：Faithfulness / Answer Relevancy / Context Relevance。
- **答案级 Critic**（`src/critic.py`）：对最终答案二次校验——检测**幻觉**（答案中无上下文支撑的断言/数字）与**漏引**（上下文有但答案遗漏的关键事实）；LLM 不可用时 `detect_unsupported_numbers` 仍做确定性数字校验。聚合输出 `critic_faithful_rate`。
- **检索级（确定性、可复现）**：recall@k / MRR / hit_rate —— 由 golden 项的 `relevant_sources` 与按秩返回的 source 列表计算，衡量检索本身的质量（需 golden 标注相关来源）。
- **引用可信校验（确定性，无需 LLM）**：`generator.validate_citations` 检查答案中每个 `[n]` 是否落在 `[1..n_chunks]` 内，越界引用被标注为「模型臆造」；`needs_abstain` 在上下文为空或答案无任何有效引用时**确定性 abstain**（明确说无法回答），而非依赖模型自觉。评测聚合输出 `citation_valid_rate` / `abstain_rate`，用户入口（run.py / app.py）默认走 `safe_answer` 安全生成。

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
│   ├── retrieval.py       # 混合检索 + RRF + 重排（云端/本地）
│   ├── rerank.py          # 本地离线重排器（embedding / cross-encoder 后端）
│   ├── agent.py           # Agentic 自省循环
│   ├── generator.py       # 带引用的生成
│   ├── cache.py           # 本地响应缓存（embedding / LLM，离线零依赖）
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

