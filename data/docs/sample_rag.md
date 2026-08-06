# RAG 系统概览（样例文档）

本文件用于演示检索增强生成（RAG）的核心概念，便于评测集引用。

## 混合检索（Hybrid Retrieval）

混合检索同时执行**稠密向量检索**和 **BM25 稀疏检索**，再用 **RRF（Reciprocal Rank Fusion）** 融合排名。向量检索擅长语义匹配但容易漏掉关键词；BM25 对专有名词和 ID 稳健但无法理解语义。两者结合兼顾语义与关键词，从而提升整体召回率。

- 向量检索：基于 Embedding 的语义相似度（如 FAISS 上的余弦 / 内积）。
- 关键词检索：BM25 等稀疏方法，对术语、编号、公式稳定。
- 融合：RRF 按排名倒数加权合并两个召回列表，无需对分数做归一化。

## 重排（Reranking）

检索阶段为了速度使用轻量方法，召回的 top-k 并不精确。**重排阶段**用 **Cross-Encoder** 对 query 与每个候选文档联合打分并重新排序，把最相关文档排到前面。

常见的云端重排服务有 **Jina Reranker** 和 **Cohere Rerank**。重排属于两阶段检索范式的第二阶段：先粗排（cheap）再精排（expensive but accurate）。

## Agentic RAG 与自省循环（Self-Reflection）

Agentic RAG 把检索组织成一个带**自省（Reflection）**的循环：

1. **Query 改写与分解**：让 LLM 把问题改写成关键词丰富的检索式，复杂问题再拆分为若干子查询（sub-queries）。
2. **多步检索并合并**：对每个查询走混合检索管道，合并去重。
3. **自省判断**：让 LLM 判断已检索到的上下文是否足以回答问题。
4. **补充检索**：若不足，则生成针对性的后续查询再次检索，直到信息充分或达到最大**迭代**次数。

这种 Self-RAG 式的 critique-then-retrieve 思路能显著减少遗漏，提升难问题的命中率。

## 常用的 RAG 评测指标

- **Faithfulness（忠实度）**：答案中的论断是否都有上下文依据，衡量幻觉程度。
- **Answer Relevancy（答案相关性）**：答案是否切题、是否真正回答了问题。
- **Context Relevance / Context Precision（上下文相关性）**：检索到的内容是否与问题相关。
- 上述指标均可由 **LLM 作为评判者（LLM-as-Judge）** 自动打分，也可补充确定性的检索级指标（如 recall@k、MRR、hit rate）来衡量检索本身的质量。
