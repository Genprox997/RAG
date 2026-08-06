"""
Streamlit UI for the Hybrid + Agentic RAG system.
Run:  streamlit run app.py
"""
import os

import streamlit as st

from dotenv import load_dotenv

load_dotenv()

import config
from src.ingestion import ingest
from src.retrieval import HybridIndex
from src.agent import RAGAgent
from src.generator import (
    generate_stream,
    build_context_block,
    annotate_invalid_citations,
    ABSTAIN_MESSAGE,
)

st.set_page_config(page_title="Agentic RAG", page_icon="🧠", layout="wide")
s = config.get_settings()

# ---------------- sidebar ----------------
st.sidebar.title("🧠 Agentic RAG")
st.sidebar.caption("Hybrid Retrieval + Self-Reflective Agent")

with st.sidebar.expander("⚙️ Configuration", expanded=False):
    st.write(f"**LLM:** `{s.llm_model}` @ `{s.llm_base_url}`")
    st.write(f"**Embed:** `{s.embed_model}` (dim={s.embed_dim})")
    st.write(f"**Reranker:** `{s.rerank_provider}`")
    st.write(f"**Max iterations:** `{s.max_iterations}`")
    st.write(f"**Top-k final:** `{s.top_k_final}`")

if not s.llm_api_key:
    st.sidebar.error("未配置 LLM_API_KEY，请在 .env 中填写（参考 .env.example）。")

docs_dir = s.docs_dir
os.makedirs(docs_dir, exist_ok=True)

if st.sidebar.button("📥 重新入库 (Ingest docs)"):
    with st.sidebar.status("正在切分 + 向量化..."):
        try:
            info = ingest()
            st.sidebar.success(f"入库完成：{info['n_chunks']} 个块")
            st.session_state["index"] = None
        except Exception as e:
            st.sidebar.error(f"入库失败：{e}")

uploaded = st.sidebar.file_uploader("上传文档 (.txt/.md/.pdf)", type=["txt", "md", "pdf"])
if uploaded:
    dest = os.path.join(docs_dir, uploaded.name)
    with open(dest, "wb") as f:
        f.write(uploaded.getbuffer())
    st.sidebar.success(f"已保存 {uploaded.name}，点击上方「重新入库」。")

# ---------------- main ----------------
st.title("检索增强生成 · Agentic RAG Demo")


@st.cache_resource(show_spinner=False)
def get_index():
    if not os.path.exists(os.path.join(s.index_dir, "vectors.faiss")):
        return None
    return HybridIndex()


idx = get_index()
if idx is None:
    st.warning("还没有建立索引。请在左侧点击「重新入库」或上传文档。")
    st.stop()

agent = RAGAgent(idx)

question = st.text_input("❓ 输入你的问题", placeholder="例如：Agentic RAG 的自省循环包含哪些步骤？")

if st.button("🚀 提问", disabled=not question):
    # ---- stream the agent's thinking process live ----
    st.subheader("🧠 思考过程 (Agent Trace)")
    think_box = st.empty()
    lines: list[str] = []

    def _push(line: str):
        lines.append(line)
        think_box.markdown("\n".join(f"- {l}" for l in lines))

    result = None
    with st.spinner("Agent 检索中..."):
        for ev in agent.run_streaming(question):
            if ev["type"] == "plan":
                subs = "、".join(ev["sub_queries"]) if ev["sub_queries"] else "（无）"
                _push(f"📝 规划检索策略：`{ev['search_query']}` · 子查询：{subs}")
            elif ev["type"] == "retrieve":
                _push(
                    f"🔎 第 {ev['iteration']} 轮检索 `{ev['query']}` → "
                    f"命中 {ev['n_retrieved']} 块（top: {', '.join(ev['top_sources'])}）"
                )
            elif ev["type"] == "reflect":
                _push(f"🤔 自省：{ev['reflection']}")
                if ev["next_query"]:
                    _push(f"➡️ 补充查询：`{ev['next_query']}`")
            elif ev["type"] == "done":
                result = ev["result"]
    if result is None:  # should not happen, but be safe
        result = agent.run(question)

    # answer (streaming)
    st.subheader("💬 回答")
    if not result.context:
        st.info(ABSTAIN_MESSAGE)
    else:
        # buffer the stream so we can annotate any out-of-range citations
        _buf = []

        def _gen():
            for tok in generate_stream(question, result.context):
                _buf.append(tok)
                yield tok

        st.write_stream(_gen())
        annotated = annotate_invalid_citations("".join(_buf), len(result.context))
        if annotated != "".join(_buf):
            st.caption(annotated.split("\n", 1)[1] if "\n" in annotated else annotated)

    # sources
    st.subheader("📚 引用来源")
    for i, h in enumerate(result.context, 1):
        with st.expander(f"[{i}] {h.source}  (score={h.score:.3f}, bm25={h.bm25_score}, vec={h.vector_score})"):
            st.write(h.text)

    # full trace (collapsible, still available for inspection)
    with st.expander("🔍 完整检索轨迹 (结构化)", expanded=False):
        for step in result.trace:
            st.markdown(
                f"**Step {step.iteration}** — query: `{step.query}`  \n"
                f"检索到 {step.n_retrieved} 块 · top sources: {', '.join(step.top_sources)}"
            )
            if step.reflection:
                st.markdown(f"→ 自省: {step.reflection}")
                if step.next_query:
                    st.markdown(f"→ 补充查询: `{step.next_query}`")
            st.divider()
        st.caption(f"共执行 {result.iterations} 轮检索迭代。")
