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
from src.generator import generate_stream, build_context_block

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
    with st.spinner("Agent 检索中..."):
        result = agent.run(question)

    # answer (streaming)
    st.subheader("💬 回答")
    st.write_stream(generate_stream(question, result.context))

    # sources
    st.subheader("📚 引用来源")
    for i, h in enumerate(result.context, 1):
        with st.expander(f"[{i}] {h.source}  (score={h.score:.3f}, bm25={h.bm25_score}, vec={h.vector_score})"):
            st.write(h.text)

    # trace
    with st.expander("🔍 检索轨迹 (Agent Trace)", expanded=False):
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
