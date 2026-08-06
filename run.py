"""
Headless CLI for quick testing without the UI.
Usage:
    python run.py "你的问题"
"""
import os
import sys

from dotenv import load_dotenv

load_dotenv()

import config
from src.ingestion import ingest
from src.retrieval import HybridIndex
from src.agent import RAGAgent
from src.generator import safe_answer


def main():
    if len(sys.argv) < 2:
        print('用法: python run.py "你的问题"')
        return
    question = sys.argv[1]
    s = config.get_settings()
    if not os.path.exists(os.path.join(s.index_dir, "vectors.faiss")):
        print("[info] 未找到索引，先执行入库...")
        ingest()
    idx = HybridIndex()
    agent = RAGAgent(idx)
    res = agent.run(question)
    answer = safe_answer(question, res.context)
    print("\n=== 回答 ===")
    print(answer)
    if not res.context:
        print("[info] 未检索到任何上下文，已确定性 abstain。")
    print("\n=== 检索轨迹 ===")
    for step in res.trace:
        print(f"Step {step.iteration}: {step.query}")
        print(f"  检索到 {step.n_retrieved} 块, top: {step.top_sources}")
        if step.reflection:
            print(f"  自省: {step.reflection}")
    print(f"\n最终使用 {len(res.context)} 个块，迭代 {res.iterations} 轮。")

    # ---- observability: per-phase latency + token estimates ----
    st = getattr(res, "stats", {})
    if st:
        print("\n=== 分阶段耗时 / 统计 ===")
        print(f"  规划(plan):   {st.get('plan_ms', 0):.1f} ms")
        print(f"  检索(retrieve): {st.get('retrieve_ms', 0):.1f} ms")
        print(f"  重排(rerank):  {st.get('rerank_ms', 0):.1f} ms")
        print(f"  自省(reflect): {st.get('reflect_ms', 0):.1f} ms")
        print(f"  总计(total):   {st.get('total_ms', 0):.1f} ms")
        print(f"  LLM 调用: {st.get('llm_calls', 0)} 次")
        print(f"  Token 估算: prompt≈{st.get('prompt_tokens', 0)}, completion≈{st.get('completion_tokens', 0)}")


if __name__ == "__main__":
    main()
