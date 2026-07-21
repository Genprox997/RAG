"""
Lightweight RAG evaluation (RAGAS-style) without external eval libs.
Three LLM-judged metrics:
  - Faithfulness      : are answer claims supported by the context?        (0-1)
  - Answer Relevancy  : does the answer address the question?               (0-1)
  - Context Relevance : is the retrieved context relevant to the question?  (0-1)
Run over a golden set and print an aggregate report.
"""
import json
import os
from typing import Optional

from src import llm
from src.agent import RAGAgent
from src.generator import generate, build_context_block
from src.retrieval import HybridIndex

FAITH_PROMPT = """You are evaluating faithfulness of an answer to retrieved context.
List each distinct factual claim in the ANSWER, then mark whether it is
supported by the CONTEXT. Respond ONLY with JSON:
{"claims": [{"claim": str, "supported": bool}]}

CONTEXT:
{context}

ANSWER:
{answer}"""

RELEV_PROMPT = """Rate how relevant the ANSWER is to the QUESTION on a scale 1-5
(1=irrelevant, 5=fully answers). Respond ONLY with JSON: {"score": int, "reason": str}
QUESTION: {question}
ANSWER: {answer}"""

CTXREL_PROMPT = """Rate how relevant the RETRIEVED CONTEXT is to the QUESTION on a scale 1-5
(1=irrelevant, 5=highly relevant). Respond ONLY with JSON: {"score": int, "reason": str}
QUESTION: {question}
CONTEXT:
{context}"""


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1:
        text = text[s : e + 1]
    return json.loads(text)


def faithfulness(answer: str, context_text: str) -> float:
    if not answer.strip():
        return 0.0
    try:
        out = _parse_json(
            llm.chat(
                [
                    {
                        "role": "user",
                        "content": FAITH_PROMPT.replace("{context}", context_text).replace(
                            "{answer}", answer
                        ),
                    }
                ],
                json_mode=True,
            )
        )
        claims = out.get("claims", [])
        if not claims:
            return 1.0
        supp = sum(1 for c in claims if c.get("supported"))
        return supp / len(claims)
    except Exception as e:
        print(f"[warn] faithfulness eval failed: {e}")
        return 0.0


def answer_relevancy(question: str, answer: str) -> float:
    try:
        out = _parse_json(
            llm.chat(
                [
                    {
                        "role": "user",
                        "content": RELEV_PROMPT.replace("{question}", question).replace(
                            "{answer}", answer
                        ),
                    }
                ],
                json_mode=True,
            )
        )
        return max(0.0, min(1.0, int(out.get("score", 0)) / 5.0))
    except Exception:
        return 0.0


def context_relevance(question: str, context_text: str) -> float:
    try:
        out = _parse_json(
            llm.chat(
                [
                    {
                        "role": "user",
                        "content": CTXREL_PROMPT.replace("{question}", question).replace(
                            "{context}", context_text
                        ),
                    }
                ],
                json_mode=True,
            )
        )
        return max(0.0, min(1.0, int(out.get("score", 0)) / 5.0))
    except Exception:
        return 0.0


def evaluate_item(agent: RAGAgent, question: str) -> dict:
    res = agent.run(question)
    context_text = build_context_block(res.context)
    answer = generate(question, res.context)
    return {
        "question": question,
        "answer": answer,
        "faithfulness": round(faithfulness(answer, context_text), 3),
        "answer_relevancy": round(answer_relevancy(question, answer), 3),
        "context_relevance": round(context_relevance(question, context_text), 3),
        "n_iterations": res.iterations,
        "n_chunks": len(res.context),
    }


def run_evaluation(golden_path: str = "evaluation/golden_set.json", index_dir: Optional[str] = None):
    with open(golden_path, "r", encoding="utf-8") as f:
        golden = json.load(f)
    agent = RAGAgent(HybridIndex(index_dir))
    rows = [evaluate_item(agent, item["question"]) for item in golden]

    n = len(rows) or 1
    agg = {
        "faithfulness": round(sum(r["faithfulness"] for r in rows) / n, 3),
        "answer_relevancy": round(sum(r["answer_relevancy"] for r in rows) / n, 3),
        "context_relevance": round(sum(r["context_relevance"] for r in rows) / n, 3),
    }
    return rows, agg


if __name__ == "__main__":
    rows, agg = run_evaluation()
    print(json.dumps(agg, indent=2, ensure_ascii=False))
    for r in rows:
        print(f"\nQ: {r['question']}\nA: {r['answer'][:200]}...\n  metrics={r}")
