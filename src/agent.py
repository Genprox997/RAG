"""
Agentic RAG loop:
  1. Query understanding  -> rewrite + decompose into sub-queries
  2. Multi-step retrieval -> hybrid search for each query
  3. Self-reflection      -> LLM judges if context is sufficient
  4. If not sufficient    -> generate a targeted follow-up query and re-retrieve
Outputs the accumulated context plus an observable retrieval trace.
"""
import json
from dataclasses import dataclass, field

from src import llm
from src.retrieval import HybridIndex, Hit

PLAN_PROMPT = """You are a retrieval planner for a RAG system.
Given the user's question, produce an optimized search strategy.
- Rewrite the question into a concise, keyword-rich search query.
- If the question is complex / multi-part, also provide up to 2 sub-queries
  that together cover the question from different angles.

Respond ONLY with JSON: {"search_query": str, "sub_queries": [str]}

Question: {question}"""

REFLECT_PROMPT = """You are evaluating whether retrieved context is sufficient to answer a question.
Context (numbered):
{context}

Question: {question}

Decide:
- sufficient: true if the context contains the information needed to answer.
- gap: what specific information is still missing (empty string if sufficient).
- next_query: a focused search query to find the missing info (empty string if sufficient).

Respond ONLY with JSON: {"sufficient": bool, "gap": str, "next_query": str}"""


@dataclass
class TraceStep:
    iteration: int
    query: str
    n_retrieved: int
    top_sources: list[str]
    reflection: str = ""        # "sufficient" / "insufficient: <gap>"
    next_query: str = ""


@dataclass
class AgentResult:
    context: list[Hit]
    trace: list[TraceStep] = field(default_factory=list)
    iterations: int = 0


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    # tolerate trailing prose: grab first {...}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


def _format_context(chunks: list[Hit], max_chars: int = 1200) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        t = c.text if len(c.text) <= max_chars else c.text[:max_chars] + "…"
        lines.append(f"[{i}] (source: {c.source})\n{t}")
    return "\n\n".join(lines)


class RAGAgent:
    def __init__(self, index: HybridIndex):
        self.index = index

    def _dedup_merge(self, acc: dict[int, Hit], hits: list[Hit]):
        for h in hits:
            if h.chunk_id in acc:
                acc[h.chunk_id].score = max(acc[h.chunk_id].score, h.score)
            else:
                acc[h.chunk_id] = h

    def run(self, question: str) -> AgentResult:
        s = self.index.s
        # 1) plan
        plan_txt = llm.chat(
            [{"role": "user", "content": PLAN_PROMPT.replace("{question}", question)}],
            json_mode=True,
        )
        plan = _parse_json(plan_txt)
        search_query = plan.get("search_query") or question
        sub_queries = plan.get("sub_queries", []) or []

        acc: dict[int, Hit] = {}
        trace: list[TraceStep] = []

        # 2) initial retrieval (main + sub queries)
        all_queries = [search_query] + sub_queries
        for q in all_queries:
            self._dedup_merge(acc, self.index.retrieve(q))
        trace.append(
            TraceStep(
                iteration=0,
                query=" | ".join(all_queries),
                n_retrieved=len(acc),
                top_sources=[h.source for h in sorted(acc.values(), key=lambda x: -x.score)[:3]],
            )
        )

        # 3) self-reflection loop
        iterations = 0
        for it in range(1, s.max_iterations + 1):
            iterations = it
            ordered = sorted(acc.values(), key=lambda x: -x.score)
            reflect_txt = llm.chat(
                [
                    {
                        "role": "user",
                        "content": REFLECT_PROMPT.replace("{context}", _format_context(ordered)).replace(
                            "{question}", question
                        ),
                    }
                ],
                json_mode=True,
            )
            try:
                verdict = _parse_json(reflect_txt)
            except Exception:
                verdict = {"sufficient": True, "gap": "", "next_query": ""}
            sufficient = bool(verdict.get("sufficient"))
            next_query = verdict.get("next_query", "") or ""
            last = trace[-1]
            last.reflection = "sufficient" if sufficient else f"insufficient: {verdict.get('gap','')}"
            last.next_query = next_query

            if sufficient or not next_query:
                break
            # 4) targeted re-retrieval
            self._dedup_merge(acc, self.index.retrieve(next_query))
            trace.append(
                TraceStep(
                    iteration=it,
                    query=next_query,
                    n_retrieved=len(acc),
                    top_sources=[h.source for h in sorted(acc.values(), key=lambda x: -x.score)[:3]],
                )
            )

        final = sorted(acc.values(), key=lambda x: -x.score)[: s.top_k_final]
        return AgentResult(context=final, trace=trace, iterations=iterations)
