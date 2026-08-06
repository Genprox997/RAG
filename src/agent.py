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
- If the question is complex / multi-part, also provide sub-queries (as many as
  needed, usually 1-3) that together cover the question from different angles.

Respond ONLY with JSON: {"search_query": str, "sub_queries": [str]}

Question: {question}"""

REFLECT_PROMPT = """You are evaluating whether retrieved context is sufficient to answer a question.
Context (numbered):
{context}

Question: {question}

Decide:
- sufficient: true if the context contains the information needed to answer.
- confidence: a float in [0,1] estimating how sure you are the context is sufficient
  (1.0 = fully confident, 0.0 = clearly insufficient).
- gap: what specific information is still missing (empty string if sufficient).
- next_query: a focused search query to find the missing info (empty string if sufficient).

Respond ONLY with JSON: {"sufficient": bool, "confidence": float, "gap": str, "next_query": str}"""


@dataclass
class TraceStep:
    iteration: int
    query: str
    n_retrieved: int
    top_sources: list[str]
    reflection: str = ""        # "sufficient" / "insufficient(conf=..): <gap>" / "no chunks retrieved"
    next_query: str = ""
    confidence: float | None = None


@dataclass
class AgentResult:
    context: list[Hit]
    trace: list[TraceStep] = field(default_factory=list)
    iterations: int = 0
    empty_retrieval: bool = False


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


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


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

    def _reflect(self, question: str, context_text: str, strict: bool = False) -> str:
        prompt = REFLECT_PROMPT.replace("{context}", context_text).replace("{question}", question)
        if strict:
            prompt += "\n\n[System] Output STRICT JSON only, no markdown fences, no extra prose."
        return llm.chat([{"role": "user", "content": prompt}], json_mode=True)

    def _parse_reflection(self, text: str) -> dict | None:
        """Parse a reflection verdict, or None if it cannot be parsed."""
        try:
            v = _parse_json(text)
        except Exception:
            return None
        sufficient = bool(v.get("sufficient"))
        try:
            confidence = _clamp01(float(v.get("confidence", 1.0 if sufficient else 0.0)))
        except Exception:
            confidence = 1.0 if sufficient else 0.0
        return {
            "sufficient": sufficient,
            "confidence": confidence,
            "gap": str(v.get("gap", "")),
            "next_query": str(v.get("next_query", "") or ""),
        }

    def run(self, question: str) -> AgentResult:
        s = self.index.s
        # 1) plan
        plan_txt = llm.chat(
            [{"role": "user", "content": PLAN_PROMPT.replace("{question}", question)}],
            json_mode=True,
        )
        try:
            plan = _parse_json(plan_txt)
        except Exception:
            plan = {"search_query": question, "sub_queries": []}
        search_query = plan.get("search_query") or question
        # LLM decides the number of sub-queries; cap for safety (was hard-coded <=2)
        sub_queries = (plan.get("sub_queries", []) or [])[: s.max_sub_queries]

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

        # graceful degradation: nothing retrieved at all
        if not acc:
            trace[-1].reflection = "no chunks retrieved"
            return AgentResult(context=[], trace=trace, iterations=0, empty_retrieval=True)

        # 3) self-reflection loop
        iterations = 0
        empty_streak = 0
        prev_n = len(acc)
        for it in range(1, s.max_iterations + 1):
            iterations = it
            ordered = sorted(acc.values(), key=lambda x: -x.score)
            reflect_txt = self._reflect(question, _format_context(ordered))
            verdict = self._parse_reflection(reflect_txt)

            # JSON parse failed -> retry once with a stricter instruction
            if verdict is None:
                verdict = self._parse_reflection(self._reflect(question, _format_context(ordered), strict=True))

            if verdict is None:
                # still unparseable: conservative fallback — assume insufficient and
                # do one more broad retrieval rather than prematurely stopping.
                verdict = {
                    "sufficient": False,
                    "confidence": 0.0,
                    "gap": "(reflection parse failed)",
                    "next_query": question,
                }

            sufficient = verdict["sufficient"]
            confidence = verdict["confidence"]
            next_query = verdict["next_query"]
            last = trace[-1]
            tag = "sufficient" if sufficient else f"insufficient(conf={confidence:.2f}): {verdict['gap']}"
            last.reflection = tag
            last.next_query = next_query
            last.confidence = confidence

            # stop when: explicitly sufficient, OR confident enough, OR no follow-up
            if sufficient or confidence >= s.reflect_confidence_threshold or not next_query:
                break

            # 4) targeted re-retrieval
            self._dedup_merge(acc, self.index.retrieve(next_query))
            n_now = len(acc)
            empty_streak = empty_streak + 1 if n_now == prev_n else 0
            prev_n = n_now
            trace.append(
                TraceStep(
                    iteration=it,
                    query=next_query,
                    n_retrieved=n_now,
                    top_sources=[h.source for h in sorted(acc.values(), key=lambda x: -x.score)[:3]],
                )
            )
            # two consecutive retrievals returned nothing new -> stop gracefully
            if empty_streak >= 2:
                trace[-1].reflection = "no new chunks; stopping"
                break

        final = sorted(acc.values(), key=lambda x: -x.score)[: s.top_k_final]
        return AgentResult(context=final, trace=trace, iterations=iterations)
