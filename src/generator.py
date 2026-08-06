"""
Answer generation with citation grounding.
The LLM is instructed to cite retrieved chunks as [n]; we map those numbers
back to sources so the UI can render clickable references.

Deterministic safety layer (no LLM needed):
- validate_citations: check every [n] references an existing chunk (1..n_chunks).
- needs_abstain: when context is empty or the answer carries no valid citation,
  the answer cannot be trusted as grounded -> we abstain instead of risking a
  hallucinated reply.
"""
import re

from src.retrieval import Hit

SYS_PROMPT = """You are a precise RAG assistant. Use ONLY the provided context to answer.
Rules:
- Cite every claim with the chunk number in square brackets, e.g. [1] or [2][3].
- Citations MUST use a number between 1 and {n_chunks} (the chunks provided).
- If the context does not contain the answer, say you don't know and do not invent facts.
- Be concise and faithful to the source text.

Context:
{context}"""

# Returned verbatim when we must deterministically abstain.
ABSTAIN_MESSAGE = "抱歉，当前知识库中没有足够信息支撑这个回答，无法给出可靠结论。"

_CITE_RE = re.compile(r"\[(\d+)\]")


def validate_citations(answer: str, n_chunks: int) -> dict:
    """Check that every [n] citation references an existing chunk (1..n_chunks).

    Returns:
        valid:        True iff there are no out-of-range citations
        citations:    sorted unique cited chunk numbers
        invalid:      out-of-range citation numbers (e.g. > n_chunks)
        has_citation: whether the answer contains any [n] at all
    """
    if not answer:
        return {"valid": True, "citations": [], "invalid": [], "has_citation": False}
    cited = [int(m) for m in _CITE_RE.findall(answer)]
    citations = sorted(set(cited))
    invalid = sorted({c for c in cited if c < 1 or c > n_chunks})
    return {
        "valid": len(invalid) == 0,
        "citations": citations,
        "invalid": invalid,
        "has_citation": len(citations) > 0,
    }


def needs_abstain(answer: str, n_chunks: int) -> bool:
    """Deterministic abstain decision.

    True when:
      - context is empty (n_chunks == 0), or
      - the answer has no valid citation (cannot be traced to any retrieved chunk).
    This guarantees we never present an ungrounded answer as if it were sourced.
    """
    if n_chunks == 0 or not answer or not answer.strip():
        return True
    return not validate_citations(answer, n_chunks)["has_citation"]


def annotate_invalid_citations(answer: str, n_chunks: int) -> str:
    """Append a warning listing citations that fall outside [1..n_chunks].

    The model sometimes cites chunks that don't exist in the retrieved window;
    surfacing them lets the user know those references are unreliable.
    """
    v = validate_citations(answer, n_chunks)
    if v["invalid"]:
        labels = ", ".join(f"[{i}]" for i in v["invalid"])
        warn = f"\n（注意：引用 {labels} 超出上下文范围，可能为模型臆造，请谨慎采信。）"
        return answer.rstrip() + warn
    return answer


def safe_answer(question: str, context: list[Hit]) -> str:
    """Generate an answer that deterministically abstains when not grounded.

    Wraps generate(): if context is empty or the answer has no valid citation,
    returns ABSTAIN_MESSAGE; otherwise returns the answer with invalid-citation
    warnings annotated.
    """
    if not context:
        return ABSTAIN_MESSAGE
    ans = generate(question, context)
    if needs_abstain(ans, len(context)):
        return ABSTAIN_MESSAGE
    return annotate_invalid_citations(ans, len(context))


def build_context_block(context: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(context, 1):
        blocks.append(f"[{i}] (source: {h.source})\n{h.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, context: list[Hit]) -> list[dict]:
    ctx = build_context_block(context)
    return [
        {"role": "system", "content": SYS_PROMPT.format(context=ctx, n_chunks=len(context))},
        {"role": "user", "content": question},
    ]


def generate(question: str, context: list[Hit]) -> str:
    from src import llm

    return llm.chat(build_messages(question, context))


def generate_stream(question: str, context: list[Hit]):
    from src import llm

    yield from llm.stream_chat(build_messages(question, context))
