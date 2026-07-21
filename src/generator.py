"""
Answer generation with citation grounding.
The LLM is instructed to cite retrieved chunks as [n]; we map those numbers
back to sources so the UI can render clickable references.
"""
from src.retrieval import Hit

SYS_PROMPT = """You are a precise RAG assistant. Use ONLY the provided context to answer.
Rules:
- Cite every claim with the chunk number in square brackets, e.g. [1] or [2][3].
- If the context does not contain the answer, say you don't know and do not invent facts.
- Be concise and faithful to the source text.

Context:
{context}"""


def build_context_block(context: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(context, 1):
        blocks.append(f"[{i}] (source: {h.source})\n{h.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, context: list[Hit]) -> list[dict]:
    ctx = build_context_block(context)
    return [
        {"role": "system", "content": SYS_PROMPT.format(context=ctx)},
        {"role": "user", "content": question},
    ]


def generate(question: str, context: list[Hit]) -> str:
    from src import llm

    return llm.chat(build_messages(question, context))


def generate_stream(question: str, context: list[Hit]):
    from src import llm

    yield from llm.stream_chat(build_messages(question, context))
