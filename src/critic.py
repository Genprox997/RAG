"""
Answer-level Critic: a second-pass self-check on the generated answer.

Complements the context-level self-reflection in agent.py (which only judges
whether the *retrieved context* is sufficient). This critic inspects the final
*answer* for:
  - hallucination: claims / numbers in the answer not grounded in the context;
  - omission: important facts present in the context that the answer ignored.

The semantic judgment is done by an LLM (injectable for testing). A deterministic
lexical check (`detect_unsupported_numbers`) also flags answer numbers absent from
the context, so the critic still adds value when the LLM is unavailable.
"""
import json
import re
from dataclasses import dataclass, field

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?%?")


CRITIC_PROMPT = """You are a strict critic verifying an answer strictly against the retrieved context.
Only trust claims that are explicitly supported by the context.

Context:
{context}

Question: {question}

Answer:
{answer}

Check for:
- Hallucination: specific claims or numbers in the answer that are NOT supported by the context.
- Omission: important facts present in the context that the answer failed to use.

Respond ONLY with JSON:
{{"faithful": bool, "hallucinated": [str], "missing": [str], "issues": [str]}}"""


def _context_to_text(context) -> str:
    if isinstance(context, str):
        return context
    # list[Hit] (has .text) or list[str]
    parts = []
    for c in context:
        parts.append(getattr(c, "text", c))
    return "\n\n".join(parts)


def extract_numbers(text: str) -> list[str]:
    return [_strip_pct(n) for n in _NUM_RE.findall(text or "")]


def _strip_pct(n: str) -> str:
    return n[:-1] if n.endswith("%") else n


def detect_unsupported_numbers(answer: str, context_text: str) -> list[str]:
    """Deterministic lexical check: numbers in the answer not present in context.

    Returns the list of answer numbers (as strings) that do not appear verbatim in
    the context. Useful as a cheap hallucination signal and as an LLM-free fallback.
    """
    ctx = (context_text or "").replace(" ", "")
    unsupported = []
    for num in extract_numbers(answer):
        # normalize both sides (drop spaces) so "99.5" matches "99.5%"/"99.5"
        if num not in ctx.replace(" ", "") and num not in context_text:
            unsupported.append(num)
    return unsupported


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


@dataclass
class CriticResult:
    faithful: bool
    hallucinated: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


def critic(question: str, answer: str, context, chat_fn=None) -> CriticResult:
    """Critique an answer against the context.

    chat_fn: optional callable(messages, json_mode=False) -> str used to obtain the
    LLM judgment. Defaults to src.llm.chat. Passing a fake makes this fully testable
    offline.
    """
    ctx_text = _context_to_text(context)
    unsupported = detect_unsupported_numbers(answer, ctx_text)

    if chat_fn is None:
        from src import llm

        chat_fn = llm.chat

    try:
        prompt = CRITIC_PROMPT.format(context=ctx_text, question=question, answer=answer)
        out = _parse_json(chat_fn([{"role": "user", "content": prompt}], json_mode=True))
    except Exception:
        out = {}

    faithful = bool(out.get("faithful", len(unsupported) == 0))
    hallucinated = list(out.get("hallucinated", unsupported)) or list(unsupported)
    missing = list(out.get("missing", []))
    issues = list(out.get("issues", []))
    if not issues and unsupported:
        issues = [f"answer contains numbers not present in context: {unsupported}"]

    return CriticResult(
        faithful=faithful,
        hallucinated=hallucinated,
        missing=missing,
        issues=issues,
        raw=out,
    )
