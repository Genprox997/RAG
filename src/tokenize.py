"""
Tokenizer for BM25 / keyword retrieval.
Handles both English (alphanumeric tokens) and Chinese (per-CJK-character)
so keyword search works for mixed-language corpora.
"""
import re

_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    tokens: list[str] = []
    # English / numeric tokens
    tokens.extend(_WORD.findall(text))
    # Each CJK character as its own token (cheap, effective for BM25)
    tokens.extend(_CJK.findall(text))
    return tokens


if __name__ == "__main__":
    print(tokenize("RAG 检索增强生成 Retrieval-Augmented Generation 2024"))
