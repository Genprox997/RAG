"""
Tokenizer for BM25 / keyword retrieval.
Handles both English (alphanumeric tokens) and Chinese (per-CJK-character)
so keyword search works for mixed-language corpora.
"""
import re

_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenizer for BM25 / keyword retrieval.

    - English / numeric runs -> lowercase word tokens.
    - CJK characters -> unigrams AND adjacent bigrams, so multi-char terms like
      "混合检索" / "MicroLED检测" yield phrase-level tokens. This improves recall
      for mixed Chinese/English technical corpora while staying dependency-free
      and fully offline (no jieba required).
    """
    text = (text or "").lower()
    tokens: list[str] = []
    # English / numeric tokens
    tokens.extend(_WORD.findall(text))

    cjk = _CJK.findall(text)
    # unigrams (single characters)
    tokens.extend(cjk)
    # adjacent bigrams (phrase-level signal)
    for i in range(len(cjk) - 1):
        tokens.append(cjk[i] + cjk[i + 1])
    return tokens


if __name__ == "__main__":
    print(tokenize("RAG 检索增强生成 Retrieval-Augmented Generation 2024"))
