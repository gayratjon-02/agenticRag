import re

# Common question/stop words dropped when reducing a query to keywords.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "what",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "which",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "will",
        "shall",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "and",
        "or",
        "but",
        "with",
        "about",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "my",
        "your",
        "this",
        "that",
    }
)


def reformulate_query(question: str) -> str:
    """Deterministically reduce a question to keywords for a second retrieval pass.

    Lowercases, keeps alphanumeric tokens, and drops common question/stop words.
    Fully inspectable and reproducible — no LLM, no new tools. Returns the original
    question unchanged if the reduction would be empty.
    """
    tokens = re.findall(r"[a-z0-9]+", question.lower())
    keywords = [token for token in tokens if token not in _STOPWORDS]
    return " ".join(keywords) or question
