from datasets import load_dataset
from typing import List, Tuple


def load_general_knowledge_dataset(
    max_corpus: int | None = None,
    max_queries: int | None = None,
) -> Tuple[List[str], List[str]]:
    """
    Load MuskumPillerum/General-Knowledge dataset.

    Filters empty / invalid entries.
    """

    dataset = load_dataset(
        "MuskumPillerum/General-Knowledge",
        split="train"
    )

    raw_corpus = dataset["Answer"]
    raw_queries = dataset["Question"]

    # Filter empty or whitespace-only entries
    corpus = [
        text.strip()
        for text in raw_corpus
        if isinstance(text, str) and text.strip()
    ]

    queries = [
        text.strip()
        for text in raw_queries
        if isinstance(text, str) and text.strip()
    ]

    if max_corpus:
        corpus = corpus[:max_corpus]

    if max_queries:
        queries = queries[:max_queries]

    return corpus, queries
