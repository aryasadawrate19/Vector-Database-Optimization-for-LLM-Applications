import numpy as np
from typing import List, Tuple


class ExactBaselineIndex:
    """
    Exact (brute-force) vector search baseline using cosine similarity.
    Used as a ground-truth and performance reference.
    """

    def __init__(self, embeddings: List[List[float]]):
        self.embeddings = np.array(embeddings)

        # Normalize once
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / norms

    def search(
        self,
        query_vector: List[float],
        k: int = 10
    ) -> Tuple[List[int], float]:
        """
        Perform exact cosine similarity search.

        Returns:
            indices: top-k document indices
            latency_ms: search time in milliseconds
        """
        query = np.array(query_vector)
        query = query / np.linalg.norm(query)

        start = np.perf_counter()
        similarities = np.dot(self.embeddings, query)
        top_k = np.argsort(similarities)[::-1][:k]
        latency_ms = (np.perf_counter() - start) * 1000

        return top_k.tolist(), latency_ms
