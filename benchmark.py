"""
Benchmark Module
Runs experiments to compare different indexing and query configurations.
Includes an exact (brute-force) baseline index.
"""

import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm

from embedding_generator import EmbeddingGenerator
from weaviate_client import WeaviateVectorDB
from baseline_index import ExactBaselineIndex


class VectorDBBenchmark:
    """
    Benchmarks vector database performance with different configurations.
    Measures query latency, recall@k, and throughput.
    """

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        corpus: List[str],
        test_queries: List[str]
    ):
        self.embedding_generator = embedding_generator
        self.corpus = corpus
        self.test_queries = test_queries
        self.results = []

        # Fast lookup for recall computation
        self.doc_id_map = {text: idx for idx, text in enumerate(corpus)}

        print(" Benchmark initialized")
        print(f"  - Corpus size: {len(corpus)} documents")
        print(f"  - Test queries: {len(test_queries)}")

    # ------------------------------------------------------------------
    # Metrics utilities
    # ------------------------------------------------------------------

    def _calculate_recall_at_k(
        self,
        retrieved_ids: List[int],
        ground_truth_ids: List[int],
        k: int
    ) -> float:
        if not ground_truth_ids:
            return 0.0

        retrieved_set = set(retrieved_ids[:k])
        ground_truth_set = set(ground_truth_ids)

        return len(retrieved_set & ground_truth_set) / min(len(ground_truth_set), k)

    def _compute_ground_truth(
        self,
        query_embeddings: List[List[float]],
        corpus_embeddings: List[List[float]],
        k: int
    ) -> List[List[int]]:
        """
        Compute exact cosine-similarity ground truth.
        """
        query_matrix = np.array(query_embeddings)
        corpus_matrix = np.array(corpus_embeddings)

        query_matrix /= np.linalg.norm(query_matrix, axis=1, keepdims=True)
        corpus_matrix /= np.linalg.norm(corpus_matrix, axis=1, keepdims=True)

        similarities = np.dot(query_matrix, corpus_matrix.T)

        ground_truth = []
        for sims in similarities:
            top_k = np.argsort(sims)[::-1][:k]
            ground_truth.append(top_k.tolist())

        return ground_truth

    # ------------------------------------------------------------------
    # Main benchmark runner
    # ------------------------------------------------------------------

    def run_benchmark(
        self,
        ef_construction_values: List[int],
        ef_values: List[int],
        max_connections_values: List[int],
        k: int = 10,
        num_iterations: int = 2
    ) -> pd.DataFrame:

        print("\n" + "=" * 60)
        print("STARTING VECTOR DATABASE BENCHMARK")
        print("=" * 60)

        # --------------------------------------------------------------
        # Embedding generation
        # --------------------------------------------------------------
        print("\n[1/4] Generating corpus embeddings...")
        corpus_embeddings = self.embedding_generator.generate_embeddings_batch(
            self.corpus, task_type="retrieval_document"
        )

        print("\n[2/4] Generating query embeddings...")
        query_embeddings = self.embedding_generator.generate_embeddings_batch(
            self.test_queries, task_type="retrieval_query"
        )

        print("\n[3/4] Computing exact ground truth...")
        ground_truth = self._compute_ground_truth(
            query_embeddings, corpus_embeddings, k
        )

        embedding_dim = len(corpus_embeddings[0])

        # --------------------------------------------------------------
        # BASELINE: Exact search
        # --------------------------------------------------------------
        print("\n[BASELINE] Running exact brute-force index...")
        baseline = ExactBaselineIndex(corpus_embeddings)

        baseline_latencies = []
        baseline_recalls = []

        for q_idx, q_emb in enumerate(tqdm(query_embeddings, desc="Baseline search")):
            retrieved_ids, latency = baseline.search(q_emb, k)
            baseline_latencies.append(latency)

            recall = self._calculate_recall_at_k(
                retrieved_ids, ground_truth[q_idx], k
            )
            baseline_recalls.append(recall)

        self.results.append({
            "index_type": "exact_baseline",
            "ef_construction": None,
            "ef": None,
            "max_connections": None,
            "avg_query_time_ms": round(np.mean(baseline_latencies), 2),
            "std_query_time_ms": round(np.std(baseline_latencies), 2),
            f"recall_at_{k}_percent": round(np.mean(baseline_recalls) * 100, 1),
            "throughput_qps": round(
                len(query_embeddings) / (sum(baseline_latencies) / 1000), 1
            ),
            "insert_time_sec": 0.0,
            "num_queries": len(query_embeddings),
            "num_iterations": 1
        })

        # --------------------------------------------------------------
        # HNSW benchmarks
        # --------------------------------------------------------------
        print("\n[4/4] Running HNSW benchmarks...")

        total_configs = (
            len(ef_construction_values)
            * len(ef_values)
            * len(max_connections_values)
        )

        config_num = 0

        for efc in ef_construction_values:
            for ef in ef_values:
                for mc in max_connections_values:
                    config_num += 1
                    print(f"\n--- Configuration {config_num}/{total_configs} ---")
                    print(f"ef_construction={efc}, ef={ef}, max_connections={mc}")

                    result = self._benchmark_hnsw_configuration(
                        corpus_embeddings,
                        query_embeddings,
                        ground_truth,
                        embedding_dim,
                        efc,
                        ef,
                        mc,
                        k,
                        num_iterations
                    )

                    self.results.append(result)

        print("\n" + "=" * 60)
        print("BENCHMARK COMPLETED")
        print("=" * 60)

        return pd.DataFrame(self.results)

    # ------------------------------------------------------------------
    # Single HNSW configuration
    # ------------------------------------------------------------------

    def _benchmark_hnsw_configuration(
        self,
        corpus_embeddings,
        query_embeddings,
        ground_truth,
        embedding_dim,
        ef_construction,
        ef,
        max_connections,
        k,
        num_iterations
    ) -> Dict[str, Any]:

        db = WeaviateVectorDB(
            collection_name=f"HNSW_{int(time.time() * 1000)}"
        )

        db.create_schema(
            vector_dimension=embedding_dim,
            ef_construction=ef_construction,
            ef=ef,
            max_connections=max_connections
        )

        start_insert = time.time()
        db.insert_data(self.corpus, corpus_embeddings, show_progress=False)
        insert_time = time.time() - start_insert

        query_times = []
        recall_scores = []

        for _ in range(num_iterations):
            for q_idx, q_emb in enumerate(query_embeddings):
                start = time.time()
                results = db.search(q_emb, limit=k, return_metadata=False)
                latency = (time.time() - start) * 1000
                query_times.append(latency)

                retrieved_ids = []
                for r in results:
                    idx = self.doc_id_map.get(r["text"])
                    if idx is not None:
                        retrieved_ids.append(idx)

                recall = self._calculate_recall_at_k(
                    retrieved_ids, ground_truth[q_idx], k
                )
                recall_scores.append(recall)

        db.delete_collection()
        db.close()

        return {
            "index_type": "hnsw",
            "ef_construction": ef_construction,
            "ef": ef,
            "max_connections": max_connections,
            "avg_query_time_ms": round(np.mean(query_times), 2),
            "std_query_time_ms": round(np.std(query_times), 2),
            f"recall_at_{k}_percent": round(np.mean(recall_scores) * 100, 1),
            "throughput_qps": round(
                len(query_embeddings) * num_iterations / (sum(query_times) / 1000), 1
            ),
            "insert_time_sec": round(insert_time, 2),
            "num_queries": len(query_embeddings),
            "num_iterations": num_iterations
        }

    def get_best_configuration(
        self,
        df: pd.DataFrame,
        optimize_for: str = "balanced"
    ) -> dict:
        if df.empty:
            return {}

        df_valid = df[df["index_type"] == "hnsw"].copy()
        if df_valid.empty:
            return {}

        recall_col = [c for c in df_valid.columns if "recall_at" in c][0]

        if optimize_for == "latency":
            best_idx = df_valid["avg_query_time_ms"].idxmin()

        elif optimize_for == "recall":
            best_idx = df_valid[recall_col].idxmax()

        else:
            latency_norm = (
                df_valid["avg_query_time_ms"] - df_valid["avg_query_time_ms"].min()
            ) / (
                df_valid["avg_query_time_ms"].max()
                - df_valid["avg_query_time_ms"].min()
                + 1e-9
            )

            recall_norm = (
                df_valid[recall_col] - df_valid[recall_col].min()
            ) / (
                df_valid[recall_col].max()
                - df_valid[recall_col].min()
                + 1e-9
            )

            score = 0.5 * (1 - latency_norm) + 0.5 * recall_norm
            best_idx = score.idxmax()

        return df_valid.loc[best_idx].to_dict()

