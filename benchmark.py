"""
Benchmark Module
Runs experiments to compare different indexing and query configurations.
"""

import time
import pandas as pd
from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm

from embedding_generator import EmbeddingGenerator
from weaviate_client import WeaviateVectorDB


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
        """
        Initialize benchmark suite.
        
        Args:
            embedding_generator: Instance of EmbeddingGenerator.
            corpus: List of documents to index.
            test_queries: List of query texts for benchmarking.
        """
        self.embedding_generator = embedding_generator
        self.corpus = corpus
        self.test_queries = test_queries
        self.results = []
        
        print(f" Benchmark initialized")
        print(f"  - Corpus size: {len(corpus)} documents")
        print(f"  - Test queries: {len(test_queries)}")
    
    def _calculate_recall_at_k(
        self, 
        retrieved_ids: List[int], 
        ground_truth_ids: List[int], 
        k: int
    ) -> float:
        """
        Calculate Recall@K metric.
        
        Args:
            retrieved_ids: IDs of retrieved documents.
            ground_truth_ids: IDs of relevant documents.
            k: Number of top results to consider.
        
        Returns:
            Recall@K score (0-1).
        """
        if not ground_truth_ids:
            return 0.0
        
        retrieved_set = set(retrieved_ids[:k])
        ground_truth_set = set(ground_truth_ids)
        
        intersection = len(retrieved_set.intersection(ground_truth_set))
        recall = intersection / min(len(ground_truth_set), k)
        
        return recall
    
    def _compute_ground_truth(
        self, 
        query_embeddings: List[List[float]], 
        corpus_embeddings: List[List[float]],
        k: int = 10
    ) -> List[List[int]]:
        """
        Compute ground truth using brute-force exact search.
        
        Args:
            query_embeddings: Query embedding vectors.
            corpus_embeddings: Corpus embedding vectors.
            k: Number of top results.
        
        Returns:
            List of top-k document indices for each query.
        """
        ground_truth = []
        
        query_matrix = np.array(query_embeddings)
        corpus_matrix = np.array(corpus_embeddings)
        
        # Normalize vectors for cosine similarity
        query_matrix = query_matrix / np.linalg.norm(query_matrix, axis=1, keepdims=True)
        corpus_matrix = corpus_matrix / np.linalg.norm(corpus_matrix, axis=1, keepdims=True)
        
        # Compute similarity matrix
        similarities = np.dot(query_matrix, corpus_matrix.T)
        
        # Get top-k indices for each query
        for query_sims in similarities:
            top_k_indices = np.argsort(query_sims)[::-1][:k].tolist()
            ground_truth.append(top_k_indices)
        
        return ground_truth
    
    def run_benchmark(
        self,
        ef_construction_values: List[int] = [64, 128, 256],
        ef_values: List[int] = [32, 64, 128],
        max_connections_values: List[int] = [16, 32, 64],
        k: int = 10,
        num_iterations: int = 3
    ) -> pd.DataFrame:
        """
        Run comprehensive benchmark across different configurations.
        
        Args:
            ef_construction_values: List of ef_construction values to test.
            ef_values: List of ef values to test.
            max_connections_values: List of max_connections values to test.
            k: Number of top results for recall calculation.
            num_iterations: Number of iterations per configuration.
        
        Returns:
            DataFrame with benchmark results.
        """
        print("\n" + "="*60)
        print("STARTING VECTOR DATABASE BENCHMARK")
        print("="*60)
        
        # Generate embeddings for corpus and queries
        print("\n[1/3] Generating corpus embeddings...")
        corpus_embeddings = self.embedding_generator.generate_embeddings_batch(
            self.corpus, 
            task_type="retrieval_document"
        )
        
        print("\n[2/3] Generating query embeddings...")
        query_embeddings = self.embedding_generator.generate_embeddings_batch(
            self.test_queries, 
            task_type="retrieval_query"
        )
        
        # Compute ground truth for recall calculation
        print("\n[3/3] Computing ground truth...")
        ground_truth = self._compute_ground_truth(query_embeddings, corpus_embeddings, k)
        
        # Get embedding dimension
        embedding_dim = len(corpus_embeddings[0])
        
        # Run benchmarks for each configuration
        total_configs = len(ef_construction_values) * len(ef_values) * len(max_connections_values)
        print(f"\nRunning {total_configs} configuration tests...\n")
        
        config_num = 0
        for ef_construction in ef_construction_values:
            for ef in ef_values:
                for max_connections in max_connections_values:
                    config_num += 1
                    print(f"\n--- Configuration {config_num}/{total_configs} ---")
                    print(f"ef_construction={ef_construction}, ef={ef}, max_connections={max_connections}")
                    
                    result = self._benchmark_configuration(
                        corpus_embeddings=corpus_embeddings,
                        query_embeddings=query_embeddings,
                        ground_truth=ground_truth,
                        embedding_dim=embedding_dim,
                        ef_construction=ef_construction,
                        ef=ef,
                        max_connections=max_connections,
                        k=k,
                        num_iterations=num_iterations
                    )
                    
                    self.results.append(result)
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*60)
        print("BENCHMARK COMPLETED")
        print("="*60)
        
        return df
    
    def _benchmark_configuration(
        self,
        corpus_embeddings: List[List[float]],
        query_embeddings: List[List[float]],
        ground_truth: List[List[int]],
        embedding_dim: int,
        ef_construction: int,
        ef: int,
        max_connections: int,
        k: int,
        num_iterations: int
    ) -> Dict[str, Any]:
        """
        Benchmark a single configuration.
        
        Returns:
            Dictionary with performance metrics.
        """
        try:
            # Initialize Weaviate client
            db = WeaviateVectorDB(collection_name=f"Benchmark_{int(time.time())}")
            
            # Create schema with specific configuration
            db.create_schema(
                vector_dimension=embedding_dim,
                ef_construction=ef_construction,
                ef=ef,
                max_connections=max_connections
            )
            
            # Insert corpus
            start_insert = time.time()
            db.insert_data(self.corpus, corpus_embeddings, show_progress=False)
            insert_time = time.time() - start_insert
            
            # Measure query performance
            query_times = []
            recall_scores = []
            
            for iteration in range(num_iterations):
                for query_idx, query_emb in enumerate(query_embeddings):
                    # Measure query time
                    start_query = time.time()
                    results = db.search(query_emb, limit=k, return_metadata=False)
                    query_time = (time.time() - start_query) * 1000  # Convert to ms
                    query_times.append(query_time)
                    
                    # Calculate recall
                    retrieved_indices = []
                    for result in results:
                        # Find index in corpus
                        try:
                            idx = self.corpus.index(result['text'])
                            retrieved_indices.append(idx)
                        except ValueError:
                            continue
                    
                    recall = self._calculate_recall_at_k(
                        retrieved_indices, 
                        ground_truth[query_idx], 
                        k
                    )
                    recall_scores.append(recall)
            
            # Calculate metrics
            avg_query_time = np.mean(query_times)
            std_query_time = np.std(query_times)
            avg_recall = np.mean(recall_scores) * 100  # Convert to percentage
            throughput = len(query_embeddings) * num_iterations / (sum(query_times) / 1000)  # queries per second
            
            # Clean up
            db.delete_collection()
            db.close()
            
            print(f"   Avg Query Time: {avg_query_time:.2f} ms")
            print(f"   Recall@{k}: {avg_recall:.1f}%")
            print(f"   Throughput: {throughput:.1f} queries/sec")
            
            return {
                "ef_construction": ef_construction,
                "ef": ef,
                "max_connections": max_connections,
                "avg_query_time_ms": round(avg_query_time, 2),
                "std_query_time_ms": round(std_query_time, 2),
                f"recall_at_{k}_percent": round(avg_recall, 1),
                "throughput_qps": round(throughput, 1),
                "insert_time_sec": round(insert_time, 2),
                "num_queries": len(query_embeddings),
                "num_iterations": num_iterations
            }
            
        except Exception as e:
            print(f"   Configuration failed: {e}")
            return {
                "ef_construction": ef_construction,
                "ef": ef,
                "max_connections": max_connections,
                "error": str(e)
            }
    
    def get_best_configuration(self, df: pd.DataFrame, optimize_for: str = "latency") -> Dict[str, Any]:
        """
        Find the best configuration based on optimization criteria.
        
        Args:
            df: Benchmark results DataFrame.
            optimize_for: Optimization criterion ('latency', 'recall', 'balanced').
        
        Returns:
            Dictionary with best configuration details.
        """
        if df.empty:
            return {}
        
        # Remove failed configurations
        df_valid = df[~df['avg_query_time_ms'].isna()].copy()
        
        if df_valid.empty:
            return {}
        
        if optimize_for == "latency":
            # Minimize query latency
            best_idx = df_valid['avg_query_time_ms'].idxmin()
        elif optimize_for == "recall":
            # Maximize recall
            recall_col = [col for col in df_valid.columns if 'recall_at' in col][0]
            best_idx = df_valid[recall_col].idxmax()
        else:  # balanced
            # Balance between latency and recall using normalized scores
            recall_col = [col for col in df_valid.columns if 'recall_at' in col][0]
            
            # Normalize metrics (0-1 scale)
            latency_range = df_valid['avg_query_time_ms'].max() - df_valid['avg_query_time_ms'].min()
            recall_range = df_valid[recall_col].max() - df_valid[recall_col].min()
            
            # Handle cases where all values are the same (avoid division by zero)
            if latency_range == 0:
                normalized_latency = 1.0  # All configs have same latency, all get max score
            else:
                normalized_latency = 1 - (df_valid['avg_query_time_ms'] - df_valid['avg_query_time_ms'].min()) / latency_range
            
            if recall_range == 0:
                normalized_recall = 1.0  # All configs have same recall, all get max score
            else:
                normalized_recall = (df_valid[recall_col] - df_valid[recall_col].min()) / recall_range
            
            # Combined score (equal weights)
            combined_score = 0.5 * normalized_latency + 0.5 * normalized_recall
            best_idx = combined_score.idxmax()
        
        return df_valid.loc[best_idx].to_dict()


# Example usage
if __name__ == "__main__":
    try:
        # Sample corpus
        corpus = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing enables computers to understand text.",
            "Computer vision allows machines to interpret visual information.",
            "Reinforcement learning trains agents through rewards and penalties."
        ]
        
        # Sample queries
        queries = [
            "What is machine learning?",
            "How do neural networks work?"
        ]
        
        # Initialize components
        generator = EmbeddingGenerator()
        benchmark = VectorDBBenchmark(generator, corpus, queries)
        
        # Run quick benchmark
        results_df = benchmark.run_benchmark(
            ef_construction_values=[64, 128],
            ef_values=[32, 64],
            max_connections_values=[16, 32],
            k=5,
            num_iterations=2
        )
        
        print("\n" + results_df.to_string())
        
        # Find best configuration
        best = benchmark.get_best_configuration(results_df, optimize_for="balanced")
        print(f"\n Best configuration: {best}")
        
    except Exception as e:
        print(f" Error: {e}")
