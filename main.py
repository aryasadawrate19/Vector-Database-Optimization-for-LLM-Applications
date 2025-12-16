"""
Main Pipeline
Integrates all modules into a complete vector database optimization workflow.
"""

import time
import pandas as pd
from typing import List

from embedding_generator import EmbeddingGenerator
from weaviate_client import WeaviateVectorDB
from benchmark import VectorDBBenchmark
from dataset_loader import load_general_knowledge_dataset


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70 + "\n")


def print_results_summary(df: pd.DataFrame, best_config: dict):
    """
    Print a formatted summary of benchmark results.
    """
    print_header("VECTOR DATABASE OPTIMIZATION RESULTS")

    if df.empty or not best_config:
        print("No valid results to display.")
        return

    print("OVERALL STATISTICS")
    print("─" * 70)
    print(f"Total Configurations Tested: {len(df)}")
    print(f"Corpus Size: {best_config.get('num_queries', 'N/A')} documents")
    print(f"Number of Test Queries: {best_config.get('num_queries', 'N/A')}")
    print(f"Iterations per Config: {best_config.get('num_iterations', 'N/A')}")

    print("\nBEST CONFIGURATION (Balanced Optimization)")
    print("─" * 70)
    print("Index Type: HNSW")
    print(f"ef_construction: {best_config.get('ef_construction', 'N/A')}")
    print(f"ef (search): {best_config.get('ef', 'N/A')}")
    print(f"max_connections: {best_config.get('max_connections', 'N/A')}")

    print("\nPERFORMANCE METRICS")
    print("─" * 70)
    print(f"Average Query Time: {best_config.get('avg_query_time_ms', 'N/A')} ms")
    print(f"Std Dev Query Time: {best_config.get('std_query_time_ms', 'N/A')} ms")

    recall_key = [k for k in best_config.keys() if "recall_at" in k]
    if recall_key:
        recall_value = best_config.get(recall_key[0], "N/A")
        k_value = recall_key[0].split("_")[2]
        print(f"Recall@{k_value}: {recall_value}%")

    print(f"Throughput: {best_config.get('throughput_qps', 'N/A')} queries/sec")
    print(f"Insert Time: {best_config.get('insert_time_sec', 'N/A')} sec")

    print("\nTOP 3 CONFIGURATIONS BY QUERY LATENCY")
    print("─" * 70)

    df_sorted = df.sort_values("avg_query_time_ms")
    top_3 = df_sorted.head(3)

    for idx, (_, row) in enumerate(top_3.iterrows(), 1):
        print(
            f"\n{idx}. ef_construction={row['ef_construction']}, "
            f"ef={row['ef']}, max_connections={row['max_connections']}"
        )
        print(f"   Query Time: {row['avg_query_time_ms']:.2f} ms")

        recall_col = [col for col in df.columns if "recall_at" in col]
        if recall_col:
            print(
                f"   {recall_col[0].replace('_', ' ').title()}: "
                f"{row[recall_col[0]]:.1f}%"
            )

    print("\n" + "=" * 70 + "\n")


def main():
    """
    Main execution pipeline for vector database optimization.
    """
    start_time = time.time()

    print_header("VECTOR DATABASE OPTIMIZATION MVP")
    print(
        "This tool benchmarks Weaviate vector database performance\n"
        "for LLM-based applications using different HNSW configurations.\n"
    )

    try:
        # Step 1: Initialize Embedding Generator
        print("Step 1: Initializing Embedding Generator...")
        generator = EmbeddingGenerator()
        embedding_dim = generator.get_embedding_dimension()
        print(f"Embedding dimension: {embedding_dim}\n")

        # Step 2: Load Dataset
        print("Step 2: Loading General Knowledge Dataset...")
        corpus, queries = load_general_knowledge_dataset(
            max_corpus=30000,
            max_queries=3000
        )
        print(f"Loaded {len(corpus)} documents")
        print(f"Loaded {len(queries)} test queries\n")

        # Step 3: Test Weaviate Connection
        print("Step 3: Testing Weaviate Connection...")
        test_db = WeaviateVectorDB(collection_name="ConnectionTest")
        test_db.close()
        print("Weaviate connection successful\n")

        # Step 4: Initialize Benchmark
        print("Step 4: Initializing Benchmark Suite...")
        benchmark = VectorDBBenchmark(generator, corpus, queries)
        print()

        # Step 5: Run Benchmarks
        print("Step 5: Running Benchmark Experiments...")
        print("This may take several minutes...\n")

        results_df = benchmark.run_benchmark(
            ef_construction_values=[64, 128],
            ef_values=[16, 32, 64],
            max_connections_values=[16, 32],
            k=10,
            num_iterations=1
        )

        # Step 6: Analyze Results
        print("\nStep 6: Analyzing Results...")
        best_config = benchmark.get_best_configuration(
            results_df, optimize_for="balanced"
        )
        print("Analysis complete\n")

        # Step 7: Save Results
        print("Step 7: Saving Results...")
        results_df.to_csv("results.csv", index=False)
        print("Results saved to: results.csv\n")

        # Step 8: Display Summary
        print_results_summary(results_df, best_config)

        elapsed_time = time.time() - start_time
        print(f"Total Execution Time: {elapsed_time:.2f} seconds\n")

        print("OPTIMIZATION RECOMMENDATIONS")
        print("─" * 70)
        print("• For low latency: Use lower ef values (16–32)")
        print("• For high recall: Increase ef_construction and ef")
        print("• For balanced performance: Use the selected best configuration")
        print("• Tune max_connections based on memory constraints")
        print("\n" + "=" * 70 + "\n")

        print("Optimization complete! Check results.csv for detailed metrics.\n")

    except Exception as e:
        print(f"\nUnexpected Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
