"""
Main Pipeline
Integrates all modules into a complete vector database optimization workflow.
"""

import os
import pandas as pd
from pathlib import Path
from typing import List
import time

from embedding_generator import EmbeddingGenerator
from weaviate_client import WeaviateVectorDB
from benchmark import VectorDBBenchmark


def load_sample_corpus() -> List[str]:
    """
    Load or create a sample text corpus for testing.
    
    Returns:
        List of text passages.
    """
    # Sample corpus of 50 diverse text passages
    corpus = [
        # AI/ML topics
        "Artificial intelligence is revolutionizing industries across the globe.",
        "Machine learning algorithms can identify patterns in vast amounts of data.",
        "Deep learning networks consist of multiple layers of artificial neurons.",
        "Neural networks are inspired by the structure of the human brain.",
        "Supervised learning requires labeled training data for model development.",
        "Unsupervised learning discovers hidden patterns without labeled data.",
        "Reinforcement learning trains agents through trial and error.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret and analyze visual information.",
        "Transfer learning leverages pre-trained models for new tasks.",
        
        # Data Science
        "Data science combines statistics, programming, and domain expertise.",
        "Big data analytics processes massive datasets to extract insights.",
        "Data preprocessing is crucial for building accurate models.",
        "Feature engineering transforms raw data into useful model inputs.",
        "Cross-validation helps prevent overfitting in machine learning models.",
        "Dimensionality reduction simplifies complex high-dimensional data.",
        "Ensemble methods combine multiple models for better predictions.",
        "Gradient descent is an optimization algorithm used in machine learning.",
        "Hyperparameter tuning optimizes model performance and accuracy.",
        "A confusion matrix evaluates classification model performance.",
        
        # Databases
        "Vector databases are optimized for similarity search operations.",
        "Traditional databases use structured queries to retrieve data.",
        "NoSQL databases offer flexible schema designs for diverse data.",
        "Graph databases excel at modeling relationships between entities.",
        "Time-series databases efficiently store temporal data points.",
        "In-memory databases provide extremely fast data access speeds.",
        "Distributed databases scale horizontally across multiple servers.",
        "Database indexing significantly improves query performance.",
        "ACID properties ensure reliable database transactions.",
        "Database normalization reduces data redundancy and inconsistency.",
        
        # Embeddings & Search
        "Word embeddings represent words as dense vectors in continuous space.",
        "Semantic search finds results based on meaning rather than keywords.",
        "Cosine similarity measures the angle between two vectors.",
        "HNSW is an efficient algorithm for approximate nearest neighbor search.",
        "Embedding models capture semantic relationships between words.",
        "Vector quantization compresses embeddings to save storage space.",
        "Dimensionality of embeddings affects model performance and efficiency.",
        "Contextual embeddings vary based on surrounding words in sentences.",
        "Sentence embeddings represent entire sentences as fixed-length vectors.",
        "Cross-encoder models directly compute similarity between text pairs.",
        
        # Cloud & Infrastructure
        "Cloud computing provides on-demand access to computing resources.",
        "Kubernetes orchestrates containerized applications at scale.",
        "Microservices architecture breaks applications into independent services.",
        "API gateways manage and secure API traffic efficiently.",
        "Load balancers distribute network traffic across multiple servers.",
        "Content delivery networks cache content closer to end users.",
        "Serverless computing abstracts away infrastructure management concerns.",
        "Edge computing processes data closer to its source.",
        "Infrastructure as code automates resource provisioning and management.",
        "Continuous integration and deployment streamline software delivery.",
    ]
    
    return corpus


def load_sample_queries() -> List[str]:
    """
    Load sample query texts for benchmarking.
    
    Returns:
        List of query strings.
    """
    queries = [
        "How does machine learning work?",
        "What are neural networks?",
        "Explain vector databases",
        "What is semantic search?",
        "How to optimize database queries?",
        "What is cloud computing?",
        "Explain data preprocessing",
        "What are embeddings?",
        "How does HNSW algorithm work?",
        "What is reinforcement learning?",
    ]
    
    return queries


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70 + "\n")


def print_results_summary(df: pd.DataFrame, best_config: dict):
    """
    Print a formatted summary of benchmark results.
    
    Args:
        df: Results DataFrame.
        best_config: Best configuration dictionary.
    """
    print_header("VECTOR DATABASE OPTIMIZATION RESULTS")
    
    if df.empty or not best_config:
        print("No valid results to display.")
        return
    
    # Overall statistics
    print("OVERALL STATISTICS")
    print(f"{'─'*70}")
    print(f"Total Configurations Tested: {len(df)}")
    print(f"Corpus Size: {best_config.get('num_queries', 'N/A')} documents")
    print(f"Number of Test Queries: {best_config.get('num_queries', 'N/A')}")
    print(f"Iterations per Config: {best_config.get('num_iterations', 'N/A')}")
    
    # Best configuration
    print(f"\nBEST CONFIGURATION (Balanced Optimization)")
    print(f"{'─'*70}")
    print(f"Index Type: HNSW")
    print(f"ef_construction: {best_config.get('ef_construction', 'N/A')}")
    print(f"ef (search): {best_config.get('ef', 'N/A')}")
    print(f"max_connections: {best_config.get('max_connections', 'N/A')}")
    
    # Performance metrics
    print(f"\nPERFORMANCE METRICS")
    print(f"{'─'*70}")
    print(f"Average Query Time: {best_config.get('avg_query_time_ms', 'N/A')} ms")
    print(f"Std Dev Query Time: {best_config.get('std_query_time_ms', 'N/A')} ms")
    
    recall_key = [k for k in best_config.keys() if 'recall_at' in k]
    if recall_key:
        recall_value = best_config.get(recall_key[0], 'N/A')
        k_value = recall_key[0].split('_')[2]
        print(f"Recall@{k_value}: {recall_value}%")
    
    print(f"Throughput: {best_config.get('throughput_qps', 'N/A')} queries/sec")
    print(f"Insert Time: {best_config.get('insert_time_sec', 'N/A')} sec")
    
    # Top 3 configurations
    print(f"\nTOP 3 CONFIGURATIONS BY QUERY LATENCY")
    print(f"{'─'*70}")
    
    df_sorted = df.sort_values('avg_query_time_ms')
    top_3 = df_sorted.head(3)
    
    for idx, (_, row) in enumerate(top_3.iterrows(), 1):
        print(f"\n{idx}. ef_construction={row['ef_construction']}, "
              f"ef={row['ef']}, max_connections={row['max_connections']}")
        print(f"   Query Time: {row['avg_query_time_ms']:.2f} ms")
        
        recall_col = [col for col in df.columns if 'recall_at' in col]
        if recall_col:
            print(f"   {recall_col[0].replace('_', ' ').title()}: {row[recall_col[0]]:.1f}%")
    
    print(f"\n{'='*70}\n")


def main():
    """
    Main execution pipeline for vector database optimization.
    """
    start_time = time.time()
    
    print_header("VECTOR DATABASE OPTIMIZATION MVP")
    print("This tool benchmarks Weaviate vector database performance")
    print("for LLM-based applications using different HNSW configurations.\n")
    
    try:
        # Step 1: Initialize Embedding Generator
        print("Step 1: Initializing Embedding Generator...")
        generator = EmbeddingGenerator()
        embedding_dim = generator.get_embedding_dimension()
        print(f"Embedding dimension: {embedding_dim}\n")
        
        # Step 2: Load Data
        print("Step 2: Loading Sample Corpus and Queries...")
        corpus = load_sample_corpus()
        queries = load_sample_queries()
        print(f"Loaded {len(corpus)} documents")
        print(f"Loaded {len(queries)} test queries\n")
        
        # Step 3: Test Weaviate Connection
        print("Step 3: Testing Weaviate Connection...")
        try:
            test_db = WeaviateVectorDB(collection_name="ConnectionTest")
            test_db.close()  # Use our close method instead
            print("Weaviate connection successful\n")
        except Exception as e:
            print(f"Connection failed: {e}")
            print("\nPlease ensure Weaviate is running on localhost:8081")
            print("   Start Weaviate with: docker-compose up -d")
            return
        
        # Step 4: Initialize Benchmark
        print("Step 4: Initializing Benchmark Suite...")
        benchmark = VectorDBBenchmark(generator, corpus, queries)
        print()
        
        # Step 5: Run Benchmarks
        print("Step 5: Running Benchmark Experiments...")
        print("   This may take several minutes...\n")
        
        results_df = benchmark.run_benchmark(
            ef_construction_values=[64, 128, 256],
            ef_values=[32, 64, 128],
            max_connections_values=[16, 32],
            k=10,
            num_iterations=3
        )
        
        # Step 6: Analyze Results
        print("\nStep 6: Analyzing Results...")
        best_config = benchmark.get_best_configuration(results_df, optimize_for="balanced")
        print("     Analysis complete\n")
        
        # Step 7: Save Results
        print("Step 7: Saving Results...")
        output_file = "results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"     Results saved to: {output_file}\n")
        
        # Step 8: Display Summary
        print_results_summary(results_df, best_config)
        
        # Execution time
        elapsed_time = time.time() - start_time
        print(f"Total Execution Time: {elapsed_time:.2f} seconds\n")
        
        # Recommendations
        print("OPTIMIZATION RECOMMENDATIONS")
        print("─"*70)
        print("• For low latency: Use lower ef values (32-64)")
        print("• For high recall: Use higher ef_construction (256+) and ef (128+)")
        print("• For balanced performance: Use the best configuration shown above")
        print("• Consider memory vs. speed tradeoffs with max_connections")
        print("• Monitor query patterns and adjust ef dynamically if needed")
        print(f"\n{'='*70}\n")
        
        print("Optimization complete! Check results.csv for detailed metrics.\n")
        
    except FileNotFoundError as e:
        print(f"\nError: Required file not found - {e}")
        print("   Please ensure all dependencies are installed.")
        
    except ConnectionError as e:
        print(f"\nConnection Error: {e}")
        print("   Please check your Weaviate instance and API keys.")
        
    except Exception as e:
        print(f"\nUnexpected Error: {e}")
        print("   Please check the error message and try again.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
