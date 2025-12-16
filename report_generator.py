import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path


def generate_pdf_report(
    results_csv: str = "results.csv",
    output_path: str = "visualizations/vector_db_benchmark_report.pdf"
):
    df = pd.read_csv(results_csv)

    Path("visualizations").mkdir(exist_ok=True)

    with PdfPages(output_path) as pdf:

        # -------------------------
        # Page 1: Title
        # -------------------------
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")

        ax.text(0.5, 0.7, "Vector Database Indexing Benchmark",
                ha="center", fontsize=22, fontweight="bold")
        ax.text(0.5, 0.6, "Weaviate HNSW vs Exact Baseline",
                ha="center", fontsize=14)

        ax.text(
            0.5, 0.45,
            "This report presents a comparative analysis of approximate\n"
            "and exact vector indexing strategies under realistic dataset scale.",
            ha="center", fontsize=11
        )

        pdf.savefig(fig)
        plt.close()

        # -------------------------
        # Page 2: Dataset Summary
        # -------------------------
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")

        ax.text(0.05, 0.9, "Dataset Description", fontsize=16, fontweight="bold")

        ax.text(
            0.05, 0.8,
            "- Dataset: MuskumPillerum / General-Knowledge\n"
            "- Corpus size: ~10,000 documents\n"
            "- Query count: ~1,000 questions\n"
            "- Domain: Open-domain general knowledge\n\n"
            "Empty and invalid entries were filtered during preprocessing "
            "to ensure robust embedding generation.",
            fontsize=11
        )

        pdf.savefig(fig)
        plt.close()

        # -------------------------
        # Page 3: Indexing Strategies
        # -------------------------
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")

        ax.text(0.05, 0.9, "Indexing Strategies Evaluated",
                fontsize=16, fontweight="bold")

        ax.text(
            0.05, 0.8,
            "1. Exact Baseline Index\n"
            "   - Brute-force cosine similarity\n"
            "   - Provides upper bound on recall\n"
            "   - Computationally expensive\n\n"
            "2. HNSW (Weaviate)\n"
            "   - Approximate nearest neighbour graph\n"
            "   - Tunable parameters: ef, efConstruction, maxConnections\n"
            "   - Optimised for low-latency retrieval",
            fontsize=11
        )

        pdf.savefig(fig)
        plt.close()

        # -------------------------
        # Page 4: Latency Comparison
        # -------------------------
        fig, ax = plt.subplots(figsize=(11, 8.5))

        df_sorted = df.sort_values("avg_query_time_ms")
        ax.barh(
            df_sorted.index.astype(str),
            df_sorted["avg_query_time_ms"]
        )

        ax.set_xlabel("Average Query Time (ms)")
        ax.set_title("Query Latency Across Indexing Strategies")

        pdf.savefig(fig)
        plt.close()

        # -------------------------
        # Page 5: Recall Comparison
        # -------------------------
        recall_col = [c for c in df.columns if "recall_at" in c][0]

        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.bar(
            df.index.astype(str),
            df[recall_col]
        )

        ax.set_ylabel("Recall (%)")
        ax.set_title("Recall Comparison Across Indexes")

        pdf.savefig(fig)
        plt.close()

        # -------------------------
        # Page 6: Trade-off Summary
        # -------------------------
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")

        ax.text(0.05, 0.9, "Observations and Trade-offs",
                fontsize=16, fontweight="bold")

        ax.text(
            0.05, 0.8,
            "- Exact search achieves maximum recall at significantly higher latency.\n"
            "- HNSW configurations offer substantial latency reductions with\n"
            "  marginal recall degradation.\n"
            "- Parameter tuning enables balanced trade-offs depending on workload.\n\n"
            "These results demonstrate the necessity of approximate indexing\n"
            "strategies for scalable LLM-backed retrieval systems.",
            fontsize=11
        )

        pdf.savefig(fig)
        plt.close()

    print(f"PDF report generated at: {output_path}")

if __name__ == "__main__":
    generate_pdf_report()
