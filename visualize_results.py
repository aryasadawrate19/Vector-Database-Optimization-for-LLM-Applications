"""
Visualization Script for Vector Database Benchmark Results

Generates comprehensive visualizations from benchmark results
while correctly handling exact baseline vs HNSW configurations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Styling
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_results(csv_path: str = "results.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} configurations from {csv_path}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: {csv_path} not found. Run benchmark first.")
        exit(1)


# ------------------------------------------------------------------
# Visualization orchestration
# ------------------------------------------------------------------

def generate_summary_report(df: pd.DataFrame, output_dir: str):
    """
    Generate a textual analysis summary for HNSW configurations only.
    """
    report_path = Path(output_dir) / "analysis_summary.txt"

    recall_col = [c for c in df.columns if "recall_at" in c][0]

    best_latency_idx = df["avg_query_time_ms"].idxmin()
    best_recall_idx = df[recall_col].idxmax()

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("VECTOR DATABASE BENCHMARK – ANALYSIS SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write("DATASET OVERVIEW\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total configurations evaluated: {len(df)}\n")
        f.write(f"Index type: HNSW (Exact baseline excluded)\n\n")

        f.write("BEST CONFIGURATION (LOWEST LATENCY)\n")
        f.write("-" * 70 + "\n")
        row = df.loc[best_latency_idx]
        f.write(f"ef_construction: {int(row['ef_construction'])}\n")
        f.write(f"ef: {int(row['ef'])}\n")
        f.write(f"max_connections: {int(row['max_connections'])}\n")
        f.write(f"Average query time: {row['avg_query_time_ms']:.2f} ms\n")
        f.write(f"Recall: {row[recall_col]:.2f}%\n")
        f.write(f"Throughput: {row['throughput_qps']:.2f} qps\n\n")

        f.write("BEST CONFIGURATION (HIGHEST RECALL)\n")
        f.write("-" * 70 + "\n")
        row = df.loc[best_recall_idx]
        f.write(f"ef_construction: {int(row['ef_construction'])}\n")
        f.write(f"ef: {int(row['ef'])}\n")
        f.write(f"max_connections: {int(row['max_connections'])}\n")
        f.write(f"Recall: {row[recall_col]:.2f}%\n")
        f.write(f"Average query time: {row['avg_query_time_ms']:.2f} ms\n\n")

        f.write("PARAMETER INSIGHTS\n")
        f.write("-" * 70 + "\n")

        for param in ["ef_construction", "ef", "max_connections"]:
            f.write(f"\nImpact of {param}:\n")
            means = df.groupby(param)["avg_query_time_ms"].mean()
            for k, v in means.items():
                f.write(f"  {int(k)} → {v:.2f} ms\n")

        f.write("\n" + "=" * 70 + "\n")

    print(f"  ✓ Analysis summary written to {report_path}")


def create_visualizations(df: pd.DataFrame, output_dir: str = "visualizations"):
    Path(output_dir).mkdir(exist_ok=True)
    print(f"\n📊 Generating visualizations in '{output_dir}/' directory...")

    # Split data
    df_hnsw = df[df["index_type"] == "hnsw"].copy()

    plot_query_time_comparison(df, output_dir)
    plot_heatmap(df_hnsw, output_dir)
    plot_parameter_impact(df_hnsw, output_dir)
    plot_throughput_comparison(df_hnsw, output_dir)
    plot_recall_vs_latency(df, output_dir)
    plot_query_time_distribution(df_hnsw, output_dir)
    plot_combined_score(df_hnsw, output_dir)
    plot_configuration_rankings(df_hnsw, output_dir)

    print(f"\n✅ All visualizations saved to '{output_dir}/' directory!")
    generate_summary_report(df[df["index_type"] == "hnsw"], output_dir)


# ------------------------------------------------------------------
# Individual plots
# ------------------------------------------------------------------

def plot_query_time_comparison(df: pd.DataFrame, output_dir: str):
    fig, ax = plt.subplots(figsize=(14, 6))

    df = df.copy()
    df["config_label"] = df.apply(
        lambda row:
            "Exact\nBaseline"
            if row["index_type"] == "exact_baseline"
            else f"ef_c={int(row['ef_construction'])}\n"
                 f"ef={int(row['ef'])}\n"
                 f"mc={int(row['max_connections'])}",
        axis=1
    )

    df_sorted = df.sort_values("avg_query_time_ms")
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(df_sorted)))

    bars = ax.bar(range(len(df_sorted)), df_sorted["avg_query_time_ms"], color=colors)

    ax.set_xlabel("Configuration", fontweight="bold")
    ax.set_ylabel("Average Query Time (ms)", fontweight="bold")
    ax.set_title("Query Time Comparison (Exact vs HNSW)", fontweight="bold")
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted["config_label"], rotation=45, ha="right", fontsize=8)

    for bar, val in zip(bars, df_sorted["avg_query_time_ms"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7
        )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/1_query_time_comparison.png", dpi=300)
    plt.close()
    print("  ✓ Query time comparison chart")


def plot_heatmap(df: pd.DataFrame, output_dir: str):
    fig, axes = plt.subplots(1, len(df["max_connections"].unique()), figsize=(16, 6))

    max_conns = sorted(df["max_connections"].unique())

    for idx, mc in enumerate(max_conns):
        subset = df[df["max_connections"] == mc]

        pivot = subset.pivot(
            index="ef_construction",
            columns="ef",
            values="avg_query_time_ms"
        )

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn_r",
            ax=axes[idx],
            cbar_kws={"label": "Query Time (ms)"}
        )

        axes[idx].set_title(f"Query Time Heatmap (max_connections={int(mc)})")
        axes[idx].set_xlabel("ef")
        axes[idx].set_ylabel("ef_construction")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/2_query_time_heatmap.png", dpi=300)
    plt.close()
    print("  ✓ Query time heatmap")


def plot_parameter_impact(df: pd.DataFrame, output_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sns.boxplot(data=df, x="ef_construction", y="avg_query_time_ms", ax=axes[0])
    axes[0].set_title("Impact of ef_construction")

    sns.boxplot(data=df, x="ef", y="avg_query_time_ms", ax=axes[1])
    axes[1].set_title("Impact of ef")

    sns.boxplot(data=df, x="max_connections", y="avg_query_time_ms", ax=axes[2])
    axes[2].set_title("Impact of max_connections")

    for ax in axes:
        ax.set_xlabel(ax.get_xlabel(), fontweight="bold")
        ax.set_ylabel("Query Time (ms)", fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/3_parameter_impact.png", dpi=300)
    plt.close()
    print("  ✓ Parameter impact analysis")


def plot_throughput_comparison(df: pd.DataFrame, output_dir: str):
    fig, ax = plt.subplots(figsize=(14, 6))

    df = df.sort_values("throughput_qps", ascending=False)
    df["config_label"] = df.apply(
        lambda r: f"ef_c={int(r['ef_construction'])}, ef={int(r['ef'])}, mc={int(r['max_connections'])}",
        axis=1
    )

    bars = ax.barh(range(len(df)), df["throughput_qps"])
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["config_label"], fontsize=8)
    ax.set_xlabel("Throughput (queries/sec)")
    ax.set_title("Throughput Comparison")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/4_throughput_comparison.png", dpi=300)
    plt.close()
    print("  ✓ Throughput comparison chart")


def plot_recall_vs_latency(df: pd.DataFrame, output_dir: str):
    recall_col = [c for c in df.columns if "recall_at" in c][0]

    fig, ax = plt.subplots(figsize=(10, 7))

    for idx_type in df["index_type"].unique():
        subset = df[df["index_type"] == idx_type]
        ax.scatter(
            subset["avg_query_time_ms"],
            subset[recall_col],
            s=120,
            label=idx_type.replace("_", " ").title(),
            alpha=0.7
        )

    ax.set_xlabel("Average Query Time (ms)")
    ax.set_ylabel("Recall (%)")
    ax.set_title("Recall vs Latency Tradeoff")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/5_recall_vs_latency.png", dpi=300)
    plt.close()
    print("  ✓ Recall vs latency plot")


def plot_query_time_distribution(df: pd.DataFrame, output_dir: str):
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="max_connections",
        y="avg_query_time_ms",
        hue="ef_construction",
        ax=ax
    )

    ax.set_title("Query Time Distribution by Configuration")
    ax.set_xlabel("max_connections")
    ax.set_ylabel("Query Time (ms)")
    ax.legend(title="ef_construction")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/6_query_time_distribution.png", dpi=300)
    plt.close()
    print("  ✓ Query time distribution plot")


def plot_combined_score(df: pd.DataFrame, output_dir: str):
    recall_col = [c for c in df.columns if "recall_at" in c][0]

    latency_norm = (df["avg_query_time_ms"] - df["avg_query_time_ms"].min()) / (
        df["avg_query_time_ms"].max() - df["avg_query_time_ms"].min()
    )

    recall_norm = (df[recall_col] - df[recall_col].min()) / (
        df[recall_col].max() - df[recall_col].min()
    )

    df["combined_score"] = 0.5 * (1 - latency_norm) + 0.5 * recall_norm
    df = df.sort_values("combined_score", ascending=False)

    df["config_label"] = df.apply(
        lambda r: f"ef_c={int(r['ef_construction'])}\n"
                  f"ef={int(r['ef'])}\n"
                  f"mc={int(r['max_connections'])}",
        axis=1
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(len(df)), df["combined_score"])

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["config_label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Combined Score")
    ax.set_title("Combined Performance Score")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/7_combined_score.png", dpi=300)
    plt.close()
    print("  ✓ Combined performance score chart")


def plot_configuration_rankings(df: pd.DataFrame, output_dir: str):
    recall_col = [c for c in df.columns if "recall_at" in c][0]

    df = df.copy()
    df["latency_rank"] = df["avg_query_time_ms"].rank()
    df["recall_rank"] = df[recall_col].rank(ascending=False)
    df["throughput_rank"] = df["throughput_qps"].rank(ascending=False)
    df["avg_rank"] = (df["latency_rank"] + df["recall_rank"] + df["throughput_rank"]) / 3

    df = df.sort_values("avg_rank")

    df["config_label"] = df.apply(
        lambda r: f"ef_c={int(r['ef_construction'])}, ef={int(r['ef'])}, mc={int(r['max_connections'])}",
        axis=1
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.barh(range(len(df)), df["avg_rank"])

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["config_label"], fontsize=8)
    ax.set_xlabel("Average Rank (Lower is Better)")
    ax.set_title("Configuration Rankings")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/8_configuration_rankings.png", dpi=300)
    plt.close()
    print("  ✓ Configuration rankings chart")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    print("\n" + "=" * 70)
    print(" VECTOR DATABASE BENCHMARK - VISUALIZATION SUITE")
    print("=" * 70 + "\n")

    df = load_results("results.csv")
    create_visualizations(df, "visualizations")

    print("\n" + "=" * 70)
    print("📊 Visualization suite complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
