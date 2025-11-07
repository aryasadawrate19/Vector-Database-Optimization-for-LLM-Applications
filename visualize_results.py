"""
Visualization Script for Vector Database Benchmark Results

This script generates comprehensive visualizations from the benchmark results
to help understand HNSW parameter impacts on performance.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_results(csv_path: str = "results.csv") -> pd.DataFrame:
    """Load benchmark results from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} configurations from {csv_path}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: {csv_path} not found. Please run the benchmark first.")
        exit(1)


def create_visualizations(df: pd.DataFrame, output_dir: str = "visualizations"):
    """Generate all visualizations and save them to output directory."""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    print(f"\n📊 Generating visualizations in '{output_dir}/' directory...")
    
    # 1. Query Time vs Configuration
    plot_query_time_comparison(df, output_dir)
    
    # 2. Heatmap: ef_construction vs ef (Query Time)
    plot_heatmap(df, output_dir)
    
    # 3. Parameter Impact Analysis
    plot_parameter_impact(df, output_dir)
    
    # 4. Throughput Comparison
    plot_throughput_comparison(df, output_dir)
    
    # 5. Recall vs Latency Scatter
    plot_recall_vs_latency(df, output_dir)
    
    # 6. Box Plot: Query Time Distribution by max_connections
    plot_query_time_distribution(df, output_dir)
    
    # 7. Combined Performance Score
    plot_combined_score(df, output_dir)
    
    # 8. Configuration Rankings
    plot_configuration_rankings(df, output_dir)
    
    print(f"\n✅ All visualizations saved to '{output_dir}/' directory!")


def plot_query_time_comparison(df: pd.DataFrame, output_dir: str):
    """Bar chart comparing query times across all configurations."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create configuration labels
    df['config_label'] = df.apply(
        lambda row: f"ef_c={int(row['ef_construction'])}\nef={int(row['ef'])}\nmc={int(row['max_connections'])}", 
        axis=1
    )
    
    # Sort by query time
    df_sorted = df.sort_values('avg_query_time_ms')
    
    # Color bars by performance (green = fast, red = slow)
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(df_sorted)))
    
    bars = ax.bar(range(len(df_sorted)), df_sorted['avg_query_time_ms'], color=colors)
    
    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_ylabel('Average Query Time (ms)', fontweight='bold')
    ax.set_title('Query Time Comparison Across All Configurations', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted['config_label'], rotation=45, ha='right', fontsize=8)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df_sorted['avg_query_time_ms'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=7)
    
    # Highlight best configuration
    best_idx = df_sorted['avg_query_time_ms'].idxmin()
    best_pos = df_sorted.index.get_loc(best_idx)
    bars[best_pos].set_edgecolor('blue')
    bars[best_pos].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/1_query_time_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Query time comparison chart")


def plot_heatmap(df: pd.DataFrame, output_dir: str):
    """Heatmap showing query time for different ef_construction and ef values."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    max_conn_values = sorted(df['max_connections'].unique())
    
    for idx, max_conn in enumerate(max_conn_values):
        df_subset = df[df['max_connections'] == max_conn]
        
        # Pivot table for heatmap
        pivot_table = df_subset.pivot(
            index='ef_construction',
            columns='ef',
            values='avg_query_time_ms'
        )
        
        # Create heatmap
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn_r',
            cbar_kws={'label': 'Query Time (ms)'},
            ax=axes[idx],
            vmin=df['avg_query_time_ms'].min(),
            vmax=df['avg_query_time_ms'].max()
        )
        
        axes[idx].set_title(f'Query Time Heatmap (max_connections={int(max_conn)})', 
                           fontweight='bold', fontsize=12)
        axes[idx].set_xlabel('ef (Search Parameter)', fontweight='bold')
        axes[idx].set_ylabel('ef_construction', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/2_query_time_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Query time heatmap")


def plot_parameter_impact(df: pd.DataFrame, output_dir: str):
    """Box plots showing impact of each parameter on query time."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # ef_construction impact
    df_sorted = df.sort_values('ef_construction')
    sns.boxplot(data=df_sorted, x='ef_construction', y='avg_query_time_ms', 
                palette='Set2', ax=axes[0])
    axes[0].set_title('Impact of ef_construction', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('ef_construction', fontweight='bold')
    axes[0].set_ylabel('Query Time (ms)', fontweight='bold')
    
    # ef impact
    df_sorted = df.sort_values('ef')
    sns.boxplot(data=df_sorted, x='ef', y='avg_query_time_ms', 
                palette='Set2', ax=axes[1])
    axes[1].set_title('Impact of ef (Search Parameter)', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('ef', fontweight='bold')
    axes[1].set_ylabel('Query Time (ms)', fontweight='bold')
    
    # max_connections impact
    df_sorted = df.sort_values('max_connections')
    sns.boxplot(data=df_sorted, x='max_connections', y='avg_query_time_ms', 
                palette='Set2', ax=axes[2])
    axes[2].set_title('Impact of max_connections', fontweight='bold', fontsize=12)
    axes[2].set_xlabel('max_connections', fontweight='bold')
    axes[2].set_ylabel('Query Time (ms)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/3_parameter_impact.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Parameter impact analysis")


def plot_throughput_comparison(df: pd.DataFrame, output_dir: str):
    """Comparison of throughput across configurations."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Sort by throughput
    df_sorted = df.sort_values('throughput_qps', ascending=False)
    
    # Create configuration labels
    df_sorted['config_label'] = df_sorted.apply(
        lambda row: f"ef_c={int(row['ef_construction'])}, ef={int(row['ef'])}, mc={int(row['max_connections'])}", 
        axis=1
    )
    
    # Color bars by performance
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(df_sorted)))
    
    bars = ax.barh(range(len(df_sorted)), df_sorted['throughput_qps'], color=colors)
    
    ax.set_ylabel('Configuration', fontweight='bold')
    ax.set_xlabel('Throughput (queries/sec)', fontweight='bold')
    ax.set_title('Throughput Comparison Across Configurations', fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['config_label'], fontsize=8)
    
    # Add value labels
    for bar, val in zip(bars, df_sorted['throughput_qps']):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', ha='left', va='center', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/4_throughput_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Throughput comparison chart")


def plot_recall_vs_latency(df: pd.DataFrame, output_dir: str):
    """Scatter plot showing recall vs latency tradeoff."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Find recall column
    recall_col = [col for col in df.columns if 'recall_at' in col.lower()][0]
    
    # Create scatter plot with different markers for max_connections
    for max_conn in sorted(df['max_connections'].unique()):
        df_subset = df[df['max_connections'] == max_conn]
        ax.scatter(
            df_subset['avg_query_time_ms'],
            df_subset[recall_col],
            s=100,
            alpha=0.7,
            label=f'max_connections={int(max_conn)}',
            marker='o' if max_conn == 16 else 's'
        )
    
    ax.set_xlabel('Average Query Time (ms)', fontweight='bold')
    ax.set_ylabel('Recall@10 (%)', fontweight='bold')
    ax.set_title('Recall vs Latency Tradeoff', fontsize=14, fontweight='bold')
    ax.legend(title='Configuration', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add best point annotation
    best_idx = df['avg_query_time_ms'].idxmin()
    ax.scatter(
        df.loc[best_idx, 'avg_query_time_ms'],
        df.loc[best_idx, recall_col],
        s=300,
        marker='*',
        color='red',
        edgecolor='black',
        linewidth=2,
        label='Best Config',
        zorder=5
    )
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/5_recall_vs_latency.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Recall vs latency scatter plot")


def plot_query_time_distribution(df: pd.DataFrame, output_dir: str):
    """Box plot showing query time distribution grouped by max_connections."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_sorted = df.sort_values('max_connections')
    
    # Create grouped box plot
    positions = []
    data_to_plot = []
    labels = []
    
    for ef_const in sorted(df['ef_construction'].unique()):
        for max_conn in sorted(df['max_connections'].unique()):
            subset = df[(df['ef_construction'] == ef_const) & (df['max_connections'] == max_conn)]
            if not subset.empty:
                data_to_plot.append(subset['avg_query_time_ms'].values)
                labels.append(f'ef_c={int(ef_const)}\nmc={int(max_conn)}')
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True)
    
    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_xlabel('Configuration (ef_construction, max_connections)', fontweight='bold')
    ax.set_ylabel('Query Time (ms)', fontweight='bold')
    ax.set_title('Query Time Distribution by Configuration', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/6_query_time_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Query time distribution box plot")


def plot_combined_score(df: pd.DataFrame, output_dir: str):
    """Plot showing combined performance score (normalized latency + recall)."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Calculate normalized scores
    recall_col = [col for col in df.columns if 'recall_at' in col.lower()][0]
    
    latency_range = df['avg_query_time_ms'].max() - df['avg_query_time_ms'].min()
    recall_range = df[recall_col].max() - df[recall_col].min()
    
    if latency_range == 0:
        normalized_latency = 1.0
    else:
        normalized_latency = 1 - (df['avg_query_time_ms'] - df['avg_query_time_ms'].min()) / latency_range
    
    if recall_range == 0:
        normalized_recall = 1.0
    else:
        normalized_recall = (df[recall_col] - df[recall_col].min()) / recall_range
    
    # Combined score (50% latency, 50% recall)
    df['combined_score'] = 0.5 * normalized_latency + 0.5 * normalized_recall
    
    # Sort by combined score
    df_sorted = df.sort_values('combined_score', ascending=False)
    
    # Create configuration labels
    df_sorted['config_label'] = df_sorted.apply(
        lambda row: f"ef_c={int(row['ef_construction'])}\nef={int(row['ef'])}\nmc={int(row['max_connections'])}", 
        axis=1
    )
    
    # Color bars
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(df_sorted)))
    
    bars = ax.bar(range(len(df_sorted)), df_sorted['combined_score'], color=colors)
    
    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_ylabel('Combined Performance Score', fontweight='bold')
    ax.set_title('Combined Performance Score (Latency + Recall)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted['config_label'], rotation=45, ha='right', fontsize=8)
    ax.set_ylim([0, 1.1])
    
    # Add value labels
    for bar, val in zip(bars, df_sorted['combined_score']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    
    # Highlight top 3
    for i in range(min(3, len(bars))):
        bars[i].set_edgecolor('red')
        bars[i].set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/7_combined_score.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Combined performance score chart")


def plot_configuration_rankings(df: pd.DataFrame, output_dir: str):
    """Stacked bar chart showing rankings across different metrics."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get recall column
    recall_col = [col for col in df.columns if 'recall_at' in col.lower()][0]
    
    # Rank configurations by different metrics
    df['latency_rank'] = df['avg_query_time_ms'].rank(ascending=True)
    df['recall_rank'] = df[recall_col].rank(ascending=False)
    df['throughput_rank'] = df['throughput_qps'].rank(ascending=False)
    
    # Sort by average rank
    df['avg_rank'] = (df['latency_rank'] + df['recall_rank'] + df['throughput_rank']) / 3
    df_sorted = df.sort_values('avg_rank')
    
    # Create configuration labels
    df_sorted['config_label'] = df_sorted.apply(
        lambda row: f"ef_c={int(row['ef_construction'])}, ef={int(row['ef'])}, mc={int(row['max_connections'])}", 
        axis=1
    )
    
    # Plot horizontal stacked bars
    x = np.arange(len(df_sorted))
    width = 0.8
    
    p1 = ax.barh(x, df_sorted['latency_rank'], width, label='Latency Rank', color='#FF6B6B')
    p2 = ax.barh(x, df_sorted['recall_rank'], width, left=df_sorted['latency_rank'], 
                 label='Recall Rank', color='#4ECDC4')
    p3 = ax.barh(x, df_sorted['throughput_rank'], width, 
                 left=df_sorted['latency_rank'] + df_sorted['recall_rank'],
                 label='Throughput Rank', color='#95E1D3')
    
    ax.set_ylabel('Configuration', fontweight='bold')
    ax.set_xlabel('Cumulative Rank Score (Lower is Better)', fontweight='bold')
    ax.set_title('Configuration Rankings Across All Metrics', fontsize=14, fontweight='bold')
    ax.set_yticks(x)
    ax.set_yticklabels(df_sorted['config_label'], fontsize=8)
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/8_configuration_rankings.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Configuration rankings chart")


def generate_summary_report(df: pd.DataFrame, output_dir: str):
    """Generate a text summary report of findings."""
    report_path = f"{output_dir}/analysis_summary.txt"
    
    recall_col = [col for col in df.columns if 'recall_at' in col.lower()][0]
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("VECTOR DATABASE BENCHMARK - VISUAL ANALYSIS SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write("-" * 70 + "\n\n")
        
        # Best overall
        best_idx = df['avg_query_time_ms'].idxmin()
        f.write("1. BEST OVERALL CONFIGURATION:\n")
        f.write(f"   ef_construction: {int(df.loc[best_idx, 'ef_construction'])}\n")
        f.write(f"   ef: {int(df.loc[best_idx, 'ef'])}\n")
        f.write(f"   max_connections: {int(df.loc[best_idx, 'max_connections'])}\n")
        f.write(f"   Query Time: {df.loc[best_idx, 'avg_query_time_ms']:.2f} ms\n")
        f.write(f"   Recall: {df.loc[best_idx, recall_col]:.1f}%\n")
        f.write(f"   Throughput: {df.loc[best_idx, 'throughput_qps']:.1f} qps\n\n")
        
        # Performance range
        f.write("2. PERFORMANCE RANGE:\n")
        f.write(f"   Query Time: {df['avg_query_time_ms'].min():.2f} - {df['avg_query_time_ms'].max():.2f} ms\n")
        f.write(f"   Recall: {df[recall_col].min():.1f} - {df[recall_col].max():.1f}%\n")
        f.write(f"   Throughput: {df['throughput_qps'].min():.1f} - {df['throughput_qps'].max():.1f} qps\n\n")
        
        # Parameter insights
        f.write("3. PARAMETER IMPACT:\n")
        
        # ef_construction impact
        ef_const_impact = df.groupby('ef_construction')['avg_query_time_ms'].mean()
        f.write(f"   ef_construction:\n")
        for val, time in ef_const_impact.items():
            f.write(f"     {int(val)}: {time:.2f} ms avg\n")
        
        # ef impact
        ef_impact = df.groupby('ef')['avg_query_time_ms'].mean()
        f.write(f"\n   ef:\n")
        for val, time in ef_impact.items():
            f.write(f"     {int(val)}: {time:.2f} ms avg\n")
        
        # max_connections impact
        max_conn_impact = df.groupby('max_connections')['avg_query_time_ms'].mean()
        f.write(f"\n   max_connections:\n")
        for val, time in max_conn_impact.items():
            f.write(f"     {int(val)}: {time:.2f} ms avg\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"  ✓ Analysis summary report saved to {report_path}")


def main():
    """Main function to generate all visualizations."""
    print("\n" + "="*70)
    print(" VECTOR DATABASE BENCHMARK - VISUALIZATION SUITE")
    print("="*70 + "\n")
    
    # Load results
    df = load_results("results.csv")
    
    # Generate visualizations
    create_visualizations(df, "visualizations")
    
    # Generate summary report
    generate_summary_report(df, "visualizations")
    
    print("\n" + "="*70)
    print("📊 Visualization suite complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. Query time comparison bar chart")
    print("  2. Query time heatmap by parameters")
    print("  3. Parameter impact analysis")
    print("  4. Throughput comparison")
    print("  5. Recall vs latency scatter plot")
    print("  6. Query time distribution box plot")
    print("  7. Combined performance score")
    print("  8. Configuration rankings")
    print("  9. Analysis summary report (text file)")
    print("\n✅ All files saved to 'visualizations/' directory")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
