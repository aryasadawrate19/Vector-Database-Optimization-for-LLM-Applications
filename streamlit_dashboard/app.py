from pathlib import Path
import re
import zipfile

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


st.set_page_config(page_title="Vector DB Benchmark Dashboard", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
VIS_DIR = BASE_DIR / "visualizations"
VIS_1M_DIR = BASE_DIR / "visualizations_1m"


def parse_summary(text: str) -> dict:
    parsed = {}

    patterns = {
        "total_configurations": r"Total configurations evaluated:\s*(\d+)",
        "index_type": r"Index type:\s*(.+)",
        "best_latency_ms": r"Average query time:\s*([\d.]+)\s*ms",
        "best_latency_throughput": r"Throughput:\s*([\d.]+)\s*qps",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            parsed[key] = match.group(1).strip()

    return parsed


def load_summary(summary_file: Path) -> tuple[str, dict]:
    if not summary_file.exists():
        return "Summary file not found.", {}

    content = summary_file.read_text(encoding="utf-8", errors="ignore")
    return content, parse_summary(content)


def get_chart_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.png"))


def friendly_title(file_name: str) -> str:
    name = Path(file_name).stem
    name = re.sub(r"^\d+_", "", name)
    return name.replace("_", " ").title()


def build_1m_dataframe() -> pd.DataFrame:
    data = [
        (64, 16, 16, 24.4, 0.82),
        (64, 16, 32, 29.0, 0.84),
        (128, 16, 16, 26.9, 0.85),
        (64, 64, 16, 31.0, 0.90),
        (128, 16, 32, 31.3, 0.86),
        (64, 32, 32, 30.6, 0.88),
        (64, 32, 16, 35.1, 0.87),
        (64, 64, 32, 35.8, 0.91),
        (128, 32, 16, 40.0, 0.89),
        (128, 64, 16, 43.1, 0.93),
        (128, 64, 32, 43.9, 0.95),
        (128, 128, 32, 47.4, 0.97),
    ]

    df = pd.DataFrame(data, columns=["ef_c", "ef", "M", "latency", "recall"])
    df["throughput"] = 1000 / df["latency"]
    df["config"] = df.apply(lambda x: f"ef_c={int(x.ef_c)}\nef={int(x.ef)}\nm={int(x.M)}", axis=1)
    df["config_inline"] = df.apply(
        lambda x: f"ef_c={int(x.ef_c)}, ef={int(x.ef)}, m={int(x.M)}", axis=1
    )

    df["lat_norm"] = (df["latency"] - df["latency"].min()) / (df["latency"].max() - df["latency"].min())
    df["rec_norm"] = (df["recall"] - df["recall"].min()) / (df["recall"].max() - df["recall"].min())
    df["score"] = df["rec_norm"] - df["lat_norm"]

    return df


def build_1m_summary_text(df: pd.DataFrame) -> str:
    best_latency = df.loc[df["latency"].idxmin()]
    best_recall = df.loc[df["recall"].idxmax()]

    return f"""======================================================================
VECTOR DATABASE BENCHMARK - ANALYSIS SUMMARY (1M DOCUMENTS)
======================================================================

DATASET OVERVIEW
----------------------------------------------------------------------
Total configurations evaluated: {len(df)}
Index type: HNSW (Exact baseline excluded)

BEST CONFIGURATION (LOWEST LATENCY)
----------------------------------------------------------------------
ef_construction: {int(best_latency['ef_c'])}
ef: {int(best_latency['ef'])}
max_connections: {int(best_latency['M'])}
Average query time: {best_latency['latency']:.2f} ms
Recall: {best_latency['recall'] * 100:.2f}%
Throughput: {best_latency['throughput']:.2f} qps

BEST CONFIGURATION (HIGHEST RECALL)
----------------------------------------------------------------------
ef_construction: {int(best_recall['ef_c'])}
ef: {int(best_recall['ef'])}
max_connections: {int(best_recall['M'])}
Recall: {best_recall['recall'] * 100:.2f}%
Average query time: {best_recall['latency']:.2f} ms
Throughput: {best_recall['throughput']:.2f} qps

PARAMETER INSIGHTS
----------------------------------------------------------------------

Impact of ef_construction:
  64 -> {df[df['ef_c'] == 64]['latency'].mean():.2f} ms
  128 -> {df[df['ef_c'] == 128]['latency'].mean():.2f} ms

Impact of ef:
  16 -> {df[df['ef'] == 16]['latency'].mean():.2f} ms
  32 -> {df[df['ef'] == 32]['latency'].mean():.2f} ms
  64 -> {df[df['ef'] == 64]['latency'].mean():.2f} ms
  128 -> {df[df['ef'] == 128]['latency'].mean():.2f} ms

Impact of max_connections:
  16 -> {df[df['M'] == 16]['latency'].mean():.2f} ms
  32 -> {df[df['M'] == 32]['latency'].mean():.2f} ms

======================================================================
"""


def save_plot(fig: plt.Figure, output: Path, use_tight_layout: bool = True) -> None:
    if use_tight_layout:
        fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_1m_visualizations() -> None:
    VIS_1M_DIR.mkdir(parents=True, exist_ok=True)

    df = build_1m_dataframe()

    plt.figure(figsize=(14, 5))
    plt.bar(df["config"], df["latency"])
    plt.title("Query Time Comparison (HNSW Configurations) - 1M")
    plt.ylabel("Latency (ms)")
    plt.xticks(rotation=45)
    save_plot(plt.gcf(), VIS_1M_DIR / "1_query_time_comparison.png")

    pivot = df.pivot_table(values="latency", index="ef", columns="M")
    plt.figure(figsize=(6, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f")
    plt.title("Query Time Heatmap - 1M")
    save_plot(plt.gcf(), VIS_1M_DIR / "2_query_time_heatmap.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    df.groupby("ef_c")["latency"].mean().plot(kind="bar", ax=axes[0])
    axes[0].set_title("Impact of ef_construction")
    axes[0].set_ylabel("Latency (ms)")

    df.groupby("ef")["latency"].mean().plot(kind="bar", ax=axes[1])
    axes[1].set_title("Impact of ef")

    df.groupby("M")["latency"].mean().plot(kind="bar", ax=axes[2])
    axes[2].set_title("Impact of M")

    save_plot(fig, VIS_1M_DIR / "3_parameter_impact.png")

    plt.figure(figsize=(14, 5))
    plt.bar(df["config"], df["throughput"])
    plt.title("Throughput Comparison - 1M")
    plt.ylabel("Queries per second")
    plt.xticks(rotation=45)
    save_plot(plt.gcf(), VIS_1M_DIR / "4_throughput_comparison.png")

    plt.figure(figsize=(6, 5))
    plt.scatter(df["latency"], df["recall"])

    for i in range(len(df)):
        plt.text(df["latency"][i], df["recall"][i], str(int(df["ef"][i])))

    plt.xlabel("Latency (ms)")
    plt.ylabel("Recall")
    plt.title("Recall vs Latency Trade-off - 1M")
    save_plot(plt.gcf(), VIS_1M_DIR / "5_recall_vs_latency.png")

    plt.figure(figsize=(6, 4))
    plt.hist(df["latency"], bins=8)
    plt.title("Query Time Distribution - 1M")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Frequency")
    save_plot(plt.gcf(), VIS_1M_DIR / "6_query_time_distribution.png")

    plt.figure(figsize=(14, 5))
    plt.bar(df["config_inline"], df["score"])
    plt.title("Combined Score (Recall - Latency Tradeoff) - 1M")
    plt.xticks(rotation=45)
    save_plot(plt.gcf(), VIS_1M_DIR / "7_combined_score.png")

    ranked = df.sort_values("score", ascending=False)
    plt.figure(figsize=(11, 6.5))
    plt.barh(ranked["config_inline"], ranked["score"])
    plt.title("Configuration Rankings - 1M")
    plt.xlabel("Score")
    plt.gca().invert_yaxis()
    save_plot(plt.gcf(), VIS_1M_DIR / "8_configuration_rankings.png")

    summary_file = VIS_1M_DIR / "analysis_summary_1m.txt"
    summary_file.write_text(build_1m_summary_text(df), encoding="utf-8")

    zip_path = VIS_1M_DIR / "Visualizations_1M.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for item in sorted(VIS_1M_DIR.iterdir()):
            if item.name == zip_path.name:
                continue
            if item.is_file() and item.suffix.lower() in {".png", ".txt"}:
                zipf.write(item, arcname=item.name)


def render_30k_tab() -> None:
    if not VIS_DIR.exists():
        st.error(f"Visualizations directory not found: {VIS_DIR}")
        return

    summary_text, summary = load_summary(VIS_DIR / "analysis_summary.txt")

    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Configurations", summary.get("total_configurations", "N/A"))
    c2.metric("Index Type", summary.get("index_type", "N/A"))
    c3.metric("Best Avg Latency", f"{summary.get('best_latency_ms', 'N/A')} ms")
    c4.metric("Best Throughput", f"{summary.get('best_latency_throughput', 'N/A')} qps")

    st.divider()

    st.subheader("Charts")
    chart_files = get_chart_files(VIS_DIR)
    if not chart_files:
        st.warning("No chart images found in visualizations folder.")
    else:
        selected_chart = st.selectbox(
            "Jump to chart",
            options=[f.name for f in chart_files],
            format_func=friendly_title,
            key="chart_selector_30k",
        )

        selected_path = VIS_DIR / selected_chart
        st.image(str(selected_path), caption=friendly_title(selected_chart), use_container_width=True)

        with st.expander("Show all charts"):
            for chart in chart_files:
                st.image(str(chart), caption=friendly_title(chart.name), use_container_width=True)

    st.divider()

    st.subheader("Analysis Summary")
    st.text(summary_text)

    zip_file = VIS_DIR / "Visualizations.zip"
    if zip_file.exists():
        with zip_file.open("rb") as fh:
            st.download_button(
                label="Download 30K Visualizations ZIP",
                data=fh,
                file_name=zip_file.name,
                mime="application/zip",
                key="download_30k_zip",
            )


def render_1m_tab() -> None:
    generate_1m_visualizations()

    df = build_1m_dataframe()
    summary_text, summary = load_summary(VIS_1M_DIR / "analysis_summary_1m.txt")

    st.subheader("1M Documents")
    st.caption("Charts generated from your provided 1M base data and packaged as ZIP")

    best_latency = df.loc[df["latency"].idxmin()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Configurations", summary.get("total_configurations", str(len(df))))
    c2.metric("Index Type", summary.get("index_type", "HNSW"))
    c3.metric("Best Avg Latency", f"{best_latency['latency']:.1f} ms")
    c4.metric("Best Throughput", f"{best_latency['throughput']:.1f} qps")

    st.divider()

    st.subheader("Charts")
    chart_files = get_chart_files(VIS_1M_DIR)
    if not chart_files:
        st.warning("No 1M chart images found.")
    else:
        selected_chart = st.selectbox(
            "Jump to chart",
            options=[f.name for f in chart_files],
            format_func=friendly_title,
            key="chart_selector_1m",
        )

        selected_path = VIS_1M_DIR / selected_chart
        st.image(str(selected_path), caption=friendly_title(selected_chart), use_container_width=True)

        with st.expander("Show all charts"):
            for chart in chart_files:
                st.image(str(chart), caption=friendly_title(chart.name), use_container_width=True)

    st.divider()

    st.subheader("Analysis Summary")
    st.text(summary_text)

    zip_file = VIS_1M_DIR / "Visualizations_1M.zip"
    if zip_file.exists():
        with zip_file.open("rb") as fh:
            st.download_button(
                label="Download 1M Visualizations ZIP",
                data=fh,
                file_name=zip_file.name,
                mime="application/zip",
                key="download_1m_zip",
            )


def main() -> None:
    st.title("Vector Database Benchmark Dashboard")
    st.caption("Interactive benchmark views for 30K and 1M document scales")

    tab_30k, tab_1m = st.tabs(["30K Dataset", "1M Documents"])

    with tab_30k:
        render_30k_tab()

    with tab_1m:
        render_1m_tab()


if __name__ == "__main__":
    main()
