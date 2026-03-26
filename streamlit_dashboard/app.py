from pathlib import Path
import re

import streamlit as st


st.set_page_config(page_title="Vector DB Benchmark Dashboard", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
VIS_DIR = BASE_DIR / "visualizations"


def parse_summary(text: str) -> dict:
    parsed = {}

    patterns = {
        "total_configurations": r"Total configurations evaluated:\s*(\d+)",
        "index_type": r"Index type:\s*(.+)",
        "best_latency_ms": r"Average query time:\s*([\d.]+)\s*ms",
        "best_latency_recall": r"Recall:\s*([\d.]+)%",
        "best_latency_throughput": r"Throughput:\s*([\d.]+)\s*qps",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            parsed[key] = match.group(1).strip()

    return parsed


def load_summary() -> tuple[str, dict]:
    summary_file = VIS_DIR / "analysis_summary.txt"
    if not summary_file.exists():
        return "Summary file not found.", {}

    content = summary_file.read_text(encoding="utf-8", errors="ignore")
    return content, parse_summary(content)


def get_chart_files() -> list[Path]:
    files = sorted(VIS_DIR.glob("*.png"))
    return files


def friendly_title(file_name: str) -> str:
    name = Path(file_name).stem
    name = re.sub(r"^\d+_", "", name)
    return name.replace("_", " ").title()


def main() -> None:
    st.title("Vector Database Benchmark Dashboard")
    st.caption("Interactive view of benchmark outputs from the visualizations folder")

    if not VIS_DIR.exists():
        st.error(f"Visualizations directory not found: {VIS_DIR}")
        return

    summary_text, summary = load_summary()

    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Configurations", summary.get("total_configurations", "N/A"))
    c2.metric("Index Type", summary.get("index_type", "N/A"))
    c3.metric("Best Avg Latency", f"{summary.get('best_latency_ms', 'N/A')} ms")
    c4.metric("Best Throughput", f"{summary.get('best_latency_throughput', 'N/A')} qps")

    st.divider()

    st.subheader("Charts")
    chart_files = get_chart_files()
    if not chart_files:
        st.warning("No chart images found in visualizations folder.")
    else:
        selected_chart = st.selectbox(
            "Jump to chart",
            options=[f.name for f in chart_files],
            format_func=friendly_title,
        )

        selected_path = VIS_DIR / selected_chart
        st.image(str(selected_path), caption=friendly_title(selected_chart), use_container_width=True)

        with st.expander("Show all charts"):
            for chart in chart_files:
                st.image(str(chart), caption=friendly_title(chart.name), use_container_width=True)

    st.divider()

    st.subheader("Analysis Summary")
    st.text(summary_text)

    pdf_file = VIS_DIR / "vector_db_benchmark_report.pdf"
    if pdf_file.exists():
        with pdf_file.open("rb") as fh:
            st.download_button(
                label="Download Full PDF Report",
                data=fh,
                file_name=pdf_file.name,
                mime="application/pdf",
            )


if __name__ == "__main__":
    main()
