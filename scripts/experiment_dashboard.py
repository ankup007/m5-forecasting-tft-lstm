from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - runtime dependency guard.
    raise SystemExit(
        "streamlit is not installed. Install the environment dependencies and try again."
    ) from exc


DEFAULT_ROOT = Path("artifacts/deepar_m5_experiments")
MODE_ORDER = ["mean", "sample-mean", "p25", "p50", "p75"]
VARIANTS = ["raw", "rounded"]


st.set_page_config(page_title="DeepAR Experiment Dashboard", layout="wide")


def find_summary_files(root: Path) -> list[Path]:
    """Return timestamped summary files sorted by most recent modification time."""

    if not root.exists():
        return []
    return sorted(root.glob("summary_*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)


@st.cache_data(show_spinner=False)
def load_csv(path_str: str) -> pd.DataFrame:
    """Load a CSV file with caching."""

    return pd.read_csv(path_str)


def resolve_forecast_path(run_dir: Path, mode: str, variant: str) -> Path | None:
    """Return the forecast artifact path for one mode/variant pair."""

    candidate = run_dir / f"holdout_forecasts_{mode}.csv"
    if variant == "rounded":
        candidate = run_dir / f"holdout_forecasts_{mode}_rounded.csv"
        if not candidate.exists():
            candidate = run_dir / f"holdout_forecasts_{mode}.csv"
    if candidate.exists():
        return candidate
    return None


def build_aggregate_table(summary_row: pd.Series, available_modes: list[str]) -> pd.DataFrame:
    """Create a compact table of run-level metrics for the selected run."""

    metrics = ["mae", "mape", "rmse", "smape", "wape", "rmsse", "wrmsse", "rmsse_l12", "wrmsse_l12"]
    rows = []
    for mode in available_modes:
        for variant in VARIANTS:
            row = {"mode": mode, "variant": variant}
            prefix = f"{mode}_{variant}_"
            for metric in metrics:
                key = f"{prefix}{metric}"
                if key in summary_row.index:
                    row[metric] = summary_row[key]
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index(["mode", "variant"])


def series_metrics_block(series_row: pd.Series, prefix: str = "metric_") -> pd.DataFrame:
    """Build a compact per-series metric table."""

    metrics = ["mae", "mape", "rmse", "smape", "rmsse"]
    data = {}
    for metric in metrics:
        key = f"{prefix}{metric}"
        if key in series_row.index:
            data[metric] = series_row[key]
    return pd.DataFrame([data])


def format_summary_choice(path: Path) -> str:
    """Label a summary file for the sidebar."""

    return path.name


def main() -> None:
    st.title("DeepAR Experiment Dashboard")
    st.caption("Select a sweep summary, pick a run, then inspect holdout forecasts and metrics.")

    root_dir_text = st.sidebar.text_input("Experiment root", value=str(DEFAULT_ROOT))
    root_dir = Path(root_dir_text)
    summary_files = find_summary_files(root_dir)

    if not summary_files:
        st.warning(f"No summary_*.csv files found under {root_dir.resolve()}.")
        st.stop()

    selected_summary_path = st.sidebar.selectbox(
        "Summary file",
        summary_files,
        format_func=format_summary_choice,
    )
    summary_df = load_csv(str(selected_summary_path))

    if "run" not in summary_df.columns:
        st.error(f"{selected_summary_path.name} does not contain a 'run' column.")
        st.stop()

    run_indices = list(summary_df.index)
    selected_run_index = st.sidebar.selectbox(
        "Run",
        run_indices,
        format_func=lambda idx: str(summary_df.loc[idx, "run"]),
    )
    selected_run_row = summary_df.loc[selected_run_index]
    selected_run_name = str(selected_run_row["run"])
    run_dir = selected_summary_path.parent / selected_run_name

    forecast_variant = st.sidebar.radio("Forecast variant", VARIANTS, horizontal=True)

    st.subheader("Selected Run")
    st.write(
        {
            "summary_file": selected_summary_path.name,
            "run": selected_run_name,
            "run_dir": str(run_dir),
        }
    )

    available_modes = []
    for mode in MODE_ORDER:
        if resolve_forecast_path(run_dir, mode, "raw") is not None:
            available_modes.append(mode)

    if not available_modes:
        st.error(f"No holdout forecast files found in {run_dir}.")
        st.stop()

    summary_table = build_aggregate_table(selected_run_row, available_modes)
    st.subheader("Aggregated Scores Across All Series")
    if summary_table.empty:
        st.info("No aggregated metrics were found for this run.")
    else:
        st.dataframe(summary_table, use_container_width=True)

    run_config_cols = [
        col
        for col in summary_df.columns
        if not any(col.startswith(prefix) for prefix in [f"{mode}_{variant}_" for mode in available_modes for variant in VARIANTS])
        and col != "run"
    ]
    if run_config_cols:
        with st.expander("Run configuration"):
            st.dataframe(pd.DataFrame([selected_run_row[run_config_cols]]), use_container_width=True)

    reference_path = resolve_forecast_path(run_dir, available_modes[0], forecast_variant)
    if reference_path is None:
        st.error("Could not load a reference forecast file for the selected run.")
        st.stop()

    reference_df = load_csv(str(reference_path))
    if "id" not in reference_df.columns:
        st.error(f"{reference_path.name} does not contain an 'id' column.")
        st.stop()

    selected_series_id = st.sidebar.selectbox("Series", reference_df["id"].astype(str).tolist())
    series_chart_title = st.sidebar.text_input("Chart title", value=selected_series_id)

    st.subheader("Series View")
    st.write({"series_id": selected_series_id, "forecast_variant": forecast_variant})

    mode_tabs = st.tabs(available_modes)
    for tab, mode in zip(mode_tabs, available_modes):
        with tab:
            forecast_path = resolve_forecast_path(run_dir, mode, forecast_variant)
            if forecast_path is None:
                st.warning(f"Missing {mode} forecast file for {forecast_variant}.")
                continue

            df = load_csv(str(forecast_path))
            series_row = df.loc[df["id"].astype(str) == selected_series_id]
            if series_row.empty:
                st.warning(f"Series {selected_series_id} is not present in {forecast_path.name}.")
                continue
            series_row = series_row.iloc[0]

            forecast_cols = [col for col in df.columns if col.startswith("F") and col[1:].isdigit()]
            actual_cols = [col for col in df.columns if col.startswith("actual_F")]
            days = list(range(1, len(forecast_cols) + 1))
            plot_df = pd.DataFrame(
                {
                    "day": days,
                    "actual": series_row[actual_cols].to_numpy(dtype=float),
                    "forecast": series_row[forecast_cols].to_numpy(dtype=float),
                }
            )

            metric_values = {
                "MAE": series_row.get("metric_mae"),
                "MAPE": series_row.get("metric_mape"),
                "RMSE": series_row.get("metric_rmse"),
                "SMAPE": series_row.get("metric_smape"),
            }

            metric_cols = st.columns(len(metric_values))
            for col, (label, value) in zip(metric_cols, metric_values.items()):
                col.metric(label, f"{value:.4f}" if pd.notna(value) else "n/a")

            st.line_chart(plot_df.set_index("day"), use_container_width=True)
            st.caption(f"{series_chart_title} - {mode} - {forecast_variant}")

            st.dataframe(series_metrics_block(series_row), use_container_width=True)
            with st.expander("Raw selected-series row"):
                st.dataframe(series_row.to_frame().T, use_container_width=True)


if __name__ == "__main__":
    main()
