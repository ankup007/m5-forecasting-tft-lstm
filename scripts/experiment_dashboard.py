from __future__ import annotations

import json
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
SERIES_MATCH_LIMIT = 100


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


@st.cache_data(show_spinner=False)
def load_json(path_str: str) -> dict:
    """Load a JSON file with caching."""

    return json.loads(Path(path_str).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_series_ids_from_csv(path_str: str) -> list[str]:
    """Load only the series ids from a forecast CSV."""

    return pd.read_csv(path_str, usecols=["id"])["id"].astype(str).tolist()


def safe_filename(series_id: str) -> str:
    """Convert a series id into a safe JSON filename."""

    return series_id.replace("/", "__").replace("\\", "__")


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


def build_aggregate_table_from_summary_row(summary_row: pd.Series, available_modes: list[str]) -> pd.DataFrame:
    """Create a compact table of run-level metrics from a flattened summary row."""

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


def build_aggregate_table_from_json(run_summary: dict) -> pd.DataFrame:
    """Create a compact table of run-level metrics from JSON artifacts."""

    rows = []
    aggregate_metrics = run_summary.get("aggregate_metrics", {})
    for mode in MODE_ORDER:
        mode_metrics = aggregate_metrics.get(mode, {})
        for variant in VARIANTS:
            metrics = mode_metrics.get(variant, {})
            if not metrics:
                continue
            row = {"mode": mode, "variant": variant, **metrics}
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index(["mode", "variant"])


def series_metrics_block(series_row: pd.Series, prefix: str = "metric_") -> pd.DataFrame:
    """Build a compact per-series metric table from a CSV row."""

    metrics = ["mae", "mape", "rmse", "smape", "rmsse"]
    data = {}
    for metric in metrics:
        key = f"{prefix}{metric}"
        if key in series_row.index:
            data[metric] = series_row[key]
    return pd.DataFrame([data])


def series_metrics_block_from_json(metrics: dict[str, object]) -> pd.DataFrame:
    """Build a compact per-series metric table from a JSON payload."""

    wanted = ["mae", "mape", "rmse", "smape", "rmsse"]
    return pd.DataFrame([{metric: metrics.get(metric) for metric in wanted}])


def format_summary_choice(path: Path) -> str:
    """Label a summary file for the sidebar."""

    return path.name


def match_series_ids(series_ids: list[str], query: str, limit: int = SERIES_MATCH_LIMIT) -> list[str]:
    """Return a bounded list of series ids matching a case-insensitive substring."""

    cleaned = query.strip().lower()
    if not cleaned:
        return series_ids[:limit]
    matches = [series_id for series_id in series_ids if cleaned in series_id.lower()]
    return matches[:limit]


def render_mode_panel(mode: str, actuals: list[float], forecast: list[float], metrics: dict[str, object], title: str) -> None:
    """Render one forecast mode panel."""

    days = list(range(1, len(forecast) + 1))
    plot_df = pd.DataFrame({"day": days, "actual": actuals, "forecast": forecast})

    metric_values = {
        "MAE": metrics.get("mae"),
        "MAPE": metrics.get("mape"),
        "RMSE": metrics.get("rmse"),
        "SMAPE": metrics.get("smape"),
    }

    st.markdown(f"### {mode}")
    metric_cols = st.columns(len(metric_values))
    for col, (label, value) in zip(metric_cols, metric_values.items()):
        col.metric(label, f"{value:.4f}" if pd.notna(value) else "n/a")

    st.line_chart(plot_df.set_index("day"), use_container_width=True)
    st.caption(title)
    st.dataframe(series_metrics_block_from_json(metrics), use_container_width=True)


def render_mode_panel_from_csv(mode: str, series_row: pd.Series, forecast_cols: list[str], actual_cols: list[str], title: str) -> None:
    """Render one forecast mode panel from a CSV row."""

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

    st.markdown(f"### {mode}")
    metric_cols = st.columns(len(metric_values))
    for col, (label, value) in zip(metric_cols, metric_values.items()):
        col.metric(label, f"{value:.4f}" if pd.notna(value) else "n/a")

    st.line_chart(plot_df.set_index("day"), use_container_width=True)
    st.caption(title)
    st.dataframe(series_metrics_block(series_row), use_container_width=True)


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

    json_dir = run_dir / "series_json"
    json_mode = (json_dir / "series_index.json").exists() and (json_dir / "run_summary.json").exists()

    if json_mode:
        series_index = load_json(str(json_dir / "series_index.json"))
        run_summary = load_json(str(json_dir / "run_summary.json"))
        available_modes = run_summary.get("available_modes", [])
        summary_table = build_aggregate_table_from_json(run_summary)
        run_config_data = run_summary.get("flat_summary", {})
        run_config_frame = pd.DataFrame([run_config_data]) if run_config_data else pd.DataFrame()
        series_ids = series_index.get("series_ids", [])
    else:
        available_modes = [mode for mode in MODE_ORDER if resolve_forecast_path(run_dir, mode, "raw") is not None]
        summary_table = build_aggregate_table_from_summary_row(selected_run_row, available_modes)
        run_config_cols = [
            col
            for col in summary_df.columns
            if not any(col.startswith(prefix) for prefix in [f"{mode}_{variant}_" for mode in available_modes for variant in VARIANTS])
            and col != "run"
        ]
        run_config_frame = pd.DataFrame([selected_run_row[run_config_cols]]) if run_config_cols else pd.DataFrame()
        reference_path = resolve_forecast_path(run_dir, available_modes[0], "raw") if available_modes else None
        if reference_path is None:
            st.error(f"No holdout forecast files found in {run_dir}.")
            st.stop()
        series_ids = load_series_ids_from_csv(str(reference_path))

    if not available_modes:
        st.error(f"No holdout forecast files found in {run_dir}.")
        st.stop()

    st.subheader("Aggregated Scores Across All Series")
    if summary_table.empty:
        st.info("No aggregated metrics were found for this run.")
    else:
        st.dataframe(summary_table, use_container_width=True)

    if not run_config_frame.empty:
        with st.expander("Run configuration"):
            st.dataframe(run_config_frame, use_container_width=True)

    if "dashboard_series_query" not in st.session_state:
        st.session_state.dashboard_series_query = ""
    if "dashboard_selected_series_id" not in st.session_state and series_ids:
        st.session_state.dashboard_selected_series_id = series_ids[0]

    with st.sidebar.form("series_selection_form", clear_on_submit=False):
        series_query = st.text_input(
            "Series search",
            value=st.session_state.dashboard_series_query,
            help="Type part of an id to filter the list, then submit.",
        )
        matched_ids = match_series_ids(series_ids, series_query)
        if not matched_ids:
            st.caption("No series ids matched your search.")
            selected_series_form = None
        else:
            default_series = st.session_state.dashboard_selected_series_id
            default_index = matched_ids.index(default_series) if default_series in matched_ids else 0
            selected_series_form = st.selectbox(
                "Series",
                matched_ids,
                index=default_index,
                help=f"Showing up to {SERIES_MATCH_LIMIT} matches.",
            )
        load_series = st.form_submit_button("Load series")

    if load_series and matched_ids and selected_series_form is not None:
        st.session_state.dashboard_series_query = series_query
        st.session_state.dashboard_selected_series_id = selected_series_form

    selected_series_id = st.session_state.dashboard_selected_series_id
    if selected_series_id not in series_ids and series_ids:
        selected_series_id = series_ids[0]
        st.session_state.dashboard_selected_series_id = selected_series_id
    series_chart_title = selected_series_id

    if not series_ids:
        st.warning("No series ids are available for this run.")
        st.stop()

    if selected_series_id not in series_ids:
        st.warning("Selected series id is not valid for this run.")
        st.stop()

    st.subheader("Series View")
    st.write({"series_id": selected_series_id, "forecast_variant": forecast_variant})

    if json_mode:
        series_path = json_dir / "series" / f"{safe_filename(selected_series_id)}.json"
        if not series_path.exists():
            st.warning(f"Series JSON not found: {series_path.name}")
            st.stop()
        series_payload = load_json(str(series_path))

        with st.expander("Raw selected-series JSON"):
            st.json(series_payload)

        for mode in available_modes:
            mode_payload = series_payload.get("modes", {}).get(mode, {})
            variant_payload = mode_payload.get(forecast_variant) or mode_payload.get("raw")
            if not variant_payload:
                st.warning(f"Missing {mode} forecast data for {forecast_variant}.")
                continue

            actuals = series_payload.get("actuals", [])
            forecast = variant_payload.get("forecast", [])
            render_mode_panel(
                mode,
                actuals,
                forecast,
                variant_payload.get("metrics", {}),
                f"{series_chart_title} - {mode} - {forecast_variant}",
            )
    else:
        forecast_cols_cache: dict[str, list[str]] = {}
        actual_cols_cache: dict[str, list[str]] = {}

        for mode in available_modes:
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

            if mode not in forecast_cols_cache:
                forecast_cols_cache[mode] = [col for col in df.columns if col.startswith("F") and col[1:].isdigit()]
                actual_cols_cache[mode] = [col for col in df.columns if col.startswith("actual_F")]

            render_mode_panel_from_csv(
                mode,
                series_row,
                forecast_cols_cache[mode],
                actual_cols_cache[mode],
                f"{series_chart_title} - {mode} - {forecast_variant}",
            )
            with st.expander(f"Raw selected-series row: {mode}"):
                st.dataframe(series_row.to_frame().T, use_container_width=True)


if __name__ == "__main__":
    main()
