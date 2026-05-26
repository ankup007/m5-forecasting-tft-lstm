from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


MODE_ORDER = ["mean", "sample-mean", "p25", "p50", "p75"]
VARIANTS = ["raw", "rounded"]
SERIES_METRICS = ["mae", "mape", "rmse", "smape", "rmsse"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build per-series JSON artifacts from one experiment run directory."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=None,
        help="Path to a single experiment run directory. If omitted, finds the latest run in --output-root.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write JSON artifacts. Defaults to <run-dir>/series_json.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/deepar_m5_experiments",
        help="Root experiment directory used when searching for the latest run.",
    )
    return parser


def json_safe(value: Any) -> Any:
    """Convert pandas/numpy scalars into JSON-safe Python values."""

    if value is None:
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, (datetime,)):
        return value.isoformat()
    return value


def safe_filename(series_id: str) -> str:
    """Convert a series id into a safe JSON filename."""

    return series_id.replace("/", "__").replace("\\", "__")


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def find_latest_run(root: Path) -> Path:
    runs = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("run_")]
    if not runs:
        raise FileNotFoundError(f"No run directories found under {root}")
    return sorted(runs, key=lambda path: path.stat().st_mtime)[-1]


def select_summary_row(summary_df: pd.DataFrame, run_name: str | None) -> pd.Series:
    if "run" not in summary_df.columns:
        raise ValueError("Summary CSV does not contain a 'run' column.")
    if run_name is not None:
        matches = summary_df.loc[summary_df["run"].astype(str) == run_name]
        if matches.empty:
            raise ValueError(f"Run {run_name!r} was not found in the summary file.")
        return matches.iloc[0]
    if len(summary_df) == 1:
        return summary_df.iloc[0]
    raise ValueError("The summary file contains multiple runs. Pass --run-name to select one.")


def build_nested_aggregate_metrics(summary_row: pd.Series) -> dict[str, dict[str, dict[str, Any]]]:
    """Extract raw/rounded aggregate metrics from a flattened summary row."""

    nested: dict[str, dict[str, dict[str, Any]]] = {}
    for mode in MODE_ORDER:
        mode_dict: dict[str, dict[str, Any]] = {}
        for variant in VARIANTS:
            prefix = f"{mode}_{variant}_"
            metrics = {
                key[len(prefix) :]: json_safe(value)
                for key, value in summary_row.items()
                if str(key).startswith(prefix)
            }
            if metrics:
                mode_dict[variant] = metrics
        if mode_dict:
            nested[mode] = mode_dict
    return nested


def get_forecast_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col.startswith("F") and col[1:].isdigit()]


def get_actual_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col.startswith("actual_F")]


def main() -> None:
    args = build_parser().parse_args()

    root = Path(args.output_root)
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        # Find latest run automatically
        run_dir = find_latest_run(root)
        print(f"No run_dir provided. Using latest run: {run_dir}")

    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    # Validate that we have at least one forecast file
    series_raw_path = run_dir / "holdout_forecasts_mean.csv"
    if not series_raw_path.exists():
        # Try any mode if mean is missing
        fallback = list(run_dir.glob("holdout_forecasts_*.csv"))
        if not fallback:
            raise FileNotFoundError(
                f"No forecast CSVs found in {run_dir}. Run the experiment with holdout evaluation first."
            )
        series_raw_path = fallback[0]

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "series_json"
    series_dir = output_dir / "series"
    series_dir.mkdir(parents=True, exist_ok=True)

    # Resolve summary_row from summary CSV if available
    summary_row: pd.Series | None = None
    summary_file: Path | None = None
    
    # Check parent and root for a summary file that contains this run
    search_dirs = [run_dir.parent, root]
    for d in search_dirs:
        summary_files = sorted(d.glob("summary_*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
        for sf in summary_files:
            try:
                df = pd.read_csv(sf)
                if "run" in df.columns:
                    matches = df.loc[df["run"].astype(str) == run_dir.name]
                    if not matches.empty:
                        summary_row = matches.iloc[0]
                        summary_file = sf
                        break
            except Exception:
                continue
        if summary_row is not None:
            break

    # Fallback if no summary CSV was found
    if summary_row is None:
        print("Warning: No summary CSV entry found for this run. Using minimal metadata.")
        summary_row = pd.Series({"run": run_dir.name})

    mode_frames: dict[str, dict[str, pd.DataFrame]] = {}
    available_modes: list[str] = []
    for mode in MODE_ORDER:
        raw_path = run_dir / f"holdout_forecasts_{mode}.csv"
        if not raw_path.exists():
            continue
        mode_frames[mode] = {"raw": load_csv(raw_path)}
        available_modes.append(mode)
        rounded_path = run_dir / f"holdout_forecasts_{mode}_rounded.csv"
        if rounded_path.exists():
            mode_frames[mode]["rounded"] = load_csv(rounded_path)

    if not available_modes:
        raise FileNotFoundError(f"No holdout forecast files found under {run_dir}")

    base_frame = mode_frames[available_modes[0]]["raw"]
    base_lookup = base_frame.set_index(base_frame["id"].astype(str))
    series_ids = base_lookup.index.astype(str).tolist()
    horizon = list(range(1, len(get_forecast_columns(base_frame)) + 1))
    actual_columns = get_actual_columns(base_frame)
    mode_lookups: dict[str, dict[str, pd.DataFrame]] = {
        mode: {variant: frame.set_index(frame["id"].astype(str)) for variant, frame in variant_frames.items()}
        for mode, variant_frames in mode_frames.items()
    }

    nested_metrics = build_nested_aggregate_metrics(summary_row)
    flat_summary = {str(key): json_safe(value) for key, value in summary_row.items()}

    run_summary = {
        "generated_at": datetime.now().isoformat(),
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "summary_file": str(summary_file) if summary_file is not None else None,
        "available_modes": available_modes,
        "variants": [
            variant
            for variant in VARIANTS
            if any(variant in mode_frames[mode] for mode in available_modes)
        ],
        "aggregate_metrics": nested_metrics,
        "flat_summary": flat_summary,
    }

    index_payload = {
        "generated_at": datetime.now().isoformat(),
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "series_count": len(series_ids),
        "series_ids": series_ids,
        "available_modes": available_modes,
        "variants": run_summary["variants"],
    }

    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    (output_dir / "series_index.json").write_text(json.dumps(index_payload, indent=2), encoding="utf-8")

    for series_id in tqdm(series_ids, desc="Generating JSONs"):
        base_row = base_lookup.loc[series_id]
        series_payload = {
            "series_id": series_id,
            "run": run_dir.name,
            "horizon": horizon,
            "actuals": [json_safe(v) for v in base_row[actual_columns].tolist()],
            "modes": {},
        }

        for mode in available_modes:
            mode_payload: dict[str, Any] = {}
            for variant in VARIANTS:
                df = mode_lookups[mode].get(variant)
                if df is None or series_id not in df.index:
                    continue
                row = df.loc[series_id]
                forecast_cols = get_forecast_columns(df)
                mode_payload[variant] = {
                    "forecast": [json_safe(v) for v in row[forecast_cols].tolist()],
                    "metrics": {
                        metric: json_safe(row.get(f"metric_{metric}"))
                        for metric in SERIES_METRICS
                        if f"metric_{metric}" in row.index
                    },
                }
            if mode_payload:
                series_payload["modes"][mode] = mode_payload

        (series_dir / f"{safe_filename(series_id)}.json").write_text(
            json.dumps(series_payload, indent=2),
            encoding="utf-8",
        )

    print(f"Wrote per-series JSON artifacts to {output_dir}")


if __name__ == "__main__":
    main()
