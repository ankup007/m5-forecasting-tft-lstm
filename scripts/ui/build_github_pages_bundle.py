from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a GitHub Pages bundle for one experiment run.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a single experiment run directory that already contains series_json/.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs",
        help="Output directory for the GitHub Pages bundle. Defaults to docs/.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    series_json_dir = run_dir / "series_json"
    if not (series_json_dir / "run_summary.json").exists() or not (series_json_dir / "series_index.json").exists():
        raise FileNotFoundError("series_json artifacts are missing. Build them before creating the bundle.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle_run_dir = output_dir / run_dir.name
    bundle_run_dir.mkdir(parents=True, exist_ok=True)
    bundle_series_json_dir = bundle_run_dir / "series_json"
    if not bundle_series_json_dir.exists():
        shutil.copytree(series_json_dir, bundle_series_json_dir)
    else:
        for filename in ["run_summary.json", "series_index.json"]:
            source = series_json_dir / filename
            target = bundle_series_json_dir / filename
            if source.exists():
                shutil.copy2(source, target)
        source_series_dir = series_json_dir / "series"
        target_series_dir = bundle_series_json_dir / "series"
        if source_series_dir.exists():
            target_series_dir.mkdir(parents=True, exist_ok=True)
            for source_file in source_series_dir.glob("*.json"):
                shutil.copy2(source_file, target_series_dir / source_file.name)
    global_naive_dir = run_dir.parent / "naive_forecasts"
    if global_naive_dir.exists():
        bundle_naive_dir = output_dir / "naive_forecasts"
        if not bundle_naive_dir.exists():
            shutil.copytree(global_naive_dir, bundle_naive_dir)
        else:
            for filename in ["run_summary.json", "series_index.json"]:
                source = global_naive_dir / filename
                target = bundle_naive_dir / filename
                if source.exists():
                    shutil.copy2(source, target)
            source_series_dir = global_naive_dir / "series"
            target_series_dir = bundle_naive_dir / "series"
            if source_series_dir.exists():
                target_series_dir.mkdir(parents=True, exist_ok=True)
                for source_file in source_series_dir.glob("*.json"):
                    shutil.copy2(source_file, target_series_dir / source_file.name)

    (output_dir / ".nojekyll").write_text("", encoding="utf-8")

    builder = Path(__file__).parent / "build_experiment_html_viewer.py"
    if not builder.exists():
        raise FileNotFoundError(f"Builder script not found at {builder}")

    import subprocess
    import sys

    subprocess.run([sys.executable, str(builder), "--root", str(output_dir), "--output-html", str(output_dir / "index.html")], check=True)

    index_redirect = output_dir / "index.html"
    if not index_redirect.exists():
        raise FileNotFoundError(index_redirect)

    print(f"Wrote GitHub Pages bundle to {output_dir}")


if __name__ == "__main__":
    main()
