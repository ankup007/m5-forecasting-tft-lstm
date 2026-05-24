from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODE_ORDER = ["mean", "sample-mean", "p25", "p50", "p75"]
VARIANTS = ["raw", "rounded"]
SEARCH_LIMIT = 100


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a generic HTML viewer for all experiment runs.")
    parser.add_argument(
        "--root",
        default="artifacts/deepar_m5_experiments",
        help="Experiment root containing run_* directories and summary_*.csv files.",
    )
    parser.add_argument(
        "--output-html",
        default=None,
        help="Output HTML file. Defaults to <root>/experiment_viewer.html.",
    )
    return parser


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def list_run_dirs(root: Path) -> list[Path]:
    runs = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("run_")]
    return sorted(runs, key=lambda path: path.stat().st_mtime, reverse=True)


def build_manifest(root: Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for run_dir in list_run_dirs(root):
        summary_path = run_dir / "series_json" / "run_summary.json"
        index_path = run_dir / "series_json" / "series_index.json"
        if not summary_path.exists() or not index_path.exists():
            continue
        summary = load_json(summary_path)
        index = load_json(index_path)
        manifest.append(
            {
                "run": run_dir.name,
                "run_dir": run_dir.name,
                "series_count": index.get("series_count", 0),
                "available_modes": summary.get("available_modes", MODE_ORDER),
                "variants": summary.get("variants", VARIANTS),
                "aggregate_metrics": summary.get("aggregate_metrics", {}),
                "series_ids": index.get("series_ids", []),
                "summary": summary,
            }
        )
    return manifest


def build_html(root: Path, manifest: list[dict[str, Any]]) -> str:
    embedded = json.dumps(
        {
            "root": root.as_posix(),
            "runs": manifest,
        },
        indent=2,
    )
    run_options = "".join(
        f'<option value="{item["run"]}">{item["run"]}</option>' for item in manifest
    )
    first_run = manifest[0]["run"] if manifest else ""
    first_series = manifest[0]["series_ids"][0] if manifest and manifest[0]["series_ids"] else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DeepAR Experiment Viewer</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    :root {{
      --bg: #0f1115;
      --panel: #171a21;
      --panel-2: #1f2430;
      --text: #e8ecf1;
      --muted: #9aa4b2;
      --line: #2b3240;
      --accent: #5b8def;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: Segoe UI, Arial, sans-serif; }}
    .page {{ max-width: 1600px; margin: 0 auto; padding: 18px; }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px; }}
    .grid {{ display: grid; grid-template-columns: 360px minmax(0, 1fr); gap: 16px; margin-bottom: 16px; }}
    .field {{ display: grid; gap: 6px; margin-bottom: 10px; }}
    input, select, button {{ width: 100%; border: 1px solid var(--line); background: var(--panel-2); color: var(--text); border-radius: 8px; padding: 10px; font: inherit; }}
    button {{ cursor: pointer; width: auto; }}
    .matches {{ max-height: 280px; overflow: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--panel-2); }}
    .match {{ padding: 8px 10px; border-bottom: 1px solid var(--line); cursor: pointer; font-size: 13px; }}
    .match:hover, .match.active {{ background: rgba(91, 141, 239, 0.18); }}
    .chips {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }}
    .chip {{ background: var(--panel-2); border: 1px solid var(--line); border-radius: 999px; padding: 6px 10px; font-size: 12px; }}
    .table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .table th, .table td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: left; }}
    .table th {{ color: var(--muted); }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-bottom: 12px; }}
    .metric {{ background: var(--panel-2); border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; }}
    .metric .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }}
    .metric .value {{ font-size: 20px; font-weight: 600; margin-top: 2px; }}
    .chart {{ height: 360px; border: 1px solid var(--line); border-radius: 10px; background: #fff; overflow: hidden; }}
    .toolbar {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }}
    .toolbar button.active {{ border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }}
    .muted {{ color: var(--muted); }}
    @media (max-width: 1100px) {{ .grid {{ grid-template-columns: 1fr; }} .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
  </style>
</head>
<body>
  <div class="page">
    <div class="panel" style="margin-bottom:16px;">
      <h1>DeepAR Experiment Viewer</h1>
      <div class="chips">
        <span class="chip">Runs: {len(manifest)}</span>
        <span class="chip">Modes: {", ".join(MODE_ORDER)}</span>
        <span class="chip">Variants: raw, rounded</span>
      </div>
    </div>

    <div class="grid">
      <div class="panel">
        <div class="field">
          <label for="run-select">Run</label>
          <select id="run-select">{run_options}</select>
        </div>
        <div class="field">
          <label for="query">Series search</label>
          <input id="query" type="text" placeholder="Type part of a series id..." />
        </div>
        <div class="field">
          <label>Matches</label>
          <div id="matches" class="matches"></div>
        </div>
        <div class="field">
          <label for="variant">Variant</label>
          <select id="variant">
            <option value="raw">Raw</option>
            <option value="rounded">Rounded</option>
          </select>
        </div>
        <div class="field">
          <button id="reload">Reload selected series</button>
        </div>
        <div class="muted" id="run-label">{first_run}</div>
        <div class="muted" id="series-label">{first_series}</div>
      </div>

      <div class="panel">
        <h2>Aggregate Scores Across All Series</h2>
        <table class="table">
          <thead><tr><th>Mode</th><th>Variant</th><th>MAE</th><th>MAPE</th><th>RMSE</th><th>SMAPE</th><th>RMSSE</th></tr></thead>
          <tbody id="aggregate-body"></tbody>
        </table>
      </div>
    </div>

    <div class="panel">
      <div class="toolbar">
        <strong>Forecast mode</strong>
        <button data-mode="mean" class="mode active">mean</button>
        <button data-mode="sample-mean" class="mode">sample-mean</button>
        <button data-mode="p25" class="mode">p25</button>
        <button data-mode="p50" class="mode">p50</button>
        <button data-mode="p75" class="mode">p75</button>
      </div>
      <div id="metrics" class="metrics"></div>
      <div id="chart" class="chart"></div>
    </div>
  </div>

  <script>
    const DATA = {embedded};
    const RUNS = DATA.runs || [];
    const runSelect = document.getElementById("run-select");
    const queryEl = document.getElementById("query");
    const matchesEl = document.getElementById("matches");
    const variantEl = document.getElementById("variant");
    const metricsEl = document.getElementById("metrics");
    const seriesLabelEl = document.getElementById("series-label");
    const runLabelEl = document.getElementById("run-label");
    const aggregateBodyEl = document.getElementById("aggregate-body");

    let selectedRun = runSelect.value;
    let selectedSeries = {json.dumps(first_series)};
    let selectedMode = "mean";
    let currentPayload = null;

    function metricCard(label, value) {{
      const safe = value === null || value === undefined || Number.isNaN(value) ? "n/a" : Number(value).toFixed(4);
      return `<div class="metric"><div class="label">${{label}}</div><div class="value">${{safe}}</div></div>`;
    }}

    function currentRunData() {{
      return RUNS.find((run) => run.run === selectedRun) || null;
    }}

    function renderAggregate() {{
      const runData = currentRunData();
      const rows = [];
      const aggregate = runData?.aggregate_metrics || {{}};
      for (const mode of {json.dumps(MODE_ORDER)}) {{
        for (const variant of ["raw", "rounded"]) {{
          const metrics = aggregate?.[mode]?.[variant];
          if (!metrics) continue;
          rows.push(`
            <tr>
              <td>${{mode}}</td><td>${{variant}}</td>
              <td>${{metrics.mae ?? "n/a"}}</td>
              <td>${{metrics.mape ?? "n/a"}}</td>
              <td>${{metrics.rmse ?? "n/a"}}</td>
              <td>${{metrics.smape ?? "n/a"}}</td>
              <td>${{metrics.rmsse ?? "n/a"}}</td>
            </tr>
          `);
        }}
      }}
      aggregateBodyEl.innerHTML = rows.join("") || "<tr><td colspan='7'>No aggregate metrics found</td></tr>";
    }}

    function makeMatches(query) {{
      const runData = currentRunData();
      const ids = runData?.series_ids || [];
      const cleaned = (query || "").trim().toLowerCase();
      const matches = cleaned ? ids.filter((id) => id.toLowerCase().includes(cleaned)) : ids.slice(0, {SEARCH_LIMIT});
      return matches.slice(0, {SEARCH_LIMIT});
    }}

    function renderMatches() {{
      const ids = makeMatches(queryEl.value);
      matchesEl.innerHTML = ids.map((id) => `
        <div class="match ${{id === selectedSeries ? 'active' : ''}}" data-series="${{id}}">${{id}}</div>
      `).join("") || '<div class="match muted">No matches</div>';
      matchesEl.querySelectorAll(".match[data-series]").forEach((el) => {{
        el.addEventListener("click", () => {{
          selectedSeries = el.dataset.series;
          seriesLabelEl.textContent = selectedSeries;
          renderMatches();
          loadSeries();
        }});
      }});
    }}

    async function loadSeries() {{
      const runData = currentRunData();
      if (!runData || !selectedSeries) return;
      const runPath = runData.run_dir || runData.run;
      const seriesPath = `${{runPath}}/series_json/series/${{encodeURIComponent(selectedSeries.replaceAll("/", "__").replaceAll("\\\\", "__"))}}.json`;
      seriesLabelEl.textContent = `${{selectedSeries}} (loading...)`;
      try {{
        const response = await fetch(seriesPath, {{ cache: "no-store" }});
        if (!response.ok) throw new Error(`Failed to load ${{seriesPath}}`);
        currentPayload = await response.json();
        seriesLabelEl.textContent = selectedSeries;
        renderSeries();
      }} catch (err) {{
        console.error(err);
        seriesLabelEl.textContent = `${{selectedSeries}} (failed to load)`;
      }}
    }}

    function renderSeries() {{
      if (!currentPayload) return;
      const modePayload = currentPayload.modes?.[selectedMode] || null;
      if (!modePayload) return;
      const variant = variantEl.value;
      const variantPayload = modePayload[variant] || modePayload.raw;
      const actuals = currentPayload.actuals || [];
      const forecast = variantPayload?.forecast || [];
      const metrics = variantPayload?.metrics || {{}};
      const days = forecast.map((_, idx) => idx + 1);
      metricsEl.innerHTML = `
        ${{metricCard("MAE", metrics.mae)}}
        ${{metricCard("MAPE", metrics.mape)}}
        ${{metricCard("RMSE", metrics.rmse)}}
        ${{metricCard("SMAPE", metrics.smape)}}
      `;
      Plotly.react("chart", [
        {{ x: days, y: actuals, type: "scatter", mode: "lines+markers", name: "Actual", line: {{ color: "#111827", width: 2 }} }},
        {{ x: days, y: forecast, type: "scatter", mode: "lines+markers", name: selectedMode + " (" + variant + ")", line: {{ color: "#2563eb", width: 2 }} }}
      ], {{
        margin: {{ l: 55, r: 20, t: 20, b: 45 }},
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        showlegend: true,
        legend: {{ orientation: "h", x: 0, y: 1.08 }},
        xaxis: {{ title: "Forecast step", gridcolor: "#e5e7eb" }},
        yaxis: {{ title: "Units", gridcolor: "#e5e7eb" }}
      }}, {{ responsive: true, displayModeBar: false }});
    }}

    document.querySelectorAll("button.mode").forEach((btn) => {{
      btn.addEventListener("click", () => {{
        document.querySelectorAll("button.mode").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        selectedMode = btn.dataset.mode;
        renderSeries();
      }});
    }});

    runSelect.addEventListener("change", () => {{
      selectedRun = runSelect.value;
      const runData = currentRunData();
      runLabelEl.textContent = selectedRun;
      queryEl.value = "";
      selectedSeries = runData?.series_ids?.[0] || "";
      seriesLabelEl.textContent = selectedSeries;
      renderAggregate();
      renderMatches();
      loadSeries();
    }});

    queryEl.addEventListener("input", renderMatches);
    variantEl.addEventListener("change", renderSeries);
    document.getElementById("reload").addEventListener("click", loadSeries);

    renderAggregate();
    renderMatches();
    if (selectedSeries) {{
      loadSeries().catch((err) => console.error(err));
    }}
  </script>
</body>
</html>
"""


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)
    manifest = build_manifest(root)
    if not manifest:
        raise FileNotFoundError(f"No runs with series_json artifacts found under {root}")
    output_html = Path(args.output_html) if args.output_html else root / "experiment_viewer.html"
    output_html.write_text(build_html(root, manifest), encoding="utf-8")
    print(f"Wrote {output_html}")


if __name__ == "__main__":
    main()
