from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODE_ORDER = ["mean", "sample-mean", "p25", "p50", "p75"]
VARIANTS = ["raw", "rounded"]
METRICS = ["mae", "mape", "rmse", "smape"]
SEARCH_LIMIT = 100


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a standalone HTML viewer for one experiment run.")
    parser.add_argument("--run-dir", required=True, help="Path to a single run directory containing series_json/.")
    parser.add_argument(
        "--output-html",
        default=None,
        help="Output HTML file. Defaults to <run-dir>/run_viewer.html.",
    )
    return parser


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def series_url(series_id: str) -> str:
    safe = series_id.replace("/", "__").replace("\\", "__")
    return f"series_json/series/{safe}.json"


def build_aggregate_rows(run_summary: dict[str, Any]) -> str:
    rows = []
    aggregate_metrics = run_summary.get("aggregate_metrics", {})
    for mode in MODE_ORDER:
        for variant in VARIANTS:
            metrics = aggregate_metrics.get(mode, {}).get(variant, {})
            if not metrics:
                continue
            cells = "".join(f"<td>{metrics.get(metric, 'n/a')}</td>" for metric in ["mae", "mape", "rmse", "smape", "rmsse"])
            rows.append(f"<tr><td>{mode}</td><td>{variant}</td>{cells}</tr>")
    return "".join(rows) if rows else "<tr><td colspan='7'>No aggregate metrics found</td></tr>"


def build_html(run_dir: Path, run_summary: dict[str, Any], series_index: dict[str, Any]) -> str:
    embedded = json.dumps(
        {
            "run_summary": run_summary,
            "series_index": series_index,
            "series_url_prefix": "series_json/series/",
        },
        indent=2,
    )
    aggregate_rows = build_aggregate_rows(run_summary)
    run_name = run_summary.get("run", run_dir.name)
    series_ids = series_index.get("series_ids", [])
    first_series = series_ids[0] if series_ids else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DeepAR Run Viewer - {run_name}</title>
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
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Segoe UI, Arial, sans-serif;
    }}
    .page {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 18px;
    }}
    .top {{
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 16px;
      margin-bottom: 16px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 14px;
    }}
    h1, h2 {{ margin: 0 0 10px 0; }}
    .muted {{ color: var(--muted); }}
    .field {{
      display: grid;
      gap: 6px;
      margin-bottom: 10px;
    }}
    input, select, button {{
      width: 100%;
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--text);
      border-radius: 8px;
      padding: 10px;
      font: inherit;
    }}
    button {{
      cursor: pointer;
    }}
    .matches {{
      max-height: 260px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
    }}
    .match {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      cursor: pointer;
      font-size: 13px;
    }}
    .match:hover, .match.active {{
      background: rgba(91, 141, 239, 0.18);
    }}
    .chips {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 10px;
    }}
    .chip {{
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
    }}
    .table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .table th, .table td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
    }}
    .table th {{ color: var(--muted); }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    .metric {{
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .metric .value {{
      font-size: 20px;
      font-weight: 600;
      margin-top: 2px;
    }}
    .chart {{
      height: 360px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
      overflow: hidden;
    }}
    .section {{
      margin-bottom: 18px;
    }}
    .section h3 {{
      margin: 0 0 8px 0;
    }}
    .toolbar {{
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }}
    .toolbar button {{
      width: auto;
      padding: 8px 12px;
    }}
    .toolbar button.active {{
      border-color: var(--accent);
      box-shadow: inset 0 0 0 1px var(--accent);
    }}
    .status {{
      margin-top: 10px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      white-space: pre-wrap;
    }}
    @media (max-width: 1100px) {{
      .top {{ grid-template-columns: 1fr; }}
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="panel" style="margin-bottom:16px;">
      <h1>DeepAR Run Viewer</h1>
      <div class="muted">{run_name}</div>
      <div class="chips">
        <span class="chip">Series: {len(series_ids)}</span>
        <span class="chip">Modes: {", ".join(MODE_ORDER)}</span>
        <span class="chip">Variants: raw, rounded</span>
      </div>
    </div>

    <div class="top">
      <div class="panel">
        <h2>Pick a series</h2>
        <div class="field">
          <label for="query">Search</label>
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
        <div class="muted" id="selected-label">{first_series or "No series available"}</div>
      </div>

      <div class="panel">
        <h2>Aggregate Scores Across All Series</h2>
        <table class="table">
          <thead>
            <tr><th>Mode</th><th>Variant</th><th>MAE</th><th>MAPE</th><th>RMSE</th><th>SMAPE</th><th>RMSSE</th></tr>
          </thead>
          <tbody>{aggregate_rows}</tbody>
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
    const ALL_SERIES = DATA.series_index.series_ids || [];
    const PREFIX = DATA.series_url_prefix;
    const MODE_ORDER = {json.dumps(MODE_ORDER)};
    let selectedSeries = {json.dumps(first_series)};
    let selectedMode = "mean";

    const queryEl = document.getElementById("query");
    const matchesEl = document.getElementById("matches");
    const variantEl = document.getElementById("variant");
    const selectedLabelEl = document.getElementById("selected-label");
    const metricsEl = document.getElementById("metrics");

    function metricCard(label, value) {{
      const safe = value === null || value === undefined || Number.isNaN(value) ? "n/a" : Number(value).toFixed(4);
      return `<div class="metric"><div class="label">${{label}}</div><div class="value">${{safe}}</div></div>`;
    }}

    function makeMatches(query) {{
      const cleaned = (query || "").trim().toLowerCase();
      const ids = cleaned ? ALL_SERIES.filter((id) => id.toLowerCase().includes(cleaned)) : ALL_SERIES.slice(0, {SEARCH_LIMIT});
      return ids.slice(0, {SEARCH_LIMIT});
    }}

    function renderMatches() {{
      const ids = makeMatches(queryEl.value);
      matchesEl.innerHTML = ids.map((id) => `
        <div class="match ${{id === selectedSeries ? 'active' : ''}}" data-series="${{id}}">
          ${{id}}
        </div>
      `).join("") || '<div class="match muted">No matches</div>';
      matchesEl.querySelectorAll(".match[data-series]").forEach((el) => {{
        el.addEventListener("click", () => {{
          selectedSeries = el.dataset.series;
          selectedLabelEl.textContent = selectedSeries;
          renderMatches();
          loadSeries();
        }});
      }});
    }}

    async function loadSeries() {{
      if (!selectedSeries) return;
      const seriesPath = PREFIX + encodeURIComponent(selectedSeries.replaceAll("/", "__").replaceAll("\\\\", "__")) + ".json";
      selectedLabelEl.textContent = `${{selectedSeries}} (loading...)`;
      try {{
        const response = await fetch(seriesPath, {{ cache: "no-store" }});
        if (!response.ok) {{
          throw new Error(`Failed to load ${{seriesPath}}`);
        }}
        const payload = await response.json();
        selectedLabelEl.textContent = selectedSeries;
        renderSeries(payload);
      }} catch (err) {{
        selectedLabelEl.textContent = `${{selectedSeries}} (load failed)`;
        console.error(err);
      }}
    }}

    function renderSeries(payload) {{
      const mode = selectedMode;
      const variant = variantEl.value;
      const modePayload = payload.modes?.[mode] || null;
      if (!modePayload) return;
      const variantPayload = modePayload[variant] || modePayload.raw;
      const actuals = payload.actuals || [];
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
        {{
          x: days,
          y: actuals,
          type: "scatter",
          mode: "lines+markers",
          name: "Actual",
          line: {{ color: "#111827", width: 2 }}
        }},
        {{
          x: days,
          y: forecast,
          type: "scatter",
          mode: "lines+markers",
          name: mode + " (" + variant + ")",
          line: {{ color: "#2563eb", width: 2 }}
        }}
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
        loadSeries();
      }});
    }});

    document.getElementById("reload").addEventListener("click", loadSeries);
    queryEl.addEventListener("input", renderMatches);
    variantEl.addEventListener("change", loadSeries);

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
    run_dir = Path(args.run_dir)
    run_summary_path = run_dir / "series_json" / "run_summary.json"
    series_index_path = run_dir / "series_json" / "series_index.json"
    if not run_summary_path.exists() or not series_index_path.exists():
        raise FileNotFoundError("Run JSON artifacts not found. Generate series_json first.")

    run_summary = load_json(run_summary_path)
    series_index = load_json(series_index_path)
    output_html = Path(args.output_html) if args.output_html else run_dir / "run_viewer.html"
    output_html.write_text(build_html(run_dir, run_summary, series_index), encoding="utf-8")
    print(f"Wrote {output_html}")


if __name__ == "__main__":
    main()
