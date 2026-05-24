from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODE_ORDER = ["mean", "sample-mean", "p25", "p50", "p75"]
VARIANTS = ["raw", "rounded"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a standalone HTML report for one per-series JSON artifact."
    )
    parser.add_argument(
        "--series-json",
        required=True,
        help="Path to a series_json/series/<series_id>.json file.",
    )
    parser.add_argument(
        "--run-summary",
        default=None,
        help="Optional path to run_summary.json. If omitted, the script looks next to the series JSON.",
    )
    parser.add_argument(
        "--output-html",
        default=None,
        help="Path to the output HTML file. Defaults to <series_json stem>.html next to the series JSON.",
    )
    return parser


def json_load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def get_metric_label(metric: str) -> str:
    return metric.upper()


def build_html(payload: dict[str, Any], run_summary: dict[str, Any] | None) -> str:
    embedded = json.dumps({"series": payload, "run_summary": run_summary or {}}, indent=2)
    aggregate_metrics = (run_summary or {}).get("aggregate_metrics", {})
    run_name = (run_summary or {}).get("run", payload.get("run", "run"))
    series_id = payload.get("series_id", "series")
    available_modes = payload.get("modes", {})

    metric_names = ["mae", "mape", "rmse", "smape", "rmsse"]
    metric_headers = "".join(f"<th>{get_metric_label(metric)}</th>" for metric in metric_names)

    aggregate_rows = []
    for mode in MODE_ORDER:
        for variant in VARIANTS:
            metrics = aggregate_metrics.get(mode, {}).get(variant, {})
            if not metrics:
                continue
            cells = "".join(
                f"<td>{metrics.get(metric, 'n/a')}</td>" for metric in metric_names
            )
            aggregate_rows.append(
                f"<tr><td>{mode}</td><td>{variant}</td>{cells}</tr>"
            )

    aggregate_body = "".join(aggregate_rows) if aggregate_rows else "<tr><td colspan='7'>No aggregate metrics found</td></tr>"
    aggregate_table = (
        "<table class='table'>"
        f"<thead><tr><th>Mode</th><th>Variant</th>{metric_headers}</tr></thead>"
        f"<tbody>{aggregate_body}</tbody>"
        "</table>"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DeepAR Series Report - {series_id}</title>
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
      --accent-2: #7dd3fc;
      --good: #22c55e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Segoe UI, Arial, sans-serif;
    }}
    .page {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 20px;
    }}
    .header {{
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 16px;
      margin-bottom: 16px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 16px;
    }}
    .title {{
      font-size: 24px;
      margin: 0 0 8px 0;
    }}
    .subtitle {{
      color: var(--muted);
      margin: 0;
      line-height: 1.5;
    }}
    .chips {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 12px;
    }}
    .chip {{
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      color: var(--text);
    }}
    .controls {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      margin: 12px 0 18px 0;
    }}
    .controls button {{
      background: var(--panel-2);
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 12px;
      cursor: pointer;
    }}
    .controls button.active {{
      border-color: var(--accent);
      box-shadow: inset 0 0 0 1px var(--accent);
    }}
    .section {{
      margin-bottom: 20px;
    }}
    .section h2 {{
      margin: 0 0 8px 0;
      font-size: 18px;
    }}
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
      font-size: 11px;
      color: var(--muted);
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
    .table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .table th, .table td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    .table th {{
      color: var(--muted);
      background: rgba(255,255,255,0.02);
    }}
    @media (max-width: 1100px) {{
      .header {{ grid-template-columns: 1fr; }}
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <div class="panel">
        <h1 class="title">DeepAR Series Report</h1>
        <p class="subtitle">
          Run-level and per-series views from the generated JSON artifacts.
          This page is standalone and can be hosted as static HTML.
        </p>
        <div class="chips">
          <span class="chip">Run: {run_name}</span>
          <span class="chip">Series: {series_id}</span>
          <span class="chip">Horizon: {len(payload.get("horizon", []))}</span>
        </div>
      </div>
      <div class="panel">
        <h2 style="margin-top:0;">Aggregate Scores Across All Series</h2>
        <div style="overflow:auto;">
          {aggregate_table}
        </div>
      </div>
    </div>

    <div class="panel">
      <div class="controls">
        <strong>Forecast variant</strong>
        <button id="btn-raw" class="active" onclick="setVariant('raw')">Raw</button>
        <button id="btn-rounded" onclick="setVariant('rounded')">Rounded</button>
      </div>
      <div id="sections"></div>
    </div>
  </div>

  <script>
    const DATA = {embedded};
    const MODE_ORDER = {json.dumps(MODE_ORDER)};
    const METRICS = ["mae", "mape", "rmse", "smape"];
    let currentVariant = "raw";

    function setVariant(variant) {{
      currentVariant = variant;
      document.getElementById("btn-raw").classList.toggle("active", variant === "raw");
      document.getElementById("btn-rounded").classList.toggle("active", variant === "rounded");
      renderAll();
    }}

    function metricCard(label, value) {{
      const safe = value === null || value === undefined || Number.isNaN(value) ? "n/a" : Number(value).toFixed(4);
      return `<div class="metric"><div class="label">${{label}}</div><div class="value">${{safe}}</div></div>`;
    }}

    function renderMode(mode) {{
      const payload = DATA.series.modes[mode];
      if (!payload) return "";
      const variantPayload = payload[currentVariant] || payload.raw;
      if (!variantPayload) return "";
      const sectionId = `chart-${{mode}}`;
      const cardId = `metrics-${{mode}}`;
      return `
        <div class="section">
          <h2>${{mode}}</h2>
          <div class="metrics" id="${{cardId}}"></div>
          <div class="chart" id="${{sectionId}}"></div>
        </div>
      `;
    }}

    function renderAll() {{
      const container = document.getElementById("sections");
      container.innerHTML = MODE_ORDER.map(renderMode).join("");
      MODE_ORDER.forEach((mode) => {{
        const payload = DATA.series.modes[mode];
        if (!payload) return;
        const variantPayload = payload[currentVariant] || payload.raw;
        if (!variantPayload) return;
        const actuals = DATA.series.actuals || [];
        const forecast = variantPayload.forecast || [];
        const days = forecast.map((_, idx) => idx + 1);
        const metrics = variantPayload.metrics || {{}};
        const traceActual = {{
          x: days,
          y: actuals,
          type: "scatter",
          mode: "lines+markers",
          name: "Actual",
          line: {{ color: "#111827", width: 2 }}
        }};
        const traceForecast = {{
          x: days,
          y: forecast,
          type: "scatter",
          mode: "lines+markers",
          name: mode + " (" + currentVariant + ")",
          line: {{ color: "#2563eb", width: 2 }}
        }};
        const layout = {{
          margin: {{ l: 55, r: 20, t: 20, b: 45 }},
          paper_bgcolor: "#ffffff",
          plot_bgcolor: "#ffffff",
          showlegend: true,
          legend: {{ orientation: "h", x: 0, y: 1.08 }},
          xaxis: {{ title: "Forecast step", gridcolor: "#e5e7eb" }},
          yaxis: {{ title: "Units", gridcolor: "#e5e7eb" }}
        }};
        Plotly.newPlot(`chart-${{mode}}`, [traceActual, traceForecast], layout, {{responsive: true, displayModeBar: false}});
        document.getElementById(`metrics-${{mode}}`).innerHTML = `
          ${{metricCard("MAE", metrics.mae)}}
          ${{metricCard("MAPE", metrics.mape)}}
          ${{metricCard("RMSE", metrics.rmse)}}
          ${{metricCard("SMAPE", metrics.smape)}}
        `;
      }});
    }}

    document.addEventListener("DOMContentLoaded", renderAll);
  </script>
</body>
</html>
"""


def main() -> None:
    args = build_parser().parse_args()
    series_json = Path(args.series_json)
    payload = json_load(series_json)

    if args.run_summary:
        run_summary = json_load(Path(args.run_summary))
    else:
        maybe_summary = series_json.parent.parent / "run_summary.json"
        run_summary = json_load(maybe_summary) if maybe_summary.exists() else None

    output_html = Path(args.output_html) if args.output_html else series_json.with_suffix(".html")
    output_html.write_text(build_html(payload, run_summary), encoding="utf-8")
    print(f"Wrote {output_html}")


if __name__ == "__main__":
    main()
