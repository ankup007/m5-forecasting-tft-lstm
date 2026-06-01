from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODE_ORDER = ["mean", "sample-mean"]
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
        json_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("series_json")]
        for json_dir in json_dirs:
            summary_path = json_dir / "run_summary.json"
            index_path = json_dir / "series_index.json"
            if not summary_path.exists() or not index_path.exists():
                continue
            try:
                summary = load_json(summary_path)
                index = load_json(index_path)
                display_name = run_dir.name
                if json_dir.name != "series_json":
                    suffix = json_dir.name.replace("series_json_eval_", "").replace("series_json_", "")
                    display_name = f"{display_name} ({suffix})"
                manifest.append(
                    {
                        "run": display_name,
                        "run_id": run_dir.name,
                        "run_dir": run_dir.name,
                        "json_dir": json_dir.name,
                        "series_count": index.get("series_count", 0),
                        "available_modes": summary.get("available_modes", MODE_ORDER),
                        "variants": summary.get("variants", VARIANTS),
                        "aggregate_metrics": summary.get("aggregate_metrics", {}),
                        "series_ids": index.get("series_ids", []),
                        "summary": summary,
                    }
                )
            except Exception as e:
                print(f"Warning: Failed to load manifest for {run_dir.name}/{json_dir.name}: {e}")
                continue
    return manifest


def load_global_baseline(root: Path) -> dict[str, Any]:
    baseline_path = root / "naive_forecasts" / "run_summary.json"
    index_path = root / "naive_forecasts" / "series_index.json"
    data = load_json(baseline_path) if baseline_path.exists() else {}
    data.pop("package_compare", None)
    if index_path.exists():
        data["series_ids"] = load_json(index_path).get("series_ids", [])
    return data


def build_html(root: Path, manifest: list[dict[str, Any]]) -> str:
    baseline_summary = load_global_baseline(root)
    all_series_ids = baseline_summary.get("series_ids", [])
    if not all_series_ids and manifest:
        all_series_ids = manifest[0]["series_ids"]

    embedded = json.dumps(
        {
            "root": root.as_posix(),
            "runs": manifest,
            "baseline_summary": baseline_summary,
            "baseline_root": "naive_forecasts",
            "all_series_ids": all_series_ids,
        },
        indent=2,
    )
    run_options = '<option value="">-- No Run Selected --</option>' + "".join(
        f'<option value="{item["run"]}">{item["run"]}</option>' for item in manifest
    )
    first_series = all_series_ids[0] if all_series_ids else ""
    
    # Use {{ and }} for literal braces in f-string
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>M5 Forecasts - DeepAR Experiment Viewer</title>
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
    .chart {{ height: 420px; border: 1px solid var(--line); border-radius: 10px; background: #fff; overflow: hidden; }}
    .toolbar {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }}
    .toolbar button.active {{ border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }}
    .muted {{ color: var(--muted); }}
    @media (max-width: 1100px) {{ .grid {{ grid-template-columns: 1fr; }} .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
  </style>
</head>
<body>
  <div class="page">
    <div class="panel" style="margin-bottom:16px;">
      <h1>M5 Forecasts - DeepAR Experiment Viewer</h1>
      <div class="muted">Forecasts are generated over the M5 evaluation window d1914-d1941. Plot shows 28 days of history + 28 days of forecast.</div>
      <div class="chips">
        <span class="chip">Runs: {len(manifest)}</span>
        <span class="chip">Modes: {", ".join(MODE_ORDER)}</span>
        <span class="chip">Variants: raw, rounded</span>
      </div>
    </div>

    <div class="panel" style="margin-bottom:16px; border-left: 4px solid var(--accent);">
      <h2>Global Baseline Reference</h2>
      <div class="muted" id="baseline-note" style="margin-bottom:12px;">Reference forecasts (Naive and Seasonal Naive) are computed once for the dataset.</div>
      <table class="table">
        <thead><tr><th>Baseline</th><th>MAE</th><th>MAPE</th><th>RMSE</th><th>SMAPE</th><th>RMSSE</th><th>WRMSSE</th></tr></thead>
        <tbody id="baseline-body"></tbody>
      </table>
    </div>

    <div class="grid">
      <div class="panel">
        <div class="field">
          <label for="run-select">Selected DeepAR Run</label>
          <select id="run-select">{run_options}</select>
        </div>
        <div class="field">
          <label for="dept-filter">Department filter</label>
          <select id="dept-filter">
            <option value="">All departments</option>
          </select>
        </div>
        <div class="field">
          <label for="store-filter">Store filter</label>
          <select id="store-filter">
            <option value="">All stores</option>
          </select>
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
        <div class="muted" id="run-label" style="font-size:11px; margin-top:10px;">None</div>
        <div class="muted" id="series-label" style="font-size:11px;">{first_series}</div>
      </div>

      <div class="panel">
        <h2>DeepAR Run Aggregate Scores</h2>
        <div class="muted" style="margin-bottom:12px;">Metrics averaged across all 30,490 series.</div>
        <table class="table">
          <thead><tr><th>Mode</th><th>Variant</th><th>MAE</th><th>MAPE</th><th>RMSE</th><th>SMAPE</th><th>RMSSE</th><th>WRMSSE</th></tr></thead>
          <tbody id="aggregate-body"></tbody>
        </table>
      </div>
    </div>

    <div class="panel">
      <div class="toolbar">
        <strong>Forecast mode</strong>
        <button data-mode="mean" class="mode active">mean</button>
        <button data-mode="sample-mean" class="mode">sample-mean</button>
      </div>
      <div id="metrics" class="metrics"></div>
      <div style="margin-top:12px;">
        <h3 style="margin: 0 0 8px 0;">Series Metrics: <span id="current-series-id" style="color:var(--accent);">{first_series}</span></h3>
        <table class="table">
          <thead><tr><th>Source</th><th>MAE</th><th>MAPE</th><th>RMSE</th><th>SMAPE</th><th>RMSSE</th></tr></thead>
          <tbody id="series-metrics-body"></tbody>
        </table>
      </div>
      <div id="chart" class="chart" style="margin-top:16px;"></div>
    </div>
  </div>

  <script>
    const DATA = {embedded};
    const RUNS = DATA.runs || [];
    const GLOBAL_BASELINE = DATA.baseline_summary || {{}};
    const BASELINE_ROOT = DATA.baseline_root || "naive_forecasts";
    const ALL_SERIES_IDS = DATA.all_series_ids || [];
    
    const runSelect = document.getElementById("run-select");
    const deptFilterEl = document.getElementById("dept-filter");
    const storeFilterEl = document.getElementById("store-filter");
    const queryEl = document.getElementById("query");
    const matchesEl = document.getElementById("matches");
    const variantEl = document.getElementById("variant");
    const metricsEl = document.getElementById("metrics");
    const seriesMetricsBodyEl = document.getElementById("series-metrics-body");
    const seriesLabelEl = document.getElementById("series-label");
    const currentSeriesIdEl = document.getElementById("current-series-id");
    const runLabelEl = document.getElementById("run-label");
    const aggregateBodyEl = document.getElementById("aggregate-body");
    const baselineBodyEl = document.getElementById("baseline-body");
    const baselineNoteEl = document.getElementById("baseline-note");

    let selectedRun = runSelect.value;
    let selectedSeries = {json.dumps(first_series)};
    let selectedMode = "mean";
    let currentPayload = null;
    let currentBaselinePayload = null;
    let currentWrmsse = null;

    function fmt(val) {{
      if (val === null || val === undefined || val === "" || Number.isNaN(Number(val))) return "n/a";
      return Number(val).toFixed(2);
    }}

    function metricCard(label, value) {{
      return `<div class="metric"><div class="label">${{label}}</div><div class="value">${{fmt(value)}}</div></div>`;
    }}

    function currentRunData() {{
      return RUNS.find((run) => run.run === selectedRun) || null;
    }}

    function parseSeriesFilters(seriesId) {{
      const parts = String(seriesId || "").split("_");
      if (parts.length < 6) return {{ dept: "", store: "" }};
      return {{
        dept: `${{parts[0]}}_${{parts[1]}}`,
        store: `${{parts[parts.length - 3]}}_${{parts[parts.length - 2]}}`,
      }};
    }}

    function filterOptionSets() {{
      const runData = currentRunData();
      const ids = runData?.series_ids || ALL_SERIES_IDS || [];
      const depts = new Set();
      const stores = new Set();
      for (const id of ids) {{
        const parsed = parseSeriesFilters(id);
        if (parsed.dept) depts.add(parsed.dept);
        if (parsed.store) stores.add(parsed.store);
      }}
      return {{
        depts: Array.from(depts).sort(),
        stores: Array.from(stores).sort(),
      }};
    }}

    function renderFilterOptions() {{
      const opts = filterOptionSets();
      const currentDept = deptFilterEl.value;
      const currentStore = storeFilterEl.value;

      deptFilterEl.innerHTML = `<option value="">All departments</option>` +
        opts.depts.map((dept) => `<option value="${{dept}}">${{dept}}</option>`).join("");
      storeFilterEl.innerHTML = `<option value="">All stores</option>` +
        opts.stores.map((store) => `<option value="${{store}}">${{store}}</option>`).join("");

      if (opts.depts.includes(currentDept)) deptFilterEl.value = currentDept;
      if (opts.stores.includes(currentStore)) storeFilterEl.value = currentStore;
    }}

    function renderAggregate() {{
      const runData = currentRunData();
      if (!runData) {{
        aggregateBodyEl.innerHTML = "<tr><td colspan='8'>Select a run to see aggregate metrics</td></tr>";
        return;
      }}
      const rows = [];
      const aggregate = runData?.aggregate_metrics || {{}};
      const wrmsseData = currentWrmsse || {{}};

      for (const mode of {json.dumps(MODE_ORDER)}) {{
        for (const variant of ["raw", "rounded"]) {{
          const metrics = aggregate?.[mode]?.[variant];
          if (!metrics) continue;
          
          const wrmsseVal = wrmsseData[mode]?.[variant]?.wrmsse;
          
          rows.push(`
            <tr>
              <td>${{mode}}</td><td>${{variant}}</td>
              <td>${{fmt(metrics.mae)}}</td>
              <td>${{fmt(metrics.mape)}}</td>
              <td>${{fmt(metrics.rmse)}}</td>
              <td>${{fmt(metrics.smape)}}</td>
              <td>${{fmt(metrics.rmsse)}}</td>
              <td style="font-weight:bold; color:var(--accent);">${{fmt(wrmsseVal)}}</td>
            </tr>
          `);
        }}
      }}
      aggregateBodyEl.innerHTML = rows.join("") || "<tr><td colspan='8'>No aggregate metrics found</td></tr>";
    }}

    function renderBaselines() {{
      const baselines = GLOBAL_BASELINE.baselines || {{
        naive: GLOBAL_BASELINE.naive,
        seasonal_naive: GLOBAL_BASELINE.seasonal_naive,
      }};
      if (!Object.keys(baselines || {{}}).length) {{
        baselineBodyEl.innerHTML = "<tr><td colspan='7'>No baseline summary found</td></tr>";
        baselineNoteEl.textContent = "Baseline artifacts are not available.";
        return;
      }}
      const rows = [];
      for (const key of ["naive", "seasonal_naive"]) {{
        const metrics = baselines?.[key];
        if (!metrics) continue;
        rows.push(`
          <tr>
            <td>${{key.replace("_", " ")}}</td>
            <td>${{fmt(metrics.mae)}}</td>
            <td>${{fmt(metrics.mape)}}</td>
            <td>${{fmt(metrics.rmse)}}</td>
            <td>${{fmt(metrics.smape)}}</td>
            <td>${{fmt(metrics.rmsse)}}</td>
            <td>${{fmt(metrics.wrmsse)}}</td>
          </tr>
        `);
      }}
      baselineBodyEl.innerHTML = rows.join("") || "<tr><td colspan='7'>No baseline summary found</td></tr>";
      baselineNoteEl.textContent = "Global reference forecasts (Naive and Seasonal Naive) are computed once for the dataset.";
    }}

    function makeMatches(query) {{
      const runData = currentRunData();
      const ids = runData?.series_ids || ALL_SERIES_IDS || [];
      const cleaned = (query || "").trim().toLowerCase();
      const selectedDept = deptFilterEl.value;
      const selectedStore = storeFilterEl.value;
      const matches = ids.filter((id) => {{
        const parsed = parseSeriesFilters(id);
        if (selectedDept && parsed.dept !== selectedDept) return false;
        if (selectedStore && parsed.store !== selectedStore) return false;
        if (cleaned && !id.toLowerCase().includes(cleaned)) return false;
        return true;
      }});
      return matches.slice(0, {SEARCH_LIMIT});
    }}

    function renderMatches() {{
      const ids = makeMatches(queryEl.value);
      if (ids.length && !ids.includes(selectedSeries)) {{
        selectedSeries = ids[0];
        seriesLabelEl.textContent = selectedSeries;
        if (currentSeriesIdEl) currentSeriesIdEl.textContent = selectedSeries;
      }}
      matchesEl.innerHTML = ids.map((id) => `
        <div class="match ${{id === selectedSeries ? 'active' : ''}}" data-series="${{id}}">${{id}}</div>
      `).join("") || '<div class="match muted">No matches</div>';
      matchesEl.querySelectorAll(".match[data-series]").forEach((el) => {{
        el.addEventListener("click", () => {{
          selectedSeries = el.dataset.series;
          seriesLabelEl.textContent = selectedSeries;
          if (currentSeriesIdEl) currentSeriesIdEl.textContent = selectedSeries;
          renderMatches();
          loadSeries();
        }});
      }});
    }}

    async function loadSeries() {{
      if (!selectedSeries) return;
      const runData = currentRunData();
      const encodedId = encodeURIComponent(selectedSeries.replaceAll("/", "__").replaceAll("\\\\", "__"));
      const baselinePath = `${{BASELINE_ROOT}}/series/${{encodedId}}.json`;
      
      seriesLabelEl.textContent = `${{selectedSeries}} (loading...)`;
      currentPayload = null;
      currentBaselinePayload = null;

      const tasks = [
        fetch(baselinePath, {{ cache: "no-store" }}).then(r => r.ok ? r.json() : null).catch(() => null)
      ];

      if (runData) {{
        const runPath = runData.run_dir || runData.run;
        const jsonDir = runData.json_dir || "series_json";
        const seriesPath = `${{runPath}}/${{jsonDir}}/series/${{encodedId}}.json`;
        tasks.push(fetch(seriesPath, {{ cache: "no-store" }}).then(r => r.ok ? r.json() : null).catch(() => null));
      }}

      const [baseline, runPayload] = await Promise.all(tasks);
      currentBaselinePayload = baseline;
      currentPayload = runPayload;

      seriesLabelEl.textContent = selectedSeries + (currentPayload ? "" : " (Run data missing)");
      if (currentSeriesIdEl) currentSeriesIdEl.textContent = selectedSeries;
      renderSeries();
    }}

    async function loadRunWrmsse() {{
      const runData = currentRunData();
      currentWrmsse = null;
      if (!runData) {{
        renderAggregate();
        return;
      }}
      const runPath = runData.run_dir || runData.run;
      const jsonDir = runData.json_dir || "series_json";
      const wrmssePath = `${{runPath}}/${{jsonDir}}/wrmsse.json`;
      try {{
        const resp = await fetch(wrmssePath, {{ cache: "no-store" }});
        if (resp.ok) {{
          currentWrmsse = await resp.json();
        }}
      }} catch (e) {{
        console.warn("Failed to load WRMSSE for run", e);
      }}
      renderAggregate();
    }}

    function renderSeries() {{
      if (!currentPayload && !currentBaselinePayload) {{
        metricsEl.innerHTML = "<div class='muted'>No data available for this series</div>";
        return;
      }}
      
      const variant = variantEl.value;
      const modePayload = currentPayload?.modes?.[selectedMode] || null;
      const variantPayload = modePayload?.[variant] || modePayload?.raw || null;
      
      const baselineNaive = currentBaselinePayload?.baselines?.naive || {{}};
      const baselineSeasonal = currentBaselinePayload?.baselines?.seasonal_naive || {{}};
      
      const preHoldoutActuals = currentPayload?.pre_holdout_actuals || currentBaselinePayload?.pre_holdout_actuals || [];
      const holdoutActuals = currentPayload?.actuals || currentBaselinePayload?.actuals || [];
      const combinedActuals = [...preHoldoutActuals, ...holdoutActuals];
      
      const forecast = variantPayload?.forecast || [];
      const metrics = variantPayload?.metrics || {{}};
      
      const naiveForecast = baselineNaive.forecast || [];
      const seasonalForecast = baselineSeasonal.forecast || [];
      
      const preHoldoutLen = preHoldoutActuals.length;
      const actualX = combinedActuals.map((_, idx) => idx + 1);
      const forecastX = forecast.map((_, idx) => preHoldoutLen + idx + 1);
      const naiveX = naiveForecast.map((_, idx) => preHoldoutLen + idx + 1);
      const seasonalX = seasonalForecast.map((_, idx) => preHoldoutLen + idx + 1);

      metricsEl.innerHTML = `
        ${{metricCard("MAE", metrics.mae)}}
        ${{metricCard("MAPE", metrics.mape)}}
        ${{metricCard("RMSE", metrics.rmse)}}
        ${{metricCard("SMAPE", metrics.smape)}}
      `;
      
      const rows = [];
      if (currentPayload) {{
        rows.push(`<tr><td>DeepAR (${{selectedMode}}, ${{variant}})</td><td>${{fmt(metrics.mae)}}</td><td>${{fmt(metrics.mape)}}</td><td>${{fmt(metrics.rmse)}}</td><td>${{fmt(metrics.smape)}}</td><td>${{fmt(metrics.rmsse)}}</td></tr>`);
      }} else {{
        rows.push(`<tr><td colspan="6" class="muted">No DeepAR run data selected or found</td></tr>`);
      }}
      
      rows.push(`<tr><td>Naive</td><td>${{fmt(baselineNaive.metrics?.mae)}}</td><td>${{fmt(baselineNaive.metrics?.mape)}}</td><td>${{fmt(baselineNaive.metrics?.rmse)}}</td><td>${{fmt(baselineNaive.metrics?.smape)}}</td><td>${{fmt(baselineNaive.metrics?.rmsse)}}</td></tr>`);
      rows.push(`<tr><td>Seasonal naive</td><td>${{fmt(baselineSeasonal.metrics?.mae)}}</td><td>${{fmt(baselineSeasonal.metrics?.mape)}}</td><td>${{fmt(baselineSeasonal.metrics?.rmse)}}</td><td>${{fmt(baselineSeasonal.metrics?.smape)}}</td><td>${{fmt(baselineSeasonal.metrics?.rmsse)}}</td></tr>`);
      
      seriesMetricsBodyEl.innerHTML = rows.join("");

      const traces = [];
      if (combinedActuals.length) {{
        traces.push({{ 
          x: actualX, 
          y: combinedActuals, 
          type: "scatter", 
          mode: "lines+markers", 
          name: "Actual", 
          line: {{ color: "#111827", width: 2 }},
          marker: {{ size: 4 }}
        }});
      }}

      if (forecast.length) {{
        traces.push({{ 
          x: forecastX, 
          y: forecast, 
          type: "scatter", 
          mode: "lines+markers", 
          name: "DeepAR (" + selectedMode + ")", 
          line: {{ color: "#2563eb", width: 2 }},
          marker: {{ size: 5 }}
        }});
      }}

      if (naiveForecast.length) {{
        traces.push({{ 
          x: naiveX, 
          y: naiveForecast, 
          type: "scatter", 
          mode: "lines+markers", 
          name: "Naive", 
          line: {{ color: "#dc2626", width: 2, dash: "dot" }},
          marker: {{ size: 4 }}
        }});
      }}
      if (seasonalForecast.length) {{
        traces.push({{ 
          x: seasonalX, 
          y: seasonalForecast, 
          type: "scatter", 
          mode: "lines+markers", 
          name: "Seasonal naive", 
          line: {{ color: "#16a34a", width: 2, dash: "dash" }},
          marker: {{ size: 4 }}
        }});
      }}

      const shapes = [];
      if (preHoldoutLen > 0) {{
        shapes.push({{
          type: 'line',
          x0: preHoldoutLen + 0.5,
          y0: 0,
          x1: preHoldoutLen + 0.5,
          y1: 1,
          yref: 'paper',
          line: {{
            color: 'rgba(0, 0, 0, 0.3)',
            width: 1.5,
            dash: 'dash'
          }}
        }});
      }}

      Plotly.react("chart", traces, {{
        margin: {{ l: 55, r: 20, t: 20, b: 45 }},
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        showlegend: true,
        legend: {{ orientation: "h", x: 0, y: 1.08 }},
        xaxis: {{ 
          title: "Day index", 
          gridcolor: "#e5e7eb",
          tickmode: "auto",
          nticks: 20
        }},
        yaxis: {{ title: "Units", gridcolor: "#e5e7eb" }},
        shapes: shapes
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
      runLabelEl.textContent = selectedRun || "None";
      renderFilterOptions();
      loadRunWrmsse();
      renderMatches();
      loadSeries();
    }});

    queryEl.addEventListener("input", renderMatches);
    deptFilterEl.addEventListener("change", () => {{
      renderMatches();
      loadSeries();
    }});
    storeFilterEl.addEventListener("change", () => {{
      renderMatches();
      loadSeries();
    }});
    variantEl.addEventListener("change", renderSeries);
    document.getElementById("reload").addEventListener("click", loadSeries);

    loadRunWrmsse();
    renderBaselines();
    renderFilterOptions();
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
    output_html = Path(args.output_html) if args.output_html else root / "experiment_viewer.html"
    output_html.write_text(build_html(root, manifest), encoding="utf-8")
    print(f"Wrote {output_html}")


if __name__ == "__main__":
    main()
