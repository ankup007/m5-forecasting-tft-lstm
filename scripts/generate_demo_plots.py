from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

import argparse

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate multi-mode plots for a DeepAR run.")
    parser.add_argument("--run-dir", help="Path to the specific run directory. If omitted, uses latest in artifacts/deepar_m5.")
    return parser

def main():
    args = build_parser().parse_args()
    
    base_dir = Path("artifacts/deepar_m5")
    
    if args.run_dir:
        artifact_dir = Path(args.run_dir)
    else:
        # Automatically find the latest run subfolder
        run_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        if not run_dirs:
            print(f"Error: No run directories found in {base_dir}")
            return
        # Sort by name (which starts with timestamp)
        artifact_dir = sorted(run_dirs)[-1]
        print(f"No run-dir specified. Using latest run: {artifact_dir}")

    modes = ["mean", "sample-mean", "p25", "p75"]
    
    # Load all forecast data
    dfs = {}
    for mode in modes:
        path = artifact_dir / f"holdout_forecasts_{mode}.csv"
        if not path.exists():
            print(f"Warning: {path} not found. Skipping mode {mode}.")
            continue
        dfs[mode] = pd.read_csv(path)

    if "mean" not in dfs:
        print("Error: primary 'mean' forecast file not found. Did you run the training with --eval-holdout?")
        return

    output_dir = artifact_dir / "plots_multi"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the 'mean' DF as the primary index
    df_main = dfs["mean"]
    f_cols = [c for c in df_main.columns if c.startswith("F") and c[1:].isdigit()]
    a_cols = [c for c in df_main.columns if c.startswith("actual_F")]
    
    print(f"Generating multi-mode plots for the first 10 series in {output_dir}")
    
    for i in range(min(10, len(df_main))):
        series_id = df_main.iloc[i]["id"]
        plt.figure(figsize=(14, 7))
        
        days = np.arange(1, len(f_cols) + 1)
        actuals = df_main.iloc[i][a_cols].values
        
        # Plot Actuals
        plt.plot(days, actuals, label="Actual Sales", color="black", marker='o', linewidth=2, zorder=5)
        
        # Plot Mean (Analytical)
        plt.plot(days, df_main.iloc[i][f_cols].values, label="Analytical Mean", color="blue", linestyle='--', alpha=0.8)
        
        # Plot Sample Mean
        if "sample-mean" in dfs:
            plt.plot(days, dfs["sample-mean"].iloc[i][f_cols].values, label="Sample Mean", color="green", linestyle='-.', alpha=0.8)
            
        # Plot Uncertainty Band (P25 to P75)
        if "p25" in dfs and "p75" in dfs:
            p25 = dfs["p25"].iloc[i][f_cols].values
            p75 = dfs["p75"].iloc[i][f_cols].values
            plt.fill_between(days, p25, p75, color="gray", alpha=0.2, label="P25-P75 Uncertainty")
        
        # Title and Labels
        # Get metrics from the mean run for display
        mae = df_main.iloc[i]["metric_mae"]
        rmse = df_main.iloc[i]["metric_rmse"]
        
        plt.title(f"M5 Multi-Mode Forecast: {series_id}\n(Mean MAE: {mae:.3f} | RMSE: {rmse:.3f})")
        plt.xlabel("Horizon Day")
        plt.ylabel("Sales Units")
        plt.legend(loc="upper left")
        plt.grid(True, alpha=0.3)
        
        plot_path = output_dir / f"{series_id}_multi.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved: {plot_path}")

    # Print global summary
    metrics_path = artifact_dir / "holdout_metrics_all_modes.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            all_metrics = json.load(f)
        raw_wrmsse_values = {
            mode: m_dict["raw"]["wrmsse"]
            for mode, m_dict in all_metrics.items()
            if "raw" in m_dict and "wrmsse" in m_dict["raw"]
        }
        rounded_wrmsse_values = {
            mode: m_dict["rounded"]["wrmsse"]
            for mode, m_dict in all_metrics.items()
            if "rounded" in m_dict and "wrmsse" in m_dict["rounded"]
        }
        if raw_wrmsse_values:
            print("\n--- Global WRMSSE Comparison ---")
            for mode, value in raw_wrmsse_values.items():
                rounded_value = rounded_wrmsse_values.get(mode)
                if rounded_value is not None:
                    print(f"{mode:12}: raw={value:.5f} rounded={rounded_value:.5f}")
                else:
                    print(f"{mode:12}: {value:.5f}")
        else:
            print("\nWRMSSE was not computed for this run.")

if __name__ == "__main__":
    main()
