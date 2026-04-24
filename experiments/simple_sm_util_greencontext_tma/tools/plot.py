#!/usr/bin/env python3
"""
Plot global LDG baseline and TMA bulk-copy bandwidth vs SM count.

Usage:
    python3 tools/plot.py results.csv [output.png]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def validate_columns(df):
    required_cols = ["mode", "num_sms", "mean_bw_gb_s", "stdev_bw_gb_s"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Error: CSV missing columns: {missing}")
        print(f"Found columns: {df.columns.tolist()}")
        sys.exit(1)


def plot_bandwidth_sweep(csv_path, output_path=None):
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"Error reading CSV: {exc}")
        sys.exit(1)

    validate_columns(df)
    df = df.sort_values(["mode", "num_sms"])

    fig, ax = plt.subplots(figsize=(12, 7))
    styles = {
        "global": {
            "label": "Global LDG baseline",
            "color": "#1F77B4",
            "marker": "o",
        },
        "tma": {
            "label": "TMA bulk copy",
            "color": "#D95F02",
            "marker": "s",
        },
    }

    for mode, group in df.groupby("mode", sort=False):
        style = styles.get(mode, {"label": mode, "color": None, "marker": "o"})
        ax.errorbar(
            group["num_sms"],
            group["mean_bw_gb_s"],
            yerr=group["stdev_bw_gb_s"],
            marker=style["marker"],
            markersize=6,
            linestyle="-",
            linewidth=2,
            capsize=4,
            capthick=1.5,
            label=style["label"],
            color=style["color"],
        )

    ax.set_xlabel("Number of SMs", fontsize=13, fontweight="bold")
    ax.set_ylabel("Bandwidth (GiB/s)", fontsize=13, fontweight="bold")
    ax.set_title(
        "DRAM Read Path Bandwidth vs SM Count\nCUDA Green Contexts: Global LDG vs TMA Bulk Copy",
        fontsize=15,
        fontweight="bold",
        pad=18,
    )
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11, loc="best", framealpha=0.9)
    ax.tick_params(axis="both", which="major", labelsize=11)
    plt.tight_layout()

    if output_path is None:
        output_path = str(Path(csv_path).with_suffix(".png"))

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    for mode, group in df.groupby("mode", sort=False):
        peak_idx = group["mean_bw_gb_s"].idxmax()
        print(
            f"{mode}: peak {df.loc[peak_idx, 'mean_bw_gb_s']:.2f} +/- "
            f"{df.loc[peak_idx, 'stdev_bw_gb_s']:.2f} GiB/s at "
            f"{int(df.loc[peak_idx, 'num_sms'])} SMs"
        )
    print("=" * 60)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 tools/plot.py <results.csv> [output.png]")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    plot_bandwidth_sweep(csv_path, output_path)


if __name__ == "__main__":
    main()
