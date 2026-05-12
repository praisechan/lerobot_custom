#!/usr/bin/env python3
"""
Plot duration-matched contention sweep results.

Usage:
    python3 tools/plot_duration_matched.py build/duration_matched_sweep/duration_matched_summary.csv [output_dir]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path).sort_values("matched_duration_ms")
    if "mem_mode" not in df.columns:
        df["mem_mode"] = "streaming"
    df["mem_retention_pct"] = 100.0 * df["mem_bandwidth_concurrent_gib_s"] / df["mem_bandwidth_isolated_gib_s"]
    df["compute_retention_pct"] = 100.0 * df["compute_concurrent_tflops"] / df["compute_isolated_tflops"]
    labels = [f"{r.mem_mode}\n{int(r.mem_mib)}M/{int(r.compute_size)}" for r in df.itertuples()]
    modes = list(df["mem_mode"].drop_duplicates())
    colors = {"streaming": "#4C78A8", "tma": "#F58518"}
    markers = {"streaming": "o", "tma": "s"}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes[0, 0]
    for mode in modes:
        sub = df[df["mem_mode"] == mode]
        ax.plot(
            sub["matched_duration_ms"],
            sub["mem_slowdown"],
            marker=markers.get(mode, "o"),
            color=colors.get(mode),
            label=f"{mode} memory",
        )
        ax.plot(
            sub["matched_duration_ms"],
            sub["compute_slowdown"],
            marker=markers.get(mode, "o"),
            color=colors.get(mode),
            linestyle="--",
            label=f"compute with {mode}",
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Matched isolated duration, geometric mean (ms)")
    ax.set_ylabel("Slowdown (x isolated time)")
    ax.set_title("Slowdown vs Matched Duration")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    ax = axes[0, 1]
    for mode in modes:
        sub = df[df["mem_mode"] == mode]
        ax.plot(
            sub["matched_duration_ms"],
            sub["mem_retention_pct"],
            marker=markers.get(mode, "o"),
            color=colors.get(mode),
            label=f"{mode} memory bandwidth",
        )
        ax.plot(
            sub["matched_duration_ms"],
            sub["compute_retention_pct"],
            marker=markers.get(mode, "o"),
            color=colors.get(mode),
            linestyle="--",
            label=f"compute with {mode}",
        )
    ax.axhline(100.0, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Matched isolated duration, geometric mean (ms)")
    ax.set_ylabel("Retention (%)")
    ax.set_title("Throughput Retention")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    ax = axes[1, 0]
    for mode in modes:
        sub = df[df["mem_mode"] == mode]
        ax.scatter(
            sub["mem_time_isolated_ms"],
            sub["compute_time_isolated_ms"],
            s=70,
            marker=markers.get(mode, "o"),
            color=colors.get(mode),
            label=mode,
        )
    max_time = max(df["mem_time_isolated_ms"].max(), df["compute_time_isolated_ms"].max()) * 1.15
    min_time = min(df["mem_time_isolated_ms"].min(), df["compute_time_isolated_ms"].min()) * 0.8
    ax.plot([min_time, max_time], [min_time, max_time], color="black", linestyle="--", linewidth=1)
    for label, row in zip(labels, df.itertuples()):
        ax.annotate(label, (row.mem_time_isolated_ms, row.compute_time_isolated_ms), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Memory isolated time (ms)")
    ax.set_ylabel("Compute isolated time (ms)")
    ax.set_title("Matched Pair Quality")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    ax = axes[1, 1]
    ax.bar(labels, df["overlap_pct"], color=[colors.get(mode, "#4C78A8") for mode in df["mem_mode"]])
    ax.set_ylabel("Estimated overlap (%)")
    ax.set_title("Concurrent Overlap by Pair")
    ax.tick_params(axis="x", labelrotation=35)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Duration-Matched Disjoint Green Context Contention Sweep by Memory Mode")
    fig.tight_layout()
    summary_path = output_dir / "duration_matched_summary.png"
    fig.savefig(summary_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {summary_path}")
    plt.close(fig)

    fig_height = max(6.0, 0.34 * len(df) + 1.6)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    y = range(len(df))
    height = 0.36
    short_labels = [
        f"{'TMA' if r.mem_mode == 'tma' else 'STR'}  {int(r.mem_mib)}M / C{int(r.compute_size)}"
        for r in df.itertuples()
    ]
    ax.barh(
        [i - height / 2 for i in y],
        df["mem_slowdown"],
        height=height,
        color=[colors.get(mode, "#4C78A8") for mode in df["mem_mode"]],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.barh(
        [i + height / 2 for i in y],
        df["compute_slowdown"],
        height=height,
        color="#54A24B",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels(short_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Slowdown (x isolated time)")
    ax.set_title("Duration-Matched Pair Slowdowns")
    ax.grid(axis="x", alpha=0.3)
    legend_handles = [
        Patch(facecolor=colors["streaming"], label="Streaming memory slowdown"),
        Patch(facecolor=colors["tma"], label="TMA memory slowdown"),
        Patch(facecolor="#54A24B", label="Compute slowdown"),
    ]
    ax.legend(handles=legend_handles, loc="lower right")
    fig.tight_layout()
    bar_path = output_dir / "duration_matched_slowdown_bars.png"
    fig.savefig(bar_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {bar_path}")
    plt.close(fig)

    print("\nDuration-matched results:")
    print(
        df[[
            "mem_mode", "mem_mib", "compute_size", "mem_time_isolated_ms", "compute_time_isolated_ms",
            "duration_ratio", "mem_slowdown", "compute_slowdown", "overlap_pct",
        ]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
