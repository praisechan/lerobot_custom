#!/usr/bin/env python3
"""
Plot compute-under-continuous-memory-pressure sweep results.

Usage:
    python3 tools/plot_compute_under_memory_pressure.py build/compute_under_memory_pressure_sweep/compute_under_memory_pressure_summary.csv [output_dir]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd


def pivot(df, value):
    return df.pivot(index="memory_pressure_level", columns="compute_size", values=value).sort_index().sort_index(axis=1)


def heatmap(ax, data, title, cmap, fmt=".2f"):
    im = ax.imshow(data.values, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Compute matrix size")
    ax.set_ylabel("Memory pressure level\n(blocks per memory SM)")
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels([str(int(c)) for c in data.columns], rotation=35, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels([str(int(float(i))) for i in data.index])
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            ax.text(x, y, format(data.values[y, x], fmt), ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df["mem_mib"] = df["mem_mib"].astype(float)
    df["compute_size"] = df["compute_size"].astype(int)
    if "memory_pressure_level" not in df.columns:
        df["memory_pressure_level"] = df["mem_mib"]
    df["memory_pressure_level"] = df["memory_pressure_level"].astype(int)
    if "compute_retention_pct" not in df.columns:
        df["compute_retention_pct"] = 100.0 * df["compute_concurrent_tflops"] / df["compute_isolated_tflops"]
    df["pressure_to_compute_wall_pct"] = 100.0 * df["pressure_wall_time_ms"] / df["compute_wall_time_ms"]

    modes = list(df["mem_mode"].drop_duplicates())
    fig, axes = plt.subplots(4, len(modes), figsize=(7.2 * len(modes), 17), squeeze=False)

    for col, mode in enumerate(modes):
        sub = df[df["mem_mode"] == mode]
        heatmap(axes[0, col], pivot(sub, "compute_slowdown"), f"{mode}: Compute Slowdown", "YlOrRd", ".2f")
        heatmap(axes[1, col], pivot(sub, "compute_retention_pct"), f"{mode}: Compute Retention (%)", "viridis", ".0f")
        heatmap(axes[2, col], pivot(sub, "memory_launches_started"), f"{mode}: Memory Launches Started", "Blues", ".0f")
        heatmap(axes[3, col], pivot(sub, "pressure_to_compute_wall_pct"), f"{mode}: Pressure Wall / Compute Wall (%)", "PuBuGn", ".0f")

    fig.suptitle("Compute Under Continuous Memory Pressure")
    fig.tight_layout()
    heatmap_path = output_dir / "compute_under_memory_pressure_heatmaps.png"
    fig.savefig(heatmap_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {heatmap_path}")
    plt.close(fig)

    colors = {"streaming": "#4C78A8", "tma": "#F58518"}
    mode_linestyles = {"streaming": "-", "tma": "--"}
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    compute_sizes = sorted(df["compute_size"].unique())
    compute_markers = {
        compute_size: marker_cycle[i % len(marker_cycle)]
        for i, compute_size in enumerate(compute_sizes)
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for mode in modes:
        for compute_size, group in df[df["mem_mode"] == mode].sort_values("memory_pressure_level").groupby("compute_size"):
            label = f"{mode} C{compute_size}"
            axes[0, 0].plot(
                group["memory_pressure_level"], group["compute_slowdown"],
                color=colors.get(mode), marker=compute_markers[compute_size],
                linestyle=mode_linestyles.get(mode, "-"), label=label,
            )
            axes[0, 1].plot(
                group["memory_pressure_level"], group["compute_retention_pct"],
                color=colors.get(mode), marker=compute_markers[compute_size],
                linestyle=mode_linestyles.get(mode, "-"), label=label,
            )
            axes[1, 0].plot(
                group["memory_pressure_level"], group["memory_launches_started"],
                color=colors.get(mode), marker=compute_markers[compute_size],
                linestyle=mode_linestyles.get(mode, "-"), label=label,
            )
            axes[1, 1].plot(
                group["memory_pressure_level"], group["compute_time_isolated_ms"],
                color=colors.get(mode), marker=compute_markers[compute_size],
                linestyle=":", alpha=0.7,
            )
            axes[1, 1].plot(
                group["memory_pressure_level"], group["compute_time_concurrent_ms"],
                color=colors.get(mode), marker=compute_markers[compute_size],
                linestyle=mode_linestyles.get(mode, "-"), label=label,
            )

    axes[0, 0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Compute Slowdown")
    axes[0, 0].set_ylabel("x isolated compute time")

    axes[0, 1].axhline(100.0, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Compute Throughput Retention")
    axes[0, 1].set_ylabel("Retention (%)")

    axes[1, 0].set_title("Memory Launches During Compute Window")
    axes[1, 0].set_ylabel("Launches started")

    axes[1, 1].set_title("Compute Time")
    axes[1, 1].set_ylabel("ms/iteration")

    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Memory pressure level (blocks per memory SM)")
        ax.grid(True, alpha=0.3, which="both")

    axes[0, 0].legend(fontsize=8, ncols=2)
    legend_handles = [
        Patch(facecolor=colors.get("streaming", "#4C78A8"), label="Streaming color"),
        Patch(facecolor=colors.get("tma", "#F58518"), label="TMA color"),
    ]
    for compute_size in compute_sizes:
        legend_handles.append(
            Line2D(
                [0], [0],
                color="black",
                marker=compute_markers[compute_size],
                linestyle="None",
                label=f"C{compute_size} marker",
            )
        )
    legend_handles.append(
        Line2D([0], [0], color="black", linestyle=":", label="Dotted: isolated compute time")
    )
    axes[1, 1].legend(handles=legend_handles, fontsize=8)

    fig.suptitle("Streaming vs TMA Memory Pressure Effects on WMMA Compute")
    fig.tight_layout()
    lines_path = output_dir / "compute_under_memory_pressure_lines.png"
    fig.savefig(lines_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {lines_path}")
    plt.close(fig)

    print("\nWorst compute slowdowns:")
    print(
        df.sort_values("compute_slowdown", ascending=False)
        [[
            "mem_mode", "memory_pressure_level", "mem_mib", "compute_size", "compute_slowdown",
            "compute_retention_pct", "memory_launches_started", "overlap_pct",
        ]]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
