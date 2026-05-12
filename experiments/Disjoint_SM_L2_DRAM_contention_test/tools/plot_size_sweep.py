#!/usr/bin/env python3
"""
Plot working-set-size x compute-size sweep results.

Usage:
    python3 tools/plot_size_sweep.py size_sweep_results/size_sweep_summary.csv [output_dir]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def heatmap(ax, data, title, cmap, fmt=".2f"):
    im = ax.imshow(data.values, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Compute matrix size")
    ax.set_ylabel("Memory working set (MiB)")
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels([str(int(c)) for c in data.columns], rotation=35, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels([str(int(float(i))) for i in data.index])
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            ax.text(x, y, format(data.values[y, x], fmt), ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def pivot(df, value):
    return df.pivot(index="mem_mib", columns="compute_size", values=value).sort_index().sort_index(axis=1)


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
    df["mem_retention_pct"] = 100.0 * df["mem_bandwidth_concurrent_gib_s"] / df["mem_bandwidth_isolated_gib_s"]
    df["compute_retention_pct"] = 100.0 * df["compute_concurrent_tflops"] / df["compute_isolated_tflops"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    heatmap(axes[0, 0], pivot(df, "mem_slowdown"), "Streaming Memory Slowdown", "YlOrRd", ".2f")
    heatmap(axes[0, 1], pivot(df, "compute_slowdown"), "WMMA Compute Slowdown", "YlOrRd", ".2f")
    heatmap(axes[1, 0], pivot(df, "overlap_pct"), "Estimated Overlap (%)", "viridis", ".0f")
    heatmap(axes[1, 1], pivot(df, "mem_bandwidth_isolated_gib_s"), "Isolated Effective Memory GiB/s", "Blues", ".0f")
    fig.suptitle("Disjoint Green Context Size Sweep")
    fig.tight_layout()
    heatmap_path = output_dir / "size_sweep_heatmaps.png"
    fig.savefig(heatmap_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {heatmap_path}")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for mem_mib, group in df.sort_values("compute_size").groupby("mem_mib"):
        axes[0].plot(group["compute_size"], group["mem_slowdown"], marker="o", label=f"{int(mem_mib)} MiB")
        axes[1].plot(group["compute_size"], group["compute_slowdown"], marker="o", label=f"{int(mem_mib)} MiB")

    for ax, title in zip(axes, ["Memory Slowdown vs Compute Size", "Compute Slowdown vs Compute Size"]):
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Compute matrix size")
        ax.set_ylabel("Slowdown (x isolated time)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Working set", fontsize=8)

    fig.tight_layout()
    lines_path = output_dir / "size_sweep_slowdown_lines.png"
    fig.savefig(lines_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {lines_path}")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for compute_size, group in df.sort_values("mem_mib").groupby("compute_size"):
        axes[0].plot(group["mem_mib"], group["mem_retention_pct"], marker="o", label=str(compute_size))
        axes[1].plot(group["mem_mib"], group["compute_retention_pct"], marker="o", label=str(compute_size))

    for ax, title in zip(axes, ["Memory Retention vs Working Set", "Compute Retention vs Working Set"]):
        ax.axhline(100.0, color="black", linestyle="--", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Memory working set (MiB)")
        ax.set_ylabel("Retention (%)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(title="Compute size", fontsize=8)

    fig.tight_layout()
    retention_path = output_dir / "size_sweep_retention_lines.png"
    fig.savefig(retention_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {retention_path}")
    plt.close(fig)

    print("\nWorst slowdowns:")
    print(
        df.sort_values("compute_slowdown", ascending=False)
        [["mem_mib", "compute_size", "mem_slowdown", "compute_slowdown", "overlap_pct"]]
        .head(8)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
