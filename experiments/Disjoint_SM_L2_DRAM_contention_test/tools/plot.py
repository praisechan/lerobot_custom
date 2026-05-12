#!/usr/bin/env python3
"""
Plot isolated vs concurrent throughput and slowdown for the disjoint-SM benchmark.

Usage:
    python3 tools/plot.py results.csv [output_prefix]
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    csv_path = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else csv_path.replace(".csv", "")
    df = pd.read_csv(csv_path)

    required = {
        "mode",
        "mem_bandwidth_gib_s",
        "compute_tflops",
        "mem_slowdown",
        "compute_slowdown",
        "overlap_pct",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")

    mem_iso = df[df["mode"] == "isolated_memory"].iloc[0]
    compute_iso = df[df["mode"] == "isolated_compute"].iloc[0]
    concurrent = df[df["mode"] == "concurrent"].iloc[0]
    mem_mode = concurrent["mem_mode"] if "mem_mode" in df.columns else "streaming"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    axes[0].bar(
        ["isolated", "concurrent"],
        [mem_iso["mem_bandwidth_gib_s"], concurrent["mem_bandwidth_gib_s"]],
        color=["#4C78A8", "#F58518"],
    )
    axes[0].set_title(f"{mem_mode} memory")
    axes[0].set_ylabel("Effective GiB/s")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        ["isolated", "concurrent"],
        [compute_iso["compute_tflops"], concurrent["compute_tflops"]],
        color=["#54A24B", "#E45756"],
    )
    axes[1].set_title("WMMA Compute")
    axes[1].set_ylabel("TFLOP/s")
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(
        ["memory", "compute"],
        [concurrent["mem_slowdown"], concurrent["compute_slowdown"]],
        color=["#4C78A8", "#54A24B"],
    )
    axes[2].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[2].set_title(f"Slowdown, overlap {concurrent['overlap_pct']:.1f}%")
    axes[2].set_ylabel("x isolated time")
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle(f"Disjoint Green Context SM Partitions: Isolated vs Concurrent ({mem_mode} memory)")
    fig.tight_layout()
    out = f"{prefix}_summary.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved: {out}")

    print("\nSummary")
    print(f"  Memory retention:  {100.0 * concurrent['mem_bandwidth_gib_s'] / mem_iso['mem_bandwidth_gib_s']:.1f}%")
    print(f"  Compute retention: {100.0 * concurrent['compute_tflops'] / compute_iso['compute_tflops']:.1f}%")
    print(f"  Memory slowdown:   {concurrent['mem_slowdown']:.2f}x")
    print(f"  Compute slowdown:  {concurrent['compute_slowdown']:.2f}x")
    print(f"  Estimated overlap: {concurrent['overlap_pct']:.1f}%")


if __name__ == "__main__":
    main()
