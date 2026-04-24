#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot summary for the synthetic wide-AI sweep")
    parser.add_argument("--input_dir", default="wide_ai_sweep_results", help="Directory containing summary.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    summary_csv = input_dir / "summary.csv"
    if not summary_csv.exists():
        print(f"Missing {summary_csv}")
        return 1

    df = pd.read_csv(summary_csv).sort_values("arithmetic_intensity")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["arithmetic_intensity"], df["min_sms_compute_for_full_bw"], "o-", linewidth=2, color="#A23B72")
    ax.set_xscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Min SMs for Compute >= Peak-BW Demand")
    ax.set_title("Synthetic Wide-AI Sweep: SMs Needed to Digest a Peak-BW Stream")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(input_dir / "wide_ai_min_sms_for_full_bw.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["arithmetic_intensity"], df["required_full_bw_GFLOPS"], "o-", linewidth=2,
            label="Required at Peak BW", color="#2E86AB")
    ax.plot(df["arithmetic_intensity"], df["max_real_GFLOPS"], "s-", linewidth=2,
            label="Max Real TS-GEMM", color="#2A9D8F")
    ax.plot(df["arithmetic_intensity"], df["max_predicted_GFLOPS"], "^-", linewidth=2,
            label="Max Predicted", color="#F18F01")
    ax.set_xscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Throughput (GFLOPS)")
    ax.set_title("Synthetic Wide-AI Sweep: Throughput Summary")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(input_dir / "wide_ai_throughput_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Generated plots in {input_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
