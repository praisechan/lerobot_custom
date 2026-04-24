#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot row sweep summary for the TS-GEMM roofline experiment")
    parser.add_argument("--input_dir", default="row_sweep_results", help="Directory containing summary.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    summary_csv = input_dir / "summary.csv"
    if not summary_csv.exists():
        print(f"Missing {summary_csv}")
        return 1

    df = pd.read_csv(summary_csv).sort_values("row_size")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["row_size"], df["min_sms_compute_for_full_bw"], "o-", linewidth=2, color="#A23B72")
    ax.set_xlabel("TS-GEMM Row Size M")
    ax.set_ylabel("Min SMs for Compute >= Peak-BW Demand")
    ax.set_title("SMs Needed to Digest a Peak-Bandwidth TS-GEMM Stream")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(input_dir / "min_sms_for_full_bw.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["row_size"], df["required_full_bw_GFLOPS"], "o-", linewidth=2, label="Required at Peak BW", color="#2E86AB")
    ax.plot(df["row_size"], df["max_real_GFLOPS"], "s-", linewidth=2, label="Max Real TS-GEMM", color="#2A9D8F")
    ax.plot(df["row_size"], df["max_predicted_GFLOPS"], "^-", linewidth=2, label="Max Predicted", color="#F18F01")
    ax.set_xlabel("TS-GEMM Row Size M")
    ax.set_ylabel("Throughput (GFLOPS)")
    ax.set_title("Peak-Bandwidth Demand vs Achievable TS-GEMM Throughput")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(input_dir / "row_sweep_throughput_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Generated plots in {input_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
