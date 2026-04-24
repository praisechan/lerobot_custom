#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_single_run(csv_path: Path, output_prefix: Path) -> None:
    df = pd.read_csv(csv_path).sort_values("num_sms")
    ai = df["arithmetic_intensity_flops_per_byte"].iloc[0]
    row_size = int(df["row_size"].iloc[0])
    k_dim = int(df["K"].iloc[0])
    n_dim = int(df["N"].iloc[0])

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(df["num_sms"], df["compute_roof_GFLOPS"], "o-", label="Compute Roof", color="#A23B72", linewidth=2)
    ax.plot(df["num_sms"], df["memory_roof_GFLOPS"], "s-", label="AI x BW Roof", color="#2E86AB", linewidth=2)
    ax.plot(df["num_sms"], df["predicted_GFLOPS"], "^-", label="Predicted Min(Roofs)", color="#F18F01", linewidth=2)
    ax.plot(df["num_sms"], df["real_GFLOPS"], "d-", label="Real TS-GEMM", color="#2A9D8F", linewidth=2)
    ax.axhline(df["required_full_bw_GFLOPS"].iloc[0], linestyle="--", color="gray", alpha=0.7,
               label="Compute Needed for Peak BW")
    ax.set_xlabel("Active SMs")
    ax.set_ylabel("Throughput (GFLOPS)")
    ax.set_title(f"TS-GEMM Roofline vs Active SMs (M={row_size}, K={k_dim}, N={n_dim}, AI={ai:.2f})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(output_prefix.name + "_throughput.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(df["num_sms"], df["bandwidth_roof_GBps"], "o-", label="Bandwidth Roof", color="#2E86AB", linewidth=2)
    ax.plot(df["num_sms"], df["implied_real_GBps"], "s-", label="Implied Real BW", color="#2A9D8F", linewidth=2)
    ax.set_xlabel("Active SMs")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title(f"TS-GEMM Bandwidth View (M={row_size}, K={k_dim}, N={n_dim})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(output_prefix.name + "_bandwidth.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    efficiency = 100.0 * df["real_GFLOPS"] / df["predicted_GFLOPS"]
    ax.plot(df["num_sms"], efficiency, "o-", color="#264653", linewidth=2)
    ax.set_xlabel("Active SMs")
    ax.set_ylabel("Real / Predicted (%)")
    ax.set_title(f"TS-GEMM Roofline Tracking Efficiency (M={row_size}, K={k_dim}, N={n_dim})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(output_prefix.name + "_efficiency.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot a single TS-GEMM roofline run")
    parser.add_argument("--csv", required=True, help="Input CSV from TSGEMM_roofline_orin_libsmctrl")
    parser.add_argument("--out-prefix", default=None, help="Output prefix path")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_prefix = Path(args.out_prefix) if args.out_prefix else csv_path.with_suffix("")
    plot_single_run(csv_path, out_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
