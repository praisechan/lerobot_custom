#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_runs(input_dir: Path, pattern: str) -> pd.DataFrame:
    csv_paths = sorted(input_dir.glob(pattern))
    if not csv_paths:
        raise FileNotFoundError(f"No files matching {pattern} found in {input_dir}")

    frames = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        df["source_csv"] = csv_path.name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def plot_roofline_by_sm(df: pd.DataFrame, output_path: Path, show_real: bool) -> None:
    df = df.sort_values(["num_sms", "arithmetic_intensity_flops_per_byte"])
    sm_values = sorted(df["num_sms"].unique())
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(11, 7))

    for idx, sm in enumerate(sm_values):
        sm_df = df[df["num_sms"] == sm].copy()
        color = cmap(idx % 10)

        ax.plot(
            sm_df["arithmetic_intensity_flops_per_byte"],
            sm_df["predicted_GFLOPS"],
            "o-",
            linewidth=2,
            markersize=6,
            color=color,
            label=f"{sm} SM Predicted Roofline",
        )

        if show_real:
            ax.plot(
                sm_df["arithmetic_intensity_flops_per_byte"],
                sm_df["real_GFLOPS"],
                "x--",
                linewidth=1.5,
                markersize=7,
                color=color,
                alpha=0.8,
                label=f"{sm} SM Real",
            )

    ai_min = df["arithmetic_intensity_flops_per_byte"].min()
    ai_max = df["arithmetic_intensity_flops_per_byte"].max()
    ai_span = ai_max - ai_min
    pad = max(ai_span * 0.2, ai_min * 0.02 if ai_min > 0 else 0.1)
    ax.set_xlim(max(ai_min - pad, 0.0), ai_max + pad)

    y_max = max(df["predicted_GFLOPS"].max(), df["real_GFLOPS"].max())
    ax.set_ylim(0.0, y_max * 1.1)

    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Throughput (GFLOPS)")
    ax.set_title("TS-GEMM Roofline Across Active SM Counts")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2)

    sweep_label = "N"
    sweep_values = sorted(df["N"].unique())
    if len(sweep_values) <= 1:
        sweep_label = "M"
        sweep_values = sorted(df["row_size"].unique())

    text = (
        f"AI range: {ai_min:.3f} - {ai_max:.3f} FLOP/byte\n"
        f"{sweep_label} sweep: {', '.join(str(int(v)) for v in sweep_values)}\n"
        f"M={int(df['row_size'].iloc[0])}, K={int(df['K'].iloc[0])}, N={int(df['N'].iloc[0]) if sweep_label == 'M' else 'var'}"
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot roofline throughput vs arithmetic intensity, with one line per SM count"
    )
    parser.add_argument(
        "--input_dir",
        default="row_sweep_results",
        help="Directory containing per-shape CSV files from a sweep script",
    )
    parser.add_argument(
        "--glob",
        default="row_*.csv",
        help="Glob pattern for per-shape CSV files (default: row_*.csv)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path (default: <input_dir>/roofline_by_sm.png)",
    )
    parser.add_argument(
        "--no-real",
        action="store_true",
        help="Hide real TS-GEMM points and show only predicted roofline lines",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.out) if args.out else input_dir / "roofline_by_sm.png"

    df = load_runs(input_dir, args.glob)
    plot_roofline_by_sm(df, output_path, show_real=not args.no_real)
    print(f"Saved plot: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
