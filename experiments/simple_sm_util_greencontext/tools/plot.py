#!/usr/bin/env python3

import argparse
import csv


def load_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Plot read bandwidth vs SM count")
    ap.add_argument("--csv", required=True, help="Input CSV path")
    ap.add_argument("--out", required=True, help="Output image path")
    args = ap.parse_args()

    rows = load_csv(args.csv)
    if not rows:
        raise SystemExit("No rows found in CSV")

    sm = [int(r["sm_count"]) for r in rows]
    bw = [float(r["read_GBps_mean"]) for r in rows]

    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))
    plt.plot(sm, bw, marker="o", linewidth=1.5)
    plt.xlabel("SM count")
    plt.ylabel("Read bandwidth (GB/s)")
    plt.title("Read Bandwidth vs SM Count")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
