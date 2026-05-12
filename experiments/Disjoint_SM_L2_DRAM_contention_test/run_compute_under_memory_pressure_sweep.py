#!/usr/bin/env python3
"""
Sweep compute-under-continuous-memory-pressure runs by memory request pressure.

The memory working set is fixed and should be chosen large enough to reach DRAM.
The pressure level controls how many memory-workload blocks are launched per
memory-partition SM: --blocks_per_sm for streaming mode and --tma_blocks_per_sm
for TMA mode.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def parse_int_list(text):
    return [int(x) for x in text.replace(",", " ").split()]


def parse_str_list(text):
    return [x.strip() for x in text.replace(",", " ").split() if x.strip()]


def read_modes(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["mode"]: row for row in rows}


def write_dicts(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(exe, csv_path, args):
    cmd = [str(exe), *args, "--csv", str(csv_path)]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(proc.stdout)
        raise SystemExit(proc.returncode)

    for line in proc.stdout.splitlines():
        if (
            line.startswith("Actual ")
            or line.startswith("Repeat")
            or line.startswith("  [Compute only]")
            or line.startswith("  [Under pressure]")
            or line.startswith("  [Memory pressure]")
            or line.startswith("Compute isolated")
            or line.startswith("Compute under pressure")
            or line.startswith("Memory pressure launches")
            or line.startswith("Pressure wall")
            or line.startswith("Results written")
        ):
            print(line)


def base_args(ns, mem_mode, pressure_level, compute_size):
    args = [
        "--experiment", "compute_under_memory_pressure",
        "--mem_mode", mem_mode,
        "--mem_sms", str(ns.mem_sms),
        "--compute_sms", str(ns.compute_sms),
        "--mem_mib", str(ns.mem_mib),
        "--compute_size", str(compute_size),
        "--iterations", str(ns.iterations),
        "--repeats", str(ns.repeats),
        "--tpb", str(ns.tpb),
        "--blocks_per_sm", str(pressure_level if mem_mode == "streaming" else ns.blocks_per_sm),
        "--mma_repeats", str(ns.mma_repeats),
    ]
    if mem_mode == "tma":
        args.extend([
            "--tma_tile_bytes", str(ns.tma_tile_bytes),
            "--tma_blocks_per_sm", str(pressure_level),
        ])
    if ns.flush_l2:
        args.extend(["--flush_l2", "--l2_flush_mib", str(ns.l2_flush_mib)])
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mem_modes", default="streaming tma",
                        help="Memory modes to run: streaming, tma, or both")
    parser.add_argument("--mem_mib", type=int, default=1024,
                        help="Fixed memory working set in MiB; choose a value larger than L2")
    parser.add_argument("--pressure_levels", default="1 2 3 4 6 8 12 16 24 32",
                        help="Memory request pressure levels, as blocks per memory SM")
    parser.add_argument("--compute_sizes", default="512 768 1024 1280 1536",
                        help="WMMA square matrix sizes")
    parser.add_argument("--mem_sms", type=int, default=8)
    parser.add_argument("--compute_sms", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tpb", type=int, default=256)
    parser.add_argument("--blocks_per_sm", type=int, default=8,
                        help="Fallback blocks per SM for modes whose pressure is not swept")
    parser.add_argument("--mma_repeats", type=int, default=8)
    parser.add_argument("--tma_tile_bytes", type=int, default=32768)
    parser.add_argument("--tma_blocks_per_sm", type=int, default=2)
    parser.add_argument("--flush_l2", action="store_true")
    parser.add_argument("--l2_flush_mib", type=int, default=256)
    parser.add_argument("--output_dir", default="build/compute_under_memory_pressure_sweep")
    parser.add_argument("--executable", default=None)
    ns = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    exe = Path(ns.executable) if ns.executable else script_dir / "build" / "Disjoint_SM_L2_DRAM_contention_test"
    if not exe.exists():
        print(f"Executable not found: {exe}")
        print(f"Build first: cmake -S {script_dir} -B {script_dir / 'build'} && cmake --build {script_dir / 'build'} -j")
        return 1

    mem_modes = parse_str_list(ns.mem_modes)
    invalid_modes = sorted(set(mem_modes) - {"streaming", "tma"})
    if not mem_modes or invalid_modes:
        print("Use --mem_modes with streaming and/or tma.")
        if invalid_modes:
            print(f"Invalid modes: {invalid_modes}")
        return 1

    pressure_levels = parse_int_list(ns.pressure_levels)
    compute_sizes = parse_int_list(ns.compute_sizes)
    out_dir = Path(ns.output_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=============================================================")
    print("Compute under continuous memory pressure sweep")
    print("=============================================================")
    print(f"Memory modes:          {mem_modes}")
    print(f"Memory working set:    {ns.mem_mib} MiB")
    print(f"Pressure levels:       {pressure_levels} blocks per memory SM")
    print(f"Compute sizes:         {compute_sizes}")
    print(f"SMs memory/compute:    {ns.mem_sms} / {ns.compute_sms}")
    print(f"Iterations/repeats:    {ns.iterations} / {ns.repeats}")
    print(f"Output directory:      {out_dir}")
    print("=============================================================")

    summary_rows = []
    for mem_mode in mem_modes:
        for pressure_level in pressure_levels:
            for compute_size in compute_sizes:
                csv_path = raw_dir / f"pressure_{mem_mode}_level_{pressure_level}_compute_{compute_size}.csv"
                print(f"\nRunning mode={mem_mode} pressure_level={pressure_level} compute_size={compute_size}")
                run_benchmark(exe, csv_path, base_args(ns, mem_mode, pressure_level, compute_size))

                rows = read_modes(csv_path)
                iso = rows["isolated_compute"]
                pressure = rows["compute_under_memory_pressure"]
                summary_rows.append({
                    "mem_mode": pressure["mem_mode"],
                    "memory_pressure_level": pressure_level,
                    "memory_blocks_per_sm": pressure_level if mem_mode == "streaming" else ns.blocks_per_sm,
                    "tma_blocks_per_sm": pressure_level if mem_mode == "tma" else "",
                    "mem_mib": pressure["mem_working_set_mib"],
                    "compute_size": pressure["compute_size"],
                    "mem_sms": pressure["mem_sms"],
                    "compute_sms": pressure["compute_sms"],
                    "iterations": pressure["iterations"],
                    "repeats": pressure["repeats"],
                    "mma_repeats": pressure["mma_repeats"],
                    "flush_l2": pressure["flush_l2"],
                    "compute_time_isolated_ms": iso["compute_time_ms"],
                    "compute_time_concurrent_ms": pressure["compute_time_ms"],
                    "compute_isolated_tflops": iso["compute_tflops"],
                    "compute_concurrent_tflops": pressure["compute_tflops"],
                    "compute_slowdown": pressure["compute_slowdown"],
                    "compute_retention_pct": pressure["compute_retention_pct"],
                    "memory_launches_started": pressure["memory_launches_started"],
                    "memory_launches_completed": pressure["memory_launches_completed"],
                    "memory_launches_completed_before_compute_done": pressure["memory_launches_completed_before_compute_done"],
                    "compute_wall_time_ms": pressure["compute_wall_time_ms"],
                    "pressure_wall_time_ms": pressure["pressure_wall_time_ms"],
                    "overlap_wall_time_ms": pressure["overlap_wall_time_ms"],
                    "overlap_pct": pressure["overlap_pct"],
                })

    fieldnames = [
        "mem_mode", "memory_pressure_level", "memory_blocks_per_sm", "tma_blocks_per_sm",
        "mem_mib", "compute_size", "mem_sms", "compute_sms", "iterations", "repeats",
        "mma_repeats", "flush_l2", "compute_time_isolated_ms", "compute_time_concurrent_ms",
        "compute_isolated_tflops", "compute_concurrent_tflops", "compute_slowdown",
        "compute_retention_pct", "memory_launches_started", "memory_launches_completed",
        "memory_launches_completed_before_compute_done", "compute_wall_time_ms",
        "pressure_wall_time_ms", "overlap_wall_time_ms", "overlap_pct",
    ]
    summary_path = out_dir / "compute_under_memory_pressure_summary.csv"
    write_dicts(summary_path, summary_rows, fieldnames)
    print(f"\nSweep complete: {summary_path}")
    print(f"Plot with: python3 {script_dir / 'tools' / 'plot_compute_under_memory_pressure.py'} {summary_path} {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
