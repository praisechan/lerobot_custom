#!/usr/bin/env python3
"""
Calibrate isolated kernel durations, select memory/compute pairs with similar
isolated runtimes, then benchmark those matched pairs.
"""

import argparse
import csv
import math
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


def run_benchmark(exe, csv_path, args, quiet=False):
    cmd = [str(exe), *args, "--csv", str(csv_path)]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(proc.stdout)
        raise SystemExit(proc.returncode)

    if not quiet:
        for line in proc.stdout.splitlines():
            if (
                line.startswith("Actual ")
                or line.startswith("Repeat")
                or line.startswith("  [")
                or line.startswith("Memory isolated")
                or line.startswith("Compute isolated")
                or line.startswith("Memory concurrent")
                or line.startswith("Compute concurrent")
                or line.startswith("Concurrent wall")
                or line.startswith("Results written")
            ):
                print(line)


def base_args(ns, mem_mode, mem_mib, compute_size, repeats, iterations):
    args = [
        "--mem_mode", mem_mode,
        "--mem_sms", str(ns.mem_sms),
        "--compute_sms", str(ns.compute_sms),
        "--mem_mib", str(mem_mib),
        "--compute_size", str(compute_size),
        "--iterations", str(iterations),
        "--repeats", str(repeats),
        "--mma_repeats", str(ns.mma_repeats),
    ]
    if mem_mode == "tma":
        args.extend([
            "--tma_tile_bytes", str(ns.tma_tile_bytes),
            "--tma_blocks_per_sm", str(ns.tma_blocks_per_sm),
        ])
    if ns.flush_l2:
        args.append("--flush_l2")
    return args


def write_dicts(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mem_modes", default="streaming tma",
                        help="Memory modes to run: streaming, tma, or both")
    parser.add_argument("--mem_sizes", default="16 32 48 64 96 128 192 256 384 512 768 1024",
                        help="Candidate memory working sets in MiB")
    parser.add_argument("--compute_sizes", default="256 384 512 640 768 896 1024 1152 1280 1408 1536",
                        help="Candidate WMMA square matrix sizes")
    parser.add_argument("--mem_sms", type=int, default=8)
    parser.add_argument("--compute_sms", type=int, default=8)
    parser.add_argument("--calibration_iterations", type=int, default=3)
    parser.add_argument("--calibration_repeats", type=int, default=1)
    parser.add_argument("--matched_iterations", type=int, default=5)
    parser.add_argument("--matched_repeats", type=int, default=3)
    parser.add_argument("--mma_repeats", type=int, default=8)
    parser.add_argument("--tma_tile_bytes", type=int, default=32768)
    parser.add_argument("--tma_blocks_per_sm", type=int, default=2)
    parser.add_argument("--anchor_mem_mib", type=int, default=128,
                        help="Memory working set used while calibrating compute sizes")
    parser.add_argument("--anchor_compute_size", type=int, default=512,
                        help="Compute size used while calibrating memory sizes")
    parser.add_argument("--max_ratio", type=float, default=1.35,
                        help="Keep pairs whose slower isolated duration is at most this ratio of the faster")
    parser.add_argument("--output_dir", default="build/duration_matched_sweep")
    parser.add_argument("--flush_l2", action="store_true")
    parser.add_argument("--executable", default=None)
    parser.add_argument("--reuse_calibration", action="store_true")
    ns = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    exe = Path(ns.executable) if ns.executable else script_dir / "build" / "Disjoint_SM_L2_DRAM_contention_test"
    if not exe.exists():
        print(f"Executable not found: {exe}")
        print(f"Build first: cmake -S {script_dir} -B {script_dir / 'build'} && cmake --build {script_dir / 'build'} -j")
        return 1

    mem_sizes = parse_int_list(ns.mem_sizes)
    compute_sizes = parse_int_list(ns.compute_sizes)
    mem_modes = parse_str_list(ns.mem_modes)
    if not mem_modes:
        print("No memory modes requested.")
        return 1
    invalid_modes = sorted(set(mem_modes) - {"streaming", "tma"})
    if invalid_modes:
        print(f"Invalid --mem_modes value(s): {invalid_modes}. Expected streaming and/or tma.")
        return 1
    out_dir = Path(ns.output_dir)
    cal_dir = out_dir / "calibration_raw"
    matched_dir = out_dir / "matched_raw"
    cal_dir.mkdir(parents=True, exist_ok=True)
    matched_dir.mkdir(parents=True, exist_ok=True)

    print("=============================================================")
    print("Duration-matched disjoint SM sweep")
    print("=============================================================")
    print(f"Memory modes:          {mem_modes}")
    print(f"Memory candidates MiB: {mem_sizes}")
    print(f"Compute candidates:    {compute_sizes}")
    print(f"Max duration ratio:    {ns.max_ratio:.2f}x")
    print(f"Output directory:      {out_dir}")
    print("=============================================================")

    mem_cal = []
    print("\nCalibrating memory durations")
    for mem_mode in mem_modes:
        for mem_mib in mem_sizes:
            csv_path = cal_dir / f"{mem_mode}_mem_{mem_mib}_compute_{ns.anchor_compute_size}.csv"
            if not (ns.reuse_calibration and csv_path.exists()):
                print(f"  mode={mem_mode}, mem_mib={mem_mib} with anchor compute_size={ns.anchor_compute_size}")
                run_benchmark(
                    exe,
                    csv_path,
                    base_args(ns, mem_mode, mem_mib, ns.anchor_compute_size, ns.calibration_repeats, ns.calibration_iterations),
                    quiet=True,
                )
            rows = read_modes(csv_path)
            mem = rows["isolated_memory"]
            mem_cal.append({
                "mem_mode": mem.get("mem_mode", mem_mode),
                "mem_mib": float(mem["mem_working_set_mib"]),
                "mem_time_ms": float(mem["mem_time_ms"]),
                "mem_bandwidth_gib_s": float(mem["mem_bandwidth_gib_s"]),
            })

    compute_cal = []
    print("\nCalibrating compute durations")
    compute_cal_mode = mem_modes[0]
    for compute_size in compute_sizes:
        csv_path = cal_dir / f"compute_{compute_cal_mode}_mem_{ns.anchor_mem_mib}_compute_{compute_size}.csv"
        if not (ns.reuse_calibration and csv_path.exists()):
            print(f"  compute_size={compute_size} with anchor mode={compute_cal_mode}, mem_mib={ns.anchor_mem_mib}")
            run_benchmark(
                exe,
                csv_path,
                base_args(ns, compute_cal_mode, ns.anchor_mem_mib, compute_size, ns.calibration_repeats, ns.calibration_iterations),
                quiet=True,
            )
        rows = read_modes(csv_path)
        compute = rows["isolated_compute"]
        compute_cal.append({
            "compute_size": int(compute["compute_size"]),
            "compute_time_ms": float(compute["compute_time_ms"]),
            "compute_tflops": float(compute["compute_tflops"]),
        })

    write_dicts(out_dir / "memory_calibration.csv", mem_cal, ["mem_mode", "mem_mib", "mem_time_ms", "mem_bandwidth_gib_s"])
    write_dicts(out_dir / "compute_calibration.csv", compute_cal, ["compute_size", "compute_time_ms", "compute_tflops"])

    candidate_pairs = []
    used = set()
    for mem in mem_cal:
        best = min(
            compute_cal,
            key=lambda c: max(mem["mem_time_ms"], c["compute_time_ms"]) / min(mem["mem_time_ms"], c["compute_time_ms"]),
        )
        ratio = max(mem["mem_time_ms"], best["compute_time_ms"]) / min(mem["mem_time_ms"], best["compute_time_ms"])
        if ratio <= ns.max_ratio:
            key = (mem["mem_mode"], mem["mem_mib"], best["compute_size"])
            if key not in used:
                used.add(key)
                candidate_pairs.append({
                    **mem,
                    **best,
                    "duration_ratio": ratio,
                    "matched_duration_ms": math.sqrt(mem["mem_time_ms"] * best["compute_time_ms"]),
                })

    if not candidate_pairs:
        print("No duration-matched pairs found. Increase --max_ratio or expand candidate sizes.")
        return 1

    candidate_pairs.sort(key=lambda r: r["matched_duration_ms"])
    write_dicts(
        out_dir / "selected_pairs.csv",
        candidate_pairs,
        ["mem_mib", "compute_size", "mem_time_ms", "compute_time_ms", "duration_ratio",
         "matched_duration_ms", "mem_bandwidth_gib_s", "compute_tflops", "mem_mode"],
    )

    print("\nSelected duration-matched pairs")
    for pair in candidate_pairs:
        print(
            f"  mode={pair['mem_mode']}, mem={pair['mem_mib']:.0f} MiB ({pair['mem_time_ms']:.3f} ms) "
            f"<-> compute={pair['compute_size']} ({pair['compute_time_ms']:.3f} ms), "
            f"ratio={pair['duration_ratio']:.2f}x"
        )

    summary_rows = []
    print("\nRunning matched-pair contention measurements")
    for pair in candidate_pairs:
        mem_mode = pair["mem_mode"]
        mem_mib = int(round(pair["mem_mib"]))
        compute_size = int(pair["compute_size"])
        csv_path = matched_dir / f"matched_{mem_mode}_mem_{mem_mib}_compute_{compute_size}.csv"
        print(f"\nMatched run: mode={mem_mode}, mem_mib={mem_mib}, compute_size={compute_size}")
        run_benchmark(
            exe,
            csv_path,
            base_args(ns, mem_mode, mem_mib, compute_size, ns.matched_repeats, ns.matched_iterations),
            quiet=False,
        )

        rows = read_modes(csv_path)
        mem = rows["isolated_memory"]
        compute = rows["isolated_compute"]
        conc = rows["concurrent"]
        mem_time = float(mem["mem_time_ms"])
        compute_time = float(compute["compute_time_ms"])
        ratio = max(mem_time, compute_time) / min(mem_time, compute_time)
        summary_rows.append({
            "mem_mode": conc.get("mem_mode", mem_mode),
            "mem_mib": conc["mem_working_set_mib"],
            "compute_size": conc["compute_size"],
            "mem_sms": conc["mem_sms"],
            "compute_sms": conc["compute_sms"],
            "iterations": conc["iterations"],
            "repeats": conc["repeats"],
            "mma_repeats": conc["mma_repeats"],
            "flush_l2": conc["flush_l2"],
            "mem_time_isolated_ms": mem["mem_time_ms"],
            "compute_time_isolated_ms": compute["compute_time_ms"],
            "duration_ratio": f"{ratio:.6f}",
            "matched_duration_ms": f"{math.sqrt(mem_time * compute_time):.6f}",
            "mem_time_concurrent_ms": conc["mem_time_ms"],
            "compute_time_concurrent_ms": conc["compute_time_ms"],
            "wall_time_ms": conc["wall_time_ms"],
            "mem_bandwidth_isolated_gib_s": mem["mem_bandwidth_gib_s"],
            "mem_bandwidth_concurrent_gib_s": conc["mem_bandwidth_gib_s"],
            "compute_isolated_tflops": compute["compute_tflops"],
            "compute_concurrent_tflops": conc["compute_tflops"],
            "mem_slowdown": conc["mem_slowdown"],
            "compute_slowdown": conc["compute_slowdown"],
            "overlap_pct": conc["overlap_pct"],
        })

    fieldnames = [
        "mem_mode", "mem_mib", "compute_size", "mem_sms", "compute_sms", "iterations", "repeats", "mma_repeats", "flush_l2",
        "mem_time_isolated_ms", "compute_time_isolated_ms", "duration_ratio", "matched_duration_ms",
        "mem_time_concurrent_ms", "compute_time_concurrent_ms", "wall_time_ms",
        "mem_bandwidth_isolated_gib_s", "mem_bandwidth_concurrent_gib_s",
        "compute_isolated_tflops", "compute_concurrent_tflops",
        "mem_slowdown", "compute_slowdown", "overlap_pct",
    ]
    summary_path = out_dir / "duration_matched_summary.csv"
    write_dicts(summary_path, summary_rows, fieldnames)
    print(f"\nMatched sweep complete: {summary_path}")
    print(f"Plot with: python3 {script_dir / 'tools' / 'plot_duration_matched.py'} {summary_path} {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
