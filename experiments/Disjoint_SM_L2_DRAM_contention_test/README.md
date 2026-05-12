# Disjoint SM L2/DRAM Contention Test

This experiment measures whether a memory-bound kernel and a compute-bound kernel still contend when they run in separate CUDA Green Context SM partitions.

It is intentionally separate from `../GEMV_GEMM_contention_test`. The older experiment puts GEMV and GEMM streams in the same Green Context partition. This experiment creates two disjoint partition descriptors from one `cuDevSmResourceSplitByCount()` call, then creates one Green Context for the streaming memory workload and another for the WMMA compute workload.

## Question

If the kernels do not share SMs, do they still interfere through shared L2 cache and DRAM?

The CSV reports isolated throughput for each kernel, concurrent throughput while both partitions are active, slowdown vs isolated execution, and an estimated overlap percentage. Slowdown with good overlap is evidence of non-SM contention, usually L2 bandwidth/capacity, DRAM bandwidth, memory fabric, or scheduler/connection effects.

## Workloads

### Memory-Bound Kernel Modes

Select the memory workload with:

```bash
--mem_mode streaming   # default
--mem_mode tma
```

The existing `streaming` mode is unchanged. It is a vectorized streaming triad over `float4` arrays:

```text
out[i] = a[i] + alpha * b[i]
```

Each element performs two coalesced 16-byte loads and one coalesced 16-byte store. Effective bandwidth is reported as:

```text
3 * num_float4_elements * sizeof(float4) * iterations / elapsed_time
```

Use `--mem_mib` to set the total working set across `a`, `b`, and `out`. To exercise DRAM rather than only cache, choose a working set larger than L2. The program prints the runtime-reported L2 size so small vs large working-set comparisons are easy:

```bash
# Cache-friendly-ish small case
./Disjoint_SM_L2_DRAM_contention_test --mem_mib 16 --compute_size 2048 --iterations 3 --repeats 3

# DRAM-streaming large case
./Disjoint_SM_L2_DRAM_contention_test --mem_mib 1024 --compute_size 1536 --iterations 5 --repeats 5
```

The new `tma` mode computes the same streaming triad, but fetches the `a` and `b` input tiles with Hopper+ Tensor Memory Accelerator bulk copies from global memory into shared memory. It follows the local reference experiment in `../simple_sm_util_greencontext_tma`: one elected warp-0 thread issues `cuda::ptx::cp_async_bulk`, and a block-scope `cuda::barrier`/mbarrier tracks completion. After both input tiles arrive, the block computes the same `out[i] = a[i] + alpha * b[i]` expression and writes `out` to global memory.

TMA mode uses `--mem_mib` the same way as streaming mode: total working set across `a`, `b`, and `out`. It reports effective GiB/s with the same accounting as streaming mode, `2` global reads plus `1` global write per element. The key difference is the input transfer path: streaming mode uses normal per-thread global loads, while TMA mode uses bulk global-to-shared copies for the two input arrays.

TMA-specific options:

| Option | Meaning | Default |
|---|---|---|
| `--tma_tile_bytes <N>` | Global-to-shared bytes per TMA tile, multiple of 16 | `32768` |
| `--tma_blocks_per_sm <N>` | TMA blocks launched per memory-partition SM | `2` |

TMA requires compute capability 9.0 or newer.

### Compute-Bound WMMA Kernel

The compute kernel is an FP16 WMMA square GEMM microbenchmark using Tensor Core instructions with FP32 accumulation. `--compute_size` controls the square matrix dimension and is rounded to a multiple of 16.

To raise arithmetic intensity, each loaded 16x16x16 WMMA tile can be reused for several `mma_sync` operations via `--mma_repeats` (default 8, max 16). That makes the kernel a stronger Tensor Core pressure workload than a naive untiled WMMA sample and reduces the chance that it is simply another memory benchmark.

## Build

```bash
cd ~/lerobot_custom/experiments/Disjoint_SM_L2_DRAM_contention_test
cmake -S . -B build
cmake --build build -j
```

## Usage

```bash
cd ~/lerobot_custom/experiments/Disjoint_SM_L2_DRAM_contention_test/build

# Default: 8 SMs for memory, 8 SMs for compute, 1 GiB memory working set, 1536 GEMM
./Disjoint_SM_L2_DRAM_contention_test

# TMA triad mode
./Disjoint_SM_L2_DRAM_contention_test \
  --mem_mode tma \
  --mem_sms 8 --compute_sms 8 \
  --mem_mib 512 --compute_size 1152 \
  --iterations 5 --repeats 3 \
  --csv tma_results.csv

# Small smoke test
./Disjoint_SM_L2_DRAM_contention_test \
  --mem_sms 8 --compute_sms 8 \
  --mem_mib 48 --compute_size 512 \
  --iterations 1 --repeats 1 \
  --csv smoke.csv

# Optional L2 flush before timed launches
./Disjoint_SM_L2_DRAM_contention_test --flush_l2 --l2_flush_mib 256
```

Options:

| Option | Meaning |
|---|---|
| `--experiment <duration_matched|compute_under_memory_pressure>` | Benchmark behavior; default keeps the original duration-matched isolated/concurrent measurement |
| `--mem_mode <streaming|tma>` | Memory workload implementation |
| `--mem_sms <N>` | SMs assigned to the memory Green Context |
| `--compute_sms <N>` | SMs assigned to the compute Green Context |
| `--mem_mib <N>` | Total memory working set across the triad arrays |
| `--compute_size <N>` | Square WMMA GEMM dimension, rounded to a multiple of 16 |
| `--iterations <N>` | Timed kernel launches per repeat |
| `--repeats <N>` | Repeated measurements to average |
| `--tpb <N>` | Threads per block for memory and flush kernels |
| `--blocks_per_sm <N>` | Blocks launched per assigned SM for memory and flush kernels |
| `--mma_repeats <N>` | WMMA repeats per loaded tile, 1 through 16 |
| `--tma_tile_bytes <N>` | TMA global-to-shared bytes per tile |
| `--tma_blocks_per_sm <N>` | TMA blocks launched per assigned memory SM |
| `--csv <path>` | CSV output path |
| `--flush_l2` | Run a streaming flush before timed launches |
| `--l2_flush_mib <N>` | Per-context flush buffer size |

CSV output includes both the row type and memory implementation:

```csv
mode,mem_mode,mem_sms,compute_sms,...
isolated_memory,streaming,8,8,...
concurrent,tma,8,8,...
```

## Continuous Memory Pressure Mode

The original benchmark launches the memory and compute workloads with the same iteration count during the concurrent phase. Duration matching makes that comparison fairer, but one workload can still finish earlier than the other.

The `compute_under_memory_pressure` experiment changes only the concurrent phase. It measures isolated compute first, then starts memory pressure on the memory Green Context and times compute on the compute Green Context while a host thread keeps relaunching the selected memory triad until compute finishes:

```bash
cd ~/lerobot_custom/experiments/Disjoint_SM_L2_DRAM_contention_test/build

./Disjoint_SM_L2_DRAM_contention_test \
  --experiment compute_under_memory_pressure \
  --mem_mode streaming \
  --mem_sms 8 --compute_sms 8 \
  --mem_mib 512 --compute_size 1024 \
  --iterations 5 --repeats 3 \
  --csv streaming_pressure.csv

./Disjoint_SM_L2_DRAM_contention_test \
  --experiment compute_under_memory_pressure \
  --mem_mode tma \
  --mem_sms 8 --compute_sms 8 \
  --mem_mib 512 --compute_size 1024 \
  --iterations 5 --repeats 3 \
  --csv tma_pressure.csv
```

This mode reports compute isolated time, compute time under memory pressure, compute slowdown, TFLOP/s retention, memory launches started/completed while compute was active, and wall-clock overlap fields. It preserves the same disjoint Green Context partitioning and the same `streaming`/`tma` triad computation.

### Continuous-Pressure Implementation

The executable still creates two disjoint Green Contexts from one SM resource split:

- memory Green Context: runs the selected memory triad on `--mem_sms`
- compute Green Context: runs the WMMA kernel on `--compute_sms`

The memory and compute contexts use separate non-blocking streams created with `cuGreenCtxStreamCreate()`. The continuous-pressure path uses two host threads:

- memory thread: waits for a start flag, launches one complete memory triad, synchronizes the memory stream, increments launch counters, and repeats while compute is still running
- compute thread: waits until at least one memory launch has started, records CUDA events around `--iterations` WMMA launches, synchronizes on the stop event, then clears the running flag

The memory thread synchronizes after each memory launch intentionally. That keeps `memory_launches_started` and `memory_launches_completed` meaningful, prevents unbounded queue buildup, and guarantees the next launch is a fresh pressure request rather than a long prequeued tail. A long memory launch can still span the entire compute interval, which is why the CSV records both launch counts and wall-clock overlap.

Both memory modes run the same triad:

```text
out = a + alpha * b
```

`streaming` mode performs normal global loads from `a` and `b` and a global store to `out`. `tma` mode uses `cp_async_bulk` TMA global-to-shared copies for the `a` and `b` input tiles, waits on a block-scope barrier, computes the same triad from shared memory, and writes `out` to global memory.

### Continuous-Pressure Measurement

Each repeat performs:

1. Warm up and measure isolated compute on the compute Green Context.
2. Optionally flush L2 in each context if `--flush_l2` is set.
3. Start the memory-pressure thread and compute timing thread together.
4. Keep memory pressure active until the compute timing thread finishes.
5. Report compute slowdown as:

```text
compute_time_under_pressure / isolated_compute_time
```

Throughput retention is:

```text
100 * pressured_compute_TFLOP/s / isolated_compute_TFLOP/s
```

The wall-clock overlap fields are host-side sanity checks. `overlap_pct` is the fraction of the measured compute wall interval covered by the memory-pressure thread. For strongest interpretation, it should be close to `100%`; use Nsight Systems if you need a timeline-level CUDA overlap check.

Important continuous-pressure CSV fields:

| Field | Meaning |
|---|---|
| `memory_pressure_level` | Swept request-pressure setting; blocks per memory-partition SM |
| `memory_blocks_per_sm` | Streaming blocks per memory SM used for this row |
| `tma_blocks_per_sm` | TMA blocks per memory SM used for this row |
| `mem_mib` | Fixed memory working set across `a`, `b`, and `out`; choose larger than L2 for DRAM pressure |
| `compute_time_isolated_ms` | Per-iteration WMMA time with no memory pressure |
| `compute_time_concurrent_ms` | Per-iteration WMMA time while memory pressure is active |
| `compute_slowdown` | `compute_time_concurrent_ms / compute_time_isolated_ms` |
| `compute_retention_pct` | Pressured compute TFLOP/s as a percentage of isolated TFLOP/s |
| `memory_launches_started` | Complete memory triad launches started before compute stopped the pressure loop |
| `memory_launches_completed_before_compute_done` | Memory launches that finished before the compute interval ended |
| `pressure_wall_time_ms` | Host wall time from first memory launch to pressure-thread exit |
| `overlap_pct` | Host-estimated compute interval covered by memory pressure |

## Plot

```bash
cd ~/lerobot_custom/experiments/Disjoint_SM_L2_DRAM_contention_test
python3 tools/plot.py build/results.csv build/results
```

This creates `build/results_summary.png`.

## Size Sweep

Sweep memory working-set size and WMMA matrix size:

```bash
cd ~/lerobot_custom/experiments/Disjoint_SM_L2_DRAM_contention_test
./run_size_sweep.sh \
  --output_dir build/size_sweep_results \
  --mem_sizes "16 48 128 512 1024" \
  --compute_sizes "512 1024 1536 2048" \
  --iterations 3 \
  --repeats 3

python3 tools/plot_size_sweep.py \
  build/size_sweep_results/size_sweep_summary.csv \
  build/size_sweep_results
```

Generated outputs:

- `size_sweep_summary.csv`: one row per working-set/matrix-size pair
- `size_sweep_heatmaps.png`: slowdown, overlap, and isolated memory-throughput heatmaps
- `size_sweep_slowdown_lines.png`: slowdown vs compute size by working-set size
- `size_sweep_retention_lines.png`: throughput retention vs working-set size by compute size

## Duration-Matched Sweep

For the fairest contention comparison, first calibrate isolated runtimes, then run only pairs whose isolated memory and compute durations are close:

```bash
cd ~/lerobot_custom/experiments/Disjoint_SM_L2_DRAM_contention_test
./run_duration_matched_sweep.py \
  --output_dir build/duration_matched_sweep \
  --mem_modes "streaming tma" \
  --calibration_iterations 3 \
  --calibration_repeats 1 \
  --matched_iterations 5 \
  --matched_repeats 3 \
  --max_ratio 1.35

python3 tools/plot_duration_matched.py \
  build/duration_matched_sweep/duration_matched_summary.csv \
  build/duration_matched_sweep
```

Generated outputs:

- `memory_calibration.csv` and `compute_calibration.csv`: isolated duration sweeps; memory calibration includes `mem_mode`
- `selected_pairs.csv`: chosen memory/compute pairs and their isolated-duration ratio, per memory mode
- `duration_matched_summary.csv`: final matched-pair isolated and concurrent measurements with `mem_mode`
- `duration_matched_summary.png`: streaming and TMA curves on shared slowdown, retention, pair-quality, and overlap figures
- `duration_matched_slowdown_bars.png`: compact per-pair slowdown comparison across both modes

## Continuous Memory Pressure Sweep

Run both memory modes across memory request pressure and compute sizes. The memory working set stays fixed; set `--mem_mib` large enough to exceed L2 so the repeatedly relaunched memory workload keeps touching DRAM rather than turning the sweep into a cache-capacity study.

The `--pressure_levels` values are interpreted as blocks per memory-partition SM. For `streaming`, each level is passed as `--blocks_per_sm`. For `tma`, each level is passed as `--tma_blocks_per_sm`.

```bash
cd ~/lerobot_custom/experiments/Disjoint_SM_L2_DRAM_contention_test
./run_compute_under_memory_pressure_sweep.py \
  --output_dir build/compute_under_memory_pressure_sweep \
  --mem_modes "streaming tma" \
  --mem_mib 1024 \
  --pressure_levels "1 2 3 4 6 8 12 16 24 32" \
  --compute_sizes "512 768 1024 1280 1536" \
  --iterations 5 \
  --repeats 3

python3 tools/plot_compute_under_memory_pressure.py \
  build/compute_under_memory_pressure_sweep/compute_under_memory_pressure_summary.csv \
  build/compute_under_memory_pressure_sweep
```

Generated outputs:

- `compute_under_memory_pressure_summary.csv`: one row per mode/pressure-level/compute-size configuration
- `compute_under_memory_pressure_heatmaps.png`: compute slowdown, retention, memory launch count, and pressure/compute wall-time coverage by memory mode
- `compute_under_memory_pressure_lines.png`: streaming vs TMA line comparisons over memory request pressure

Figure details:

- `compute_under_memory_pressure_heatmaps.png` has one column per memory mode. Rows are memory pressure levels and columns are compute sizes. The four heatmap rows show compute slowdown, compute throughput retention, memory launches started, and pressure-wall/compute-wall coverage. The pressure-wall coverage panel helps identify rows where the pressure loop lasted much longer than compute because a single memory launch was still draining after compute ended.
- `compute_under_memory_pressure_lines.png` plots the same data as lines. Color identifies memory mode, line style identifies memory mode, and marker shape identifies compute size, so C256/C512/etc. remain visually distinct. The compute-time panel uses dotted lines for isolated compute time and solid/dashed lines for compute under pressure.

## Nsight Systems Overlap Check

If `nsys` is installed:

```bash
cd ~/lerobot_custom/experiments/Disjoint_SM_L2_DRAM_contention_test/build
nsys profile --trace=cuda,nvtx --stats=false -o disjoint_short \
  ./Disjoint_SM_L2_DRAM_contention_test \
  --mem_mode streaming --mem_sms 8 --compute_sms 8 --mem_mib 128 --compute_size 1024 \
  --iterations 2 --repeats 1 --csv nsys_short.csv

nsys export --type sqlite --force-overwrite=true -o disjoint_short.sqlite disjoint_short.nsys-rep
python3 ../tools/nsys_overlap.py disjoint_short.sqlite
```

The program also reports a host/event-based overlap estimate, but Nsight Systems is the better validation. In streaming mode, look for `streaming_triad_kernel` and `wmma_compute_kernel` intervals overlapping on different Green Context streams. In TMA mode, look for `tma_streaming_triad_kernel` instead of `streaming_triad_kernel`.

## Interpreting Results

`memory slowdown > 1` means the streaming kernel took longer while the compute partition was active.

`compute slowdown > 1` means the WMMA kernel took longer while the streaming partition was active.

When `overlap_pct` is high, slowdowns are more meaningful because the kernels were active at the same time. If overlap is low, tune `--mem_mib`, `--compute_size`, `--iterations`, or `--mma_repeats` so per-kernel runtimes are closer.

Duration-matched contention answers: "What happens when a memory workload and compute workload with similar isolated durations run together for the same number of launches?"

Continuous-memory-pressure compute slowdown answers: "How much does compute slow down when memory pressure is kept present for the whole compute interval?" Use this mode when the compute kernel is the primary workload and the memory kernel is intended to be background pressure rather than a symmetric peer.

Comparing small and large `--mem_mib` values helps separate cache-resident behavior from DRAM streaming. A small working set near or below L2 may mostly test L2 capacity/bandwidth. A large working set well above L2 should put sustained pressure on DRAM and the L2/DRAM path.

When comparing `streaming` and `tma`, both modes now perform the same triad computation and use the same effective-bandwidth accounting. Remaining differences come from the transfer mechanism and its side effects: normal LD/ST input loads in streaming mode versus TMA bulk global-to-shared input copies in TMA mode.

## Green Context Limitations

CUDA Green Contexts provide disjoint SM resource partitions, not stable physical SM ID selection. This experiment does not claim that the memory kernel runs on specific SM numbers, only that CUDA created disjoint SM resource sets.

CUDA documentation also notes that disjoint Green Context partitions do not guarantee concurrency or forward progress in every situation. Other GPU resources, hardware connections, dynamic parallelism behavior, MPS settings, and scheduler details can still affect execution. Use Nsight Systems when overlap itself is part of the claim.
