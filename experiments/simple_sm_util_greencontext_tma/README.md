# SM Bandwidth Sweep with CUDA Green Contexts and TMA

This directory is a TMA-enabled copy of `../simple_sm_util_greencontext`. The original directory is the read-only baseline/reference implementation and was not modified.

## What The Original Benchmark Does

The original benchmark uses CUDA Green Contexts to split the device SM resource and run the same read-bandwidth kernel with different SM counts. For each SM count it:

- Creates a Green Context containing the requested SM partition.
- Allocates a large global input buffer and a tiny global sink buffer.
- Launches `bandwidth_read_kernel` with `2` blocks per SM and `256` threads per block by default.
- Has each thread issue strided, coalesced `Vec4U32` loads, where each load is 16 bytes.
- Mixes every loaded value into a register checksum and writes one `uint64_t` sink value per block.
- Times one kernel launch with CUDA events after three warm-up launches.
- Repeats measurements and reports mean/stdev bandwidth.

The original benchmark does not use TMA. Its device code is ordinary vectorized global-load code; disassembly shows `LDG.E.128` loads in the baseline kernel.

## Modes

- `global`: Preserves the original benchmark behavior as closely as possible. Same launch geometry, same `Vec4U32` global-load kernel, same block-level checksum pattern.
- `tma`: Uses Hopper+ one-dimensional TMA bulk copies from global memory to shared memory via `cuda::ptx::cp_async_bulk`. One elected warp-0 thread issues each bulk copy, and a block-scope `cuda::barrier`/mbarrier tracks copy completion.
- `both`: Runs `global` then `tma` for each SM count and writes both curves to one CSV.

The TMA mode is intentionally not a semantic clone of the global-load checksum loop. It measures global-to-shared TMA bulk-copy bandwidth, including the required TMA completion barrier and a tiny shared-memory checksum per tile.

## Build

```bash
cd /home/juchan.lee/lerobot_custom/experiments/simple_sm_util_greencontext_tma
cmake -S . -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda-13.0
cmake --build build -j
```

The copied CMake architecture list remains `80;90` to avoid changing the original baseline compiler settings. On the local GB10 test system, the sm_90 image contains the TMA bulk-copy instruction sequence and runs successfully.

## Usage

```bash
# Compare global LDG baseline vs TMA bulk copy
./build/sm_bw_sweep --mode both --csv results.csv

# Baseline only
./build/sm_bw_sweep --mode global --csv global.csv

# TMA only with a larger tile
./build/sm_bw_sweep --mode tma --tma_tile_bytes 65536 --tma_blocks_per_sm 2 --csv tma.csv
```

Options:

| Option | Description | Default |
| --- | --- | --- |
| `--mode <global|tma|both>` | Select benchmark mode | `both` |
| `--min_sms <N>` | Minimum SM count | `8` |
| `--max_sms <N>` | Maximum SM count | device max |
| `--bytes <N>` | Bytes read/copied per measurement | `1073741824` |
| `--tpb <N>` | Threads per block | `256` |
| `--iters <N>` | Override baseline global-load iterations | auto |
| `--repeats <N>` | Measurement repeats | `5` |
| `--csv <path>` | CSV output path | `./results.csv` |
| `--tma_tile_bytes <N>` | TMA bytes per global-to-shared tile | `32768` |
| `--tma_blocks_per_sm <N>` | TMA blocks per Green Context SM | `2` |

## CSV And Plotting

CSV output includes both mode and statistics:

```csv
mode,num_sms,mean_bw_gb_s,stdev_bw_gb_s
global,8,198.745,1.181
tma,8,229.741,1.439
```

Plot both curves:

```bash
python3 tools/plot.py results.csv results.png
```

## Verification

Before TMA changes were implemented, the copied baseline was built and run with the original code. On the local NVIDIA GB10 system it reproduced the expected early saturation trend:

```text
8 SMs   199.56 +/- 0.60 GiB/s
16 SMs  221.53 +/- 3.52 GiB/s
24 SMs  226.94 +/- 0.48 GiB/s
32 SMs  211.97 +/- 9.48 GiB/s
40 SMs  224.77 +/- 3.11 GiB/s
48 SMs  226.19 +/- 1.41 GiB/s
```

Generated-code checks:

```bash
/usr/local/cuda-13.0/bin/cuobjdump --dump-sass build/sm_bw_sweep | grep 'UBLKCP\|LDG.E.128'
```

- `global` baseline kernel contains `LDG.E.128` vectorized global loads.
- `tma` kernel contains `UBLKCP.S.G` plus mbarrier/SYNCS transaction instructions, corresponding to `cp.async.bulk` global-to-shared bulk copy.

## Methodology And Caveats

- Bandwidth is reported in GiB/s, matching the original benchmark calculation.
- Green Context SM counts are rounded to the device resource minimum partition size and co-scheduled alignment.
- TMA requires compute capability 9.0 or newer and exits early on older GPUs.
- TMA tile size must be a multiple of 16 bytes; the default 32 KiB tile satisfies the one-dimensional bulk-copy alignment requirements from NVIDIA documentation.
- TMA mode uses shared memory, so its occupancy/resource profile differs from the LDG baseline. Tune `--tma_tile_bytes` and `--tma_blocks_per_sm` when exploring TMA-specific saturation.
- The benchmark isolates read/copy paths, not end-to-end application performance.

## References

- NVIDIA CUDA Programming Guide, "Asynchronous Data Copies", especially "Using TMA to transfer one-dimensional arrays": https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/04-special-topics/async-copies.html
- NVIDIA CUDA Driver API Green Contexts: https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html
- Local Green Context notes: [docs/green_context_reference.md](docs/green_context_reference.md)
