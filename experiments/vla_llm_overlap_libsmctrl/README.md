# VLA + Cosmos LLM Overlap With libsmctrl

This experiment profiles contention between Pi0 VLA stages and Cosmos-Reason2 LLM stages while keeping them on disjoint RTX A6000 TPC partitions with libsmctrl.

It is intentionally **not** a CUDA GreenContext experiment. RTX A6000 does not support GreenContext, so this uses libsmctrl stream masks. libsmctrl masks TPCs, not individual SMs; on Ampere RTX A6000 this is expected to be 2 SMs per TPC. The script prints the detected GPU name, total SMs, total TPCs, SMs per TPC, selected TPC ranges, actual SM counts, stream object ids, stream pointers, and disabled-mask values at startup.

Important mask semantic: libsmctrl uses inverted masks. A set bit means that TPC is disabled.

## Layout

- `overlap_experiment.py`: main benchmark and Nsight Systems target.
- `libsmctrl_extension.py`: JIT builds a small Python extension around libsmctrl.
- `csrc/smctrl_pybind.cpp`: pybind wrapper for TPC info, mask creation, stream masks, and optional validation.
- `build.sh`: prebuilds/imports the extension and prints TPC info.
- `scripts/run_smoke.sh`: tiny VLA-only smoke run.
- `results/`: default CSV output directory.

The VLA code is imported from:

```bash
/home/juchanlee/lerobot_custom/3rdparty/realtime-vla/pi0_infer.py
```

The original realtime-vla files are not modified.

## Setup

Use the Cosmos-Reason2 environment:

```bash
source /home/juchanlee/lerobot_custom/3rdparty/cosmos-reason2/.venv/bin/activate
cd /home/juchanlee/lerobot_custom/experiments/vla_llm_overlap_libsmctrl

python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("has Qwen3VL", hasattr(transformers, "Qwen3VLForConditionalGeneration"))
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("sms", torch.cuda.get_device_properties(0).multi_processor_count)
PY
```

If `which python` does not point inside `/home/juchanlee/lerobot_custom/3rdparty/cosmos-reason2/.venv`, run commands with the explicit interpreter:

```bash
/home/juchanlee/lerobot_custom/3rdparty/cosmos-reason2/.venv/bin/python overlap_experiment.py ...
```

Prebuild the libsmctrl extension:

```bash
./build.sh
```

Set `VLA_LLM_SMCTRL_BUILD_VERBOSE=1` to show the extension build command during normal experiment runs.

## Run

Default model is `nvidia/Cosmos-Reason2-8B`. The 8B model is expected to need around 32GB before experiment buffers. If it cannot load on the local RTX A6000 setup, the script fails with a clean error; use `--model nvidia/Cosmos-Reason2-2B` for smoke tests.

Example full run:

```bash
python overlap_experiment.py \
  --vla-tpc-start 0 --vla-tpc-count 16 \
  --llm-tpc-start 16 --llm-tpc-count 16 \
  --case all \
  --iterations 20 \
  --warmup 3 \
  --profile-markers
```

Use direct VLA calls instead of Pi0 graph replay for clearer kernel/NVTX attribution:

```bash
python overlap_experiment.py \
  --vla-tpc-start 0 --vla-tpc-count 16 \
  --llm-tpc-start 16 --llm-tpc-count 16 \
  --case overlap-vla-decoder-llm-prefill \
  --iterations 5 \
  --warmup 1 \
  --vla-direct \
  --profile-markers
```

Tiny VLA-only smoke run:

```bash
./scripts/run_smoke.sh
```

LLM smoke run with the smaller model:

```bash
python overlap_experiment.py \
  --vla-tpc-start 0 --vla-tpc-count 1 \
  --llm-tpc-start 1 --llm-tpc-count 1 \
  --case llm-decode-only \
  --model nvidia/Cosmos-Reason2-2B \
  --llm-prompt-tokens 32 \
  --iterations 1 \
  --warmup 0
```

## Cases

```text
--case vla-encoder-only
--case llm-decode-only
--case overlap-vla-encoder-llm-decode
--case vla-decoder-only
--case llm-prefill-only
--case overlap-vla-decoder-llm-prefill
--case all
```

Overlap cases launch the VLA and LLM stage functions from two Python threads with a `threading.Barrier`, then synchronize the two masked streams before returning.

## Profiling

Use Nsight Systems with CUDA, NVTX, OS runtime, CUDA memory usage, and GPU metrics:

```bash
nsys profile -t cuda,nvtx,osrt \
  --cuda-memory-usage true \
  --gpu-metrics-devices=all \
  -o results/vla_llm_overlap \
  python overlap_experiment.py \
    --vla-tpc-start 0 --vla-tpc-count 16 \
    --llm-tpc-start 16 --llm-tpc-count 16 \
    --case all \
    --iterations 20 \
    --warmup 3 \
    --profile-markers
```

For graph replay visibility, add CUDA graph node tracing:

```bash
nsys profile -t cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  --cuda-memory-usage true \
  --gpu-metrics-devices=all \
  -o results/vla_llm_overlap_graph_nodes \
  python overlap_experiment.py \
    --vla-tpc-start 0 --vla-tpc-count 16 \
    --llm-tpc-start 16 --llm-tpc-count 16 \
    --case all \
    --iterations 5 \
    --warmup 1 \
    --profile-markers
```

Confirm these NVTX ranges in the timeline:

- `VLA Encoder Only`
- `VLA Decoder Only`
- `LLM Prefill Only`
- `LLM Decode Only`
- `Overlap VLA Encoder + LLM Decode Total`
- `Overlap VLA Encoder`
- `Overlap LLM Decode`
- `Overlap VLA Decoder + LLM Prefill Total`
- `Overlap VLA Decoder`
- `Overlap LLM Prefill`

## Output

The script prints per-case mean, median, and standard deviation in milliseconds. For overlap cases, when the corresponding isolated runs are present, it also prints:

- `concurrent_over_max_isolated`
- `concurrent_vs_vla_only`
- `concurrent_vs_llm_only`

CSV is written to `results/vla_llm_overlap_<timestamp>.csv` unless `--csv PATH` is provided.

## Notes

- Partition arguments are TPC ranges/counts, not SM counts.
- Ranges must be disjoint and within the detected total TPC count.
- Stream masks are set with `libsmctrl_set_stream_mask_ext` on persistent `torch.cuda.Stream` objects.
- `--validate-stream-masks` calls libsmctrl's validator for each stream and echoes validator output.
- VLA dummy checkpoint format matches the GreenContext experiment: `{"language_embeds": torch.empty(prompt_len, 2048, dtype=torch.bfloat16)}`.
- On the local CUDA 12.9 driver, BulletServe's libsmctrl table does not yet name a 12.9 stream-mask offset. The runner sets `MASK_OFF=24` automatically when the driver reports `12090+`, which reuses the CUDA 12.7/12.8 offset (`0x4fc`). Override `MASK_OFF` explicitly if you are validating another driver build.
