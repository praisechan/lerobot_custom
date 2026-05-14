# Pi0 GreenContext Latency Experiment

This experiment measures Pi0 encoder and decoder contention while forcing the two stages onto disjoint CUDA GreenContext SM partitions.

The original `pi0_infer.py`, `benchmark.py`, and `pi0_infer_concurrent.py` paths are unchanged. The new entry point is:

```bash
python pi0_infer_greenctx_experiment.py \
  --num_views 2 \
  --prompt_len 0 \
  --chunk_size 63 \
  --encoder-sms <N> \
  --decoder-sms <M> \
  --iterations 50 \
  --warmup 5
```

Optional checkpoint:

```bash
python pi0_infer_greenctx_experiment.py \
  --checkpoint-dir /path/to/checkpoint_dir \
  --num_views 2 \
  --encoder-sms <N> \
  --decoder-sms <M>
```

The helper extension is built automatically on first import through `torch.utils.cpp_extension`. Set `PI0_GREENCTX_BUILD_VERBOSE=1` to show the build command.

## Environment Setup Used

Do not run this with the base `python` unless that environment already has CUDA-enabled PyTorch, Triton, and Ninja. The profiling runs above used a local venv in this directory:

```bash
cd /home/juchan.lee/lerobot_custom/3rdparty/realtime-vla

python -m venv .greenctx_venv
source .greenctx_venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.11.0 triton==3.6.0 matplotlib ninja
```

Run the experiment from that venv and point extension builds at the local CUDA 13 install:

```bash
source .greenctx_venv/bin/activate

CUDA_HOME=/usr/local/cuda-13.0 python pi0_infer_greenctx_experiment.py \
  --num_views 2 \
  --encoder-sms 24 \
  --decoder-sms 24 \
  --iterations 5 \
  --warmup 1 \
  --case green-concurrent
```

Useful sanity checks:

```bash
which python
python - <<'PY'
import torch, triton
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("sms", torch.cuda.get_device_properties(0).multi_processor_count)
print("triton", triton.__version__)
PY
```

Expected on the GB10 machine used for these runs:

```text
.../realtime-vla/.greenctx_venv/bin/python
torch 2.11.0+cu130
torch cuda 13.0
cuda available True
device NVIDIA GB10
sms 48
triton 3.6.0
```

The first run builds `greenctx_helper.cpp` through `torch.utils.cpp_extension`, so `ninja` is required. To show the build command:

```bash
PI0_GREENCTX_BUILD_VERBOSE=1 CUDA_HOME=/usr/local/cuda-13.0 python pi0_infer_greenctx_experiment.py ...
```

## What It Reports

The script reports:

1. Sequential full-model baseline on the normal full-SM CUDA context.
2. Existing stream-concurrent baseline equivalent to `forward(..., concurrent=True)`, still sharing all SMs.
3. GreenContext encoder-only latency on the VLM partition.
4. GreenContext decoder-only latency on the DiT partition.
5. GreenContext concurrent encoder+decoder latency on disjoint SM partitions.

It also logs GreenContext IDs, stream IDs, requested SM counts, actual SM counts, and the split granularity. The encoder and decoder partitions are disjoint because their descriptors are built from non-overlapping groups returned by one `cuDevSmResourceSplitByCount` call.

For uneven splits such as `--encoder-sms 40 --decoder-sms 8` on a 48-SM GB10, the helper first tries CUDA's default co-scheduling-aware split. CUDA may create fewer symmetric groups than the arithmetic split would suggest; for example, it can return only four 8-SM groups when six are needed. In that case the helper retries with `CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING`, then combines non-overlapping 8-SM groups into the requested 40-SM and 8-SM GreenContext descriptors. The runtime log exposes this as:

```text
split_use_flags: 1
split_ignores_sm_coscheduling: True
```

This enables finer uneven SM partitions, but it opts out of CUDA's default SM co-scheduling hierarchy guarantees for that split.

## Profiling

Use Nsight Systems with CUDA and NVTX tracing:

```bash
nsys profile -t cuda,nvtx,osrt \
  -o pi0_greenctx_2v_50_50 \
  python pi0_infer_greenctx_experiment.py \
    --num_views 2 \
    --encoder-sms <N> \
    --decoder-sms <M> \
    --iterations 50 \
    --profile-markers
```

For CUDA graph replay on GreenContext streams, prefer node-level CUDA graph tracing:

```bash
nsys profile -t cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  --cuda-memory-usage true \
  --gpu-metrics-devices=all \
  -o pi0_greenctx_graph_nodes_2v_24_24 \
  python pi0_infer_greenctx_experiment.py \
    --num_views 2 \
    --encoder-sms 24 \
    --decoder-sms 24 \
    --iterations 5 \
    --warmup 0 \
    --case green-concurrent
```

`--cuda-graph-trace=node` does not change the program's CUDA graph execution. The script still replays instantiated graphs through `torch.cuda.CUDAGraph.replay()`, which maps to `cudaGraphLaunch`. This Nsight option only changes profiling granularity:

- `--cuda-graph-trace=graph` records one high-level activity for each CUDA graph launch.
- `--cuda-graph-trace=node` records graph node/kernel activities inside each launched graph.

On the GB10 GreenContext run, default graph-level tracing can drop the encoder graph activity as an incomplete CUPTI event, leaving only the decoder graph visible in the CUDA HW tab. Node-level tracing keeps CUDA graph execution enabled while making both encoder and decoder replay work visible through graph-node kernel rows. Node tracing records more events, so use it for timeline/debug visibility rather than final latency numbers.

Inspect the `.nsys-rep` with:

```bash
nsys-ui pi0_greenctx_2v_50_50.nsys-rep
```

Confirm these items:

- `GreenCtx Concurrent Total` spans overlapping `GreenCtx Concurrent Encoder` and `GreenCtx Concurrent Decoder` NVTX ranges.
- CUDA kernels inside those ranges are launched on different stream IDs. Match them with the runtime log lines:
  - `encoder_green_ctx_id`, `encoder_stream_id`, `encoder_stream_ptr`
  - `decoder_green_ctx_id`, `decoder_stream_id`, `decoder_stream_ptr`
- The runtime log shows `encoder_actual_sms == --encoder-sms` and `decoder_actual_sms == --decoder-sms`.
- The log line `partitions are disjoint by construction` means both GreenContext descriptors were generated from non-overlapping resource groups returned by the same split call.

If Nsight Systems does not display GreenContext partition IDs directly, use the script's runtime log as the authoritative mapping from stream ID to GreenContext ID and SM count.
