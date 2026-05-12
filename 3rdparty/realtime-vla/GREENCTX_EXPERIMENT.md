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

## What It Reports

The script reports:

1. Sequential full-model baseline on the normal full-SM CUDA context.
2. Existing stream-concurrent baseline equivalent to `forward(..., concurrent=True)`, still sharing all SMs.
3. GreenContext encoder-only latency on the VLM partition.
4. GreenContext decoder-only latency on the DiT partition.
5. GreenContext concurrent encoder+decoder latency on disjoint SM partitions.

It also logs GreenContext IDs, stream IDs, requested SM counts, actual SM counts, and the split granularity. The encoder and decoder partitions are disjoint because their descriptors are built from non-overlapping groups returned by one `cuDevSmResourceSplitByCount` call.

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
