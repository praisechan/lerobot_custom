import argparse
import os
import statistics
import time

import torch

from pi0_infer_greenctx import Pi0GreenContextInference


def load_checkpoint(checkpoint_dir, prompt_len):
    if checkpoint_dir:
        return torch.load(os.path.join(checkpoint_dir, "checkpoint.pth"), map_location="cpu")
    return {"language_embeds": torch.empty(prompt_len, 2048, dtype=torch.bfloat16)}


def make_inputs(num_views, chunk_size):
    return (
        torch.randn(num_views, 224, 224, 3, dtype=torch.bfloat16, device="cuda"),
        torch.randn(32, dtype=torch.bfloat16, device="cuda"),
        torch.randn(chunk_size, 32, dtype=torch.bfloat16, device="cuda"),
    )


def summarize_ms(samples):
    mean_ms = statistics.fmean(samples)
    median_ms = statistics.median(samples)
    if len(samples) > 1:
        stdev_ms = statistics.stdev(samples)
    else:
        stdev_ms = 0.0
    return mean_ms, median_ms, stdev_ms


def measure(name, fn, iterations, warmup):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        stop = time.perf_counter()
        samples.append((stop - start) * 1000.0)
    mean_ms, median_ms, stdev_ms = summarize_ms(samples)
    print(f"{name}: mean={mean_ms:.3f} ms median={median_ms:.3f} ms stdev={stdev_ms:.3f} ms")
    return {
        "name": name,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "stdev_ms": stdev_ms,
        "samples_ms": samples,
    }


def cuda_profiler_start():
    result = torch.cuda.cudart().cudaProfilerStart()
    if result not in (None, 0):
        raise RuntimeError(f"cudaProfilerStart failed: {result}")


def cuda_profiler_stop():
    result = torch.cuda.cudart().cudaProfilerStop()
    if result not in (None, 0):
        raise RuntimeError(f"cudaProfilerStop failed: {result}")


def print_greenctx_info(info):
    print("GreenContext configuration:")
    keys = [
        "device_index",
        "total_sms",
        "min_sm_partition_size",
        "sm_coscheduled_alignment",
        "base_sms",
        "split_groups_created",
        "encoder_requested_sms",
        "encoder_actual_sms",
        "encoder_groups",
        "encoder_green_ctx_id",
        "encoder_stream_id",
        "encoder_stream_ptr",
        "decoder_requested_sms",
        "decoder_actual_sms",
        "decoder_groups",
        "decoder_green_ctx_id",
        "decoder_stream_id",
        "decoder_stream_ptr",
    ]
    for key in keys:
        value = info[key]
        if key.endswith("_ptr"):
            value = hex(value)
        print(f"  {key}: {value}")
    print("  disjoint_sm_resources: true (non-overlapping groups from one split call)")


def main():
    parser = argparse.ArgumentParser(
        description="Pi0 encoder/decoder latency and contention experiment using CUDA GreenContext SM partitions."
    )
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--prompt_len", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=63)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--encoder-sms", type=int, required=True)
    parser.add_argument("--decoder-sms", type=int, required=True)
    parser.add_argument("--profile-markers", action="store_true")
    parser.add_argument(
        "--fresh-baseline-streams",
        action="store_true",
        help="Use Pi0Inference.forward(..., concurrent=True) exactly, which creates new streams each iteration.",
    )
    parser.add_argument(
        "--greenctx-only",
        action="store_true",
        help="Skip full-SM baselines to keep Nsight profiles focused on GreenContext streams.",
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Call cudaProfilerStart/Stop around timed sections for nsys --capture-range=cudaProfilerApi.",
    )
    parser.add_argument(
        "--direct-greenctx",
        action="store_true",
        help="Run GreenContext stages directly instead of replaying CUDA graphs. Use this when you want GreenCtx NVTX projected onto CUDA HW kernel rows.",
    )
    parser.add_argument(
        "--pre-measure-sleep",
        type=float,
        default=0.0,
        help="Sleep after setup and before measurements. Useful with nsys --delay to exclude setup streams from the report.",
    )
    parser.add_argument(
        "--case",
        choices=[
            "all",
            "sequential",
            "stream-concurrent",
            "green-encoder",
            "green-decoder",
            "green-concurrent",
        ],
        default="all",
        help="Run one measurement case or all cases.",
    )
    args = parser.parse_args()

    torch.manual_seed(100)
    torch.cuda.init()

    checkpoint = load_checkpoint(args.checkpoint_dir, args.prompt_len)
    infer = Pi0GreenContextInference(
        checkpoint,
        num_views=args.num_views,
        chunk_size=args.chunk_size,
        encoder_sms=args.encoder_sms,
        decoder_sms=args.decoder_sms,
        verbose=True,
    )

    input_image, input_state, input_noise = make_inputs(args.num_views, args.chunk_size)
    infer.prepare_inputs(input_image, input_state, input_noise)
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("Real test starts!!!") 
    print(
        "Pi0 dimensions: "
        f"num_views={args.num_views} prompt_len={infer.prompt_len} chunk_size={args.chunk_size} "
        f"encoder_seq_len={infer.encoder_seq_len} decoder_seq_len={infer.decoder_seq_len}"
    )
    print_greenctx_info(infer.greenctx_info())
    if args.pre_measure_sleep > 0:
        print(f"Sleeping {args.pre_measure_sleep:.3f} s before measurements")
        time.sleep(args.pre_measure_sleep)

    def sequential_full_model():
        infer.greenctx.restore_primary_current()
        _ = infer.forward(input_image, input_state, input_noise, concurrent=False)
        torch.cuda.synchronize()

    def stream_concurrent_full_sm():
        infer.greenctx.restore_primary_current()
        if args.fresh_baseline_streams:
            _ = infer.forward(input_image, input_state, input_noise, concurrent=True)
        else:
            _ = infer.forward_concurrent_cached_streams(input_image, input_state, input_noise)
        torch.cuda.synchronize()

    def green_encoder_only():
        if args.direct_greenctx:
            infer.run_encoder_direct_greenctx(
                synchronize=True, profile_markers=args.profile_markers
            )
        else:
            infer.replay_encoder_greenctx(
                synchronize=True, profile_markers=args.profile_markers
            )

    def green_decoder_only():
        if args.direct_greenctx:
            infer.run_decoder_direct_greenctx(
                synchronize=True, profile_markers=args.profile_markers
            )
        else:
            infer.replay_decoder_greenctx(
                synchronize=True, profile_markers=args.profile_markers
            )

    def green_concurrent():
        if args.direct_greenctx:
            infer.run_concurrent_direct_greenctx(profile_markers=args.profile_markers)
        else:
            infer.replay_concurrent_greenctx(profile_markers=args.profile_markers)

    if args.cuda_profiler_range:
        torch.cuda.synchronize()
        cuda_profiler_start()

    try:
        results = []
        if not args.greenctx_only and args.case in ("all", "sequential"):
            results.append(
                measure(
                    "1. sequential full-model baseline (full SM)",
                    sequential_full_model,
                    args.iterations,
                    args.warmup,
                )
            )
        if not args.greenctx_only and args.case in ("all", "stream-concurrent"):
            results.append(
                measure(
                    "2. stream-concurrent baseline (full SM shared)",
                    stream_concurrent_full_sm,
                    args.iterations,
                    args.warmup,
                )
            )

        if args.case in ("all", "green-encoder", "green-decoder", "green-concurrent"):
            infer.prepare_inputs(input_image, input_state, input_noise)
        if args.case == "all":
            infer.replay_encoder_greenctx(synchronize=True, profile_markers=False)

        if args.case in ("all", "green-encoder"):
            results.append(
                measure(
                    "3. GreenContext encoder-only (VLM partition)",
                    green_encoder_only,
                    args.iterations,
                    args.warmup,
                )
            )

        if args.case in ("all", "green-decoder"):
            results.append(
                measure(
                    "4. GreenContext decoder-only (DiT partition)",
                    green_decoder_only,
                    args.iterations,
                    args.warmup,
                )
            )

        if args.case in ("all", "green-concurrent"):
            results.append(
                measure(
                    "5. GreenContext concurrent encoder+decoder (disjoint SM)",
                    green_concurrent,
                    args.iterations,
                    args.warmup,
                )
            )
    finally:
        if args.cuda_profiler_range:
            torch.cuda.synchronize()
            cuda_profiler_stop()

    if args.case == "all":
        enc = results[-3]["median_ms"]
        dec = results[-2]["median_ms"]
        conc = results[-1]["median_ms"]
        mode = "direct GreenContext execution" if args.direct_greenctx else "GreenContext CUDA graph replay"
        print(f"Derived slowdown/overlap ({mode}):")
        print(f"  concurrent_vs_encoder_only: {conc / enc:.3f}x")
        print(f"  concurrent_vs_decoder_only: {conc / dec:.3f}x")
        print(f"  ideal_disjoint_wall_from_isolated: {max(enc, dec):.3f} ms")
        print(f"  concurrent_over_ideal_disjoint_wall: {conc / max(enc, dec):.3f}x")

    torch.cuda.nvtx.range_push("Real test starts!!!")

if __name__ == "__main__":
    main()
