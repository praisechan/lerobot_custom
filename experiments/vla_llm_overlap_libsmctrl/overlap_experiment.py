import argparse
import csv
import math
import os
import statistics
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

from libsmctrl_extension import load_libsmctrl_extension


ROOT = Path(__file__).resolve().parent
LEROBOT_ROOT = ROOT.parents[1]
REALTIME_VLA_ROOT = LEROBOT_ROOT / "3rdparty" / "realtime-vla"

CASES = (
    "vla-encoder-only",
    "llm-decode-only",
    "overlap-vla-encoder-llm-decode",
    "vla-decoder-only",
    "llm-prefill-only",
    "overlap-vla-decoder-llm-prefill",
    "all",
)

DEFAULT_LLM_DECODE_TOKENS = 4


@contextmanager
def nvtx_range(name, enabled):
    if not enabled:
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def stream_ptr(stream):
    return int(getattr(stream, "cuda_stream"))


def summarize_ms(samples):
    mean_ms = statistics.fmean(samples)
    median_ms = statistics.median(samples)
    stdev_ms = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return mean_ms, median_ms, stdev_ms


def ratio_or_nan(numerator, denominator):
    return numerator / denominator if denominator and not math.isnan(denominator) else math.nan


@dataclass
class Measurement:
    case: str
    samples_ms: list[float]

    @property
    def mean_ms(self):
        return summarize_ms(self.samples_ms)[0]

    @property
    def median_ms(self):
        return summarize_ms(self.samples_ms)[1]

    @property
    def stdev_ms(self):
        return summarize_ms(self.samples_ms)[2]


class VLARunner:
    def __init__(self, args, stream):
        if str(REALTIME_VLA_ROOT) not in sys.path:
            sys.path.insert(0, str(REALTIME_VLA_ROOT))
        from pi0_infer import Pi0Inference, decoder_model, encoder_model

        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.stream = stream
        self.direct = bool(args.vla_direct)
        self.num_views = args.num_views
        self.chunk_size = args.chunk_size
        self.prompt_len = args.prompt_len

        checkpoint = self._load_checkpoint(args.checkpoint_dir, args.prompt_len)
        self.infer = Pi0Inference(
            checkpoint,
            num_views=args.num_views,
            chunk_size=args.chunk_size,
        )
        self.input_image = torch.randn(
            args.num_views, 224, 224, 3, dtype=torch.bfloat16, device="cuda"
        )
        self.input_state = torch.randn(32, dtype=torch.bfloat16, device="cuda")
        self.input_noise = torch.randn(
            args.chunk_size, 32, dtype=torch.bfloat16, device="cuda"
        )
        self.prepare_inputs()
        torch.cuda.synchronize()

    @staticmethod
    def _load_checkpoint(checkpoint_dir, prompt_len):
        if checkpoint_dir:
            return torch.load(Path(checkpoint_dir) / "checkpoint.pth", map_location="cpu")
        return {"language_embeds": torch.empty(prompt_len, 2048, dtype=torch.bfloat16)}

    def prepare_inputs(self):
        self.infer.buffers["observation_images_normalized"].copy_(self.input_image)
        self.infer.buffers["observation_state_normalized"].copy_(self.input_state)
        self.infer.buffers["diffusion_noise"].copy_(self.input_noise)
        if self.infer.prompt_len:
            offset = self.infer.num_views * 256
            self.infer.buffers["encoder_x"][offset:].copy_(self.infer.weights["language_embeds"])

    def encoder_stage(self, nvtx_name=None, profile_markers=False, synchronize=True):
        with torch.cuda.stream(self.stream):
            with nvtx_range(nvtx_name or "VLA Encoder", profile_markers):
                if self.direct:
                    self.encoder_model(self.infer.weights, self.infer.buffers, self.infer.num_views)
                else:
                    self.infer.encoder_graph.replay()
        if synchronize:
            self.stream.synchronize()

    def decoder_stage(self, nvtx_name=None, profile_markers=False, synchronize=True):
        with torch.cuda.stream(self.stream):
            with nvtx_range(nvtx_name or "VLA Decoder", profile_markers):
                if self.direct:
                    self.decoder_model(
                        self.infer.weights, self.infer.buffers, self.infer.encoder_seq_len
                    )
                else:
                    self.infer.decoder_graph.replay()
        if synchronize:
            self.stream.synchronize()


class LLMRunner:
    def __init__(self, args, stream):
        self.args = args
        self.stream = stream
        self.decode_token_count = args.llm_decode_tokens
        self.prompt_tokens = args.llm_prompt_tokens
        self.model_name = args.model
        self.model = None
        self.tokenizer = None
        self.input_ids = None
        self.attention_mask = None
        self.decode_input_ids = None
        self.decode_attention_mask = None
        self.decode_past_key_values = None
        self._load_model()
        self._build_inputs()
        self.prepare_decode_cache()
        torch.cuda.synchronize()

    def _load_model(self):
        try:
            import transformers
        except Exception as exc:
            raise RuntimeError(
                "Failed to import transformers. Activate the Cosmos-Reason2 venv first."
            ) from exc

        model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
        if model_cls is None:
            raise RuntimeError(
                f"transformers {transformers.__version__} does not expose "
                "Qwen3VLForConditionalGeneration"
            )

        base_kwargs = {
            "dtype": torch.float16,
            "device_map": "cuda",
            "attn_implementation": self.args.attn_implementation,
        }
        attempts = [base_kwargs]
        torch_dtype_kwargs = dict(base_kwargs)
        torch_dtype_kwargs["torch_dtype"] = torch_dtype_kwargs.pop("dtype")
        attempts.append(torch_dtype_kwargs)
        mapped_kwargs = dict(base_kwargs)
        mapped_kwargs["device_map"] = {"": f"cuda:{self.args.device}"}
        attempts.append(mapped_kwargs)
        auto_kwargs = dict(base_kwargs)
        auto_kwargs["device_map"] = "auto"
        attempts.append(auto_kwargs)
        last_error = None
        try:
            for kwargs in attempts:
                try:
                    self.model = model_cls.from_pretrained(self.model_name, **kwargs)
                    break
                except TypeError as exc:
                    last_error = exc
                    continue
                except ValueError as exc:
                    last_error = exc
                    continue
            if self.model is None and last_error is not None:
                raise last_error
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                f"CUDA OOM while loading {self.model_name}. The default 8B model is "
                "expected to need about 32GB before experiment buffers; try "
                "--model nvidia/Cosmos-Reason2-2B for smoke testing."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load {self.model_name}: {type(exc).__name__}: {exc}"
            ) from exc

        self.model.eval()
        processor_cls = getattr(transformers, "AutoProcessor", None)
        tokenizer_cls = getattr(transformers, "AutoTokenizer", None)
        try:
            if tokenizer_cls is not None:
                self.tokenizer = tokenizer_cls.from_pretrained(self.model_name)
            elif processor_cls is not None:
                processor = processor_cls.from_pretrained(self.model_name)
                self.tokenizer = getattr(processor, "tokenizer", processor)
            else:
                raise RuntimeError("No AutoTokenizer or AutoProcessor in transformers")
        except Exception as exc:
            raise RuntimeError(f"Failed to load tokenizer/processor for {self.model_name}") from exc

    def _build_inputs(self):
        token_ids = self.tokenizer.encode(" latency", add_special_tokens=False)
        if not token_ids:
            fallback = self.tokenizer.eos_token_id
            token_ids = [fallback if fallback is not None else 0]
        repeats = math.ceil(self.prompt_tokens / len(token_ids))
        prompt = (token_ids * repeats)[: self.prompt_tokens]
        if self.tokenizer.bos_token_id is not None and len(prompt) < self.prompt_tokens:
            prompt = [self.tokenizer.bos_token_id] + prompt
            prompt = prompt[: self.prompt_tokens]

        self.input_ids = torch.tensor([prompt], dtype=torch.long, device="cuda")
        self.attention_mask = torch.ones_like(self.input_ids, dtype=torch.long, device="cuda")
        decode_id = self.tokenizer.eos_token_id
        if decode_id is None:
            decode_id = int(self.input_ids[0, -1])
        self.decode_input_ids = torch.full((1, 1), decode_id, dtype=torch.long, device="cuda")

    def prepare_decode_cache(self):
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            outputs = self.model(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
                use_cache=True,
            )
            self.decode_past_key_values = outputs.past_key_values
            self.decode_attention_mask = torch.ones(
                (1, self.input_ids.shape[1] + 1), dtype=torch.long, device="cuda"
            )
        self.stream.synchronize()

    def prefill_stage(self, nvtx_name=None, profile_markers=False, synchronize=True):
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            with nvtx_range(nvtx_name or "LLM Prefill", profile_markers):
                outputs = self.model(
                    input_ids=self.input_ids,
                    attention_mask=self.attention_mask,
                    use_cache=True,
                )
                self.last_prefill_past_key_values = outputs.past_key_values
        if synchronize:
            self.stream.synchronize()

    def decode_stage(self, nvtx_name=None, profile_markers=False, synchronize=True):
        past = self.decode_past_key_values
        attention_mask = self.decode_attention_mask
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            with nvtx_range(nvtx_name or "LLM Decode", profile_markers):
                for _ in range(self.decode_token_count):
                    outputs = self.model(
                        input_ids=self.decode_input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past,
                        use_cache=True,
                    )
                    past = outputs.past_key_values
                    if self.decode_token_count > 1:
                        extra = torch.ones((1, 1), dtype=torch.long, device="cuda")
                        attention_mask = torch.cat([attention_mask, extra], dim=1)
        if synchronize:
            self.stream.synchronize()


def measure(case_name, fn, iterations, warmup, prepare_fn=None):
    for _ in range(warmup):
        if prepare_fn is not None:
            prepare_fn()
        fn()
    samples = []
    for _ in range(iterations):
        if prepare_fn is not None:
            prepare_fn()
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    result = Measurement(case_name, samples)
    print(
        f"{case_name}: mean={result.mean_ms:.3f} ms "
        f"median={result.median_ms:.3f} ms stdev={result.stdev_ms:.3f} ms"
    )
    return result


def run_overlap(case_name, left_name, left_fn, right_name, right_fn, profile_markers):
    errors = []
    barrier = threading.Barrier(3)

    def worker(name, fn):
        try:
            barrier.wait(timeout=60.0)
            fn(name)
        except BaseException as exc:
            errors.append(exc)
            try:
                barrier.abort()
            except Exception:
                pass

    with nvtx_range(case_name, profile_markers):
        left = threading.Thread(target=worker, args=(left_name, left_fn), name=left_name)
        right = threading.Thread(target=worker, args=(right_name, right_fn), name=right_name)
        left.start()
        right.start()
        try:
            barrier.wait(timeout=60.0)
        finally:
            left.join()
            right.join()
    if errors:
        raise errors[0]


def make_streams_and_masks(args, smctrl):
    vla_start = args.vla_tpc_start
    vla_end = args.vla_tpc_start + args.vla_tpc_count
    llm_start = args.llm_tpc_start
    llm_end = args.llm_tpc_start + args.llm_tpc_count

    if max(vla_start, llm_start) < min(vla_end, llm_end):
        raise ValueError(
            f"TPC ranges overlap: VLA [{vla_start}, {vla_end}) and "
            f"LLM [{llm_start}, {llm_end})"
        )

    info = smctrl.get_tpc_info(args.device)
    driver_version = int(info.get("cuda_driver_version", 0))
    if driver_version >= 12090 and "MASK_OFF" not in os.environ:
        os.environ["MASK_OFF"] = "24"
        print(
            "libsmctrl note: CUDA driver version "
            f"{driver_version} is newer than BulletServe's stream-mask table; "
            "using MASK_OFF=24, matching the CUDA 12.7/12.8 stream-mask offset."
        )
    total_tpcs = int(info["total_tpcs"])
    if vla_start < 0 or llm_start < 0 or vla_end > total_tpcs or llm_end > total_tpcs:
        raise ValueError(
            f"TPC ranges must be within [0, {total_tpcs}); got "
            f"VLA [{vla_start}, {vla_end}) LLM [{llm_start}, {llm_end})"
        )

    vla_stream = torch.cuda.Stream(device=args.device)
    llm_stream = torch.cuda.Stream(device=args.device)

    vla_mask = smctrl.set_stream_mask(stream_ptr(vla_stream), vla_start, vla_end)
    llm_mask = smctrl.set_stream_mask(stream_ptr(llm_stream), llm_start, llm_end)

    if args.validate_stream_masks:
        smctrl.validate_stream_mask(stream_ptr(vla_stream), vla_start, vla_end, True)
        smctrl.validate_stream_mask(stream_ptr(llm_stream), llm_start, llm_end, True)

    return info, vla_stream, llm_stream, vla_mask, llm_mask


def print_partition_info(info, args, vla_stream, llm_stream, vla_mask, llm_mask):
    sms_per_tpc = float(info["sms_per_tpc_exact"])
    vla_sms = args.vla_tpc_count * sms_per_tpc
    llm_sms = args.llm_tpc_count * sms_per_tpc
    print("Partition configuration:")
    print(f"  gpu_name: {info['gpu_name']}")
    print(f"  device_index: {info['device_index']}")
    print(f"  compute_capability: {info['compute_capability']}")
    print(f"  cuda_driver_version: {info.get('cuda_driver_version', 'unknown')}")
    print(f"  total_sms: {info['total_sms']}")
    print(f"  total_tpcs: {info['total_tpcs']}")
    print(f"  sms_per_tpc: {sms_per_tpc:.3f}")
    print(
        f"  vla_tpc_range: [{args.vla_tpc_start}, "
        f"{args.vla_tpc_start + args.vla_tpc_count})"
    )
    print(f"  vla_actual_sms: {vla_sms:.3f}")
    print(f"  vla_stream_object_id: {id(vla_stream)}")
    print(f"  vla_stream_ptr: 0x{stream_ptr(vla_stream):x}")
    print(f"  vla_disabled_mask: {vla_mask['disabled_mask_hex']}")
    print(
        f"  llm_tpc_range: [{args.llm_tpc_start}, "
        f"{args.llm_tpc_start + args.llm_tpc_count})"
    )
    print(f"  llm_actual_sms: {llm_sms:.3f}")
    print(f"  llm_stream_object_id: {id(llm_stream)}")
    print(f"  llm_stream_ptr: 0x{stream_ptr(llm_stream):x}")
    print(f"  llm_disabled_mask: {llm_mask['disabled_mask_hex']}")
    print("  mask_semantics: bit set means disabled TPC")
    print("  disjoint_tpc_ranges: true")


def selected_cases(case):
    if case == "all":
        return [
            "vla-encoder-only",
            "llm-decode-only",
            "overlap-vla-encoder-llm-decode",
            "vla-decoder-only",
            "llm-prefill-only",
            "overlap-vla-decoder-llm-prefill",
        ]
    return [case]


def write_csv(path, results, ratios):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "kind",
                "case",
                "sample_index",
                "sample_ms",
                "mean_ms",
                "median_ms",
                "stdev_ms",
                "metric",
                "value",
            ],
        )
        writer.writeheader()
        for result in results.values():
            for i, sample in enumerate(result.samples_ms):
                writer.writerow(
                    {
                        "kind": "sample",
                        "case": result.case,
                        "sample_index": i,
                        "sample_ms": f"{sample:.6f}",
                        "mean_ms": f"{result.mean_ms:.6f}",
                        "median_ms": f"{result.median_ms:.6f}",
                        "stdev_ms": f"{result.stdev_ms:.6f}",
                        "metric": "",
                        "value": "",
                    }
                )
        for case, metric, value in ratios:
            writer.writerow(
                {
                    "kind": "ratio",
                    "case": case,
                    "sample_index": "",
                    "sample_ms": "",
                    "mean_ms": "",
                    "median_ms": "",
                    "stdev_ms": "",
                    "metric": metric,
                    "value": f"{value:.6f}",
                }
            )


def main():
    parser = argparse.ArgumentParser(
        description="Overlap VLA stages with Cosmos-Reason2 stages on disjoint libsmctrl TPC masks."
    )
    parser.add_argument("--case", choices=CASES, default="all")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--profile-markers", action="store_true")
    parser.add_argument("--validate-stream-masks", action="store_true")
    parser.add_argument("--csv", type=Path, default=None)

    parser.add_argument("--vla-tpc-start", type=int, required=True)
    parser.add_argument("--vla-tpc-count", type=int, required=True)
    parser.add_argument("--llm-tpc-start", type=int, required=True)
    parser.add_argument("--llm-tpc-count", type=int, required=True)

    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--prompt-len", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=63)
    parser.add_argument("--vla-direct", action="store_true")

    parser.add_argument("--model", default="nvidia/Cosmos-Reason2-8B")
    parser.add_argument("--llm-prompt-tokens", type=int, default=512)
    parser.add_argument(
        "--llm-decode-tokens",
        type=int,
        default=DEFAULT_LLM_DECODE_TOKENS,
        help=(
            "Number of autoregressive decode steps per LLM decode stage. "
            f"Default: {DEFAULT_LLM_DECODE_TOKENS}, intentionally longer than "
            "the original one-token decode so the LLM side covers the VLA stage "
            "during overlap profiling. Use 1 for the previous short decode."
        ),
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--pre-measure-sleep", type=float, default=0.0)
    args = parser.parse_args()

    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.vla_tpc_count < 1 or args.llm_tpc_count < 1:
        raise ValueError("TPC counts must be >= 1")
    if args.llm_prompt_tokens < 1:
        raise ValueError("--llm-prompt-tokens must be >= 1")
    if args.llm_decode_tokens < 1:
        raise ValueError("--llm-decode-tokens must be >= 1")

    torch.manual_seed(100)
    torch.cuda.set_device(args.device)
    torch.cuda.init()

    smctrl = load_libsmctrl_extension()
    info, vla_stream, llm_stream, vla_mask, llm_mask = make_streams_and_masks(args, smctrl)
    print_partition_info(info, args, vla_stream, llm_stream, vla_mask, llm_mask)

    cases = selected_cases(args.case)
    needs_vla = any("vla" in case for case in cases)
    needs_llm = any("llm" in case for case in cases)

    vla = VLARunner(args, vla_stream) if needs_vla else None
    llm = LLMRunner(args, llm_stream) if needs_llm else None

    if vla is not None:
        mode = "direct calls" if args.vla_direct else "Pi0Inference graph replay"
        print(
            "VLA configuration: "
            f"mode={mode} num_views={args.num_views} prompt_len={args.prompt_len} "
            f"chunk_size={args.chunk_size} encoder_seq_len={vla.infer.encoder_seq_len} "
            f"decoder_seq_len={vla.infer.decoder_seq_len}"
        )
    if llm is not None:
        print(
            "LLM configuration: "
            f"model={args.model} prompt_tokens={args.llm_prompt_tokens} "
            f"decode_tokens={args.llm_decode_tokens} attn={args.attn_implementation}"
        )

    if args.pre_measure_sleep > 0:
        print(f"Sleeping {args.pre_measure_sleep:.3f} s before measurements")
        time.sleep(args.pre_measure_sleep)

    results = {}

    def run_case(case):
        if case == "vla-encoder-only":
            return measure(
                case,
                lambda: vla.encoder_stage("VLA Encoder Only", args.profile_markers),
                args.iterations,
                args.warmup,
            )
        if case == "vla-decoder-only":
            return measure(
                case,
                lambda: vla.decoder_stage("VLA Decoder Only", args.profile_markers),
                args.iterations,
                args.warmup,
            )
        if case == "llm-prefill-only":
            return measure(
                case,
                lambda: llm.prefill_stage("LLM Prefill Only", args.profile_markers),
                args.iterations,
                args.warmup,
            )
        if case == "llm-decode-only":
            return measure(
                case,
                lambda: llm.decode_stage("LLM Decode Only", args.profile_markers),
                args.iterations,
                args.warmup,
                prepare_fn=llm.prepare_decode_cache,
            )
        if case == "overlap-vla-encoder-llm-decode":
            return measure(
                case,
                lambda: run_overlap(
                    "Overlap VLA Encoder + LLM Decode Total",
                    "Overlap VLA Encoder",
                    lambda name: vla.encoder_stage(name, args.profile_markers),
                    "Overlap LLM Decode",
                    lambda name: llm.decode_stage(name, args.profile_markers),
                    args.profile_markers,
                ),
                args.iterations,
                args.warmup,
                prepare_fn=llm.prepare_decode_cache,
            )
        if case == "overlap-vla-decoder-llm-prefill":
            return measure(
                case,
                lambda: run_overlap(
                    "Overlap VLA Decoder + LLM Prefill Total",
                    "Overlap VLA Decoder",
                    lambda name: vla.decoder_stage(name, args.profile_markers),
                    "Overlap LLM Prefill",
                    lambda name: llm.prefill_stage(name, args.profile_markers),
                    args.profile_markers,
                ),
                args.iterations,
                args.warmup,
            )
        raise ValueError(f"unknown case {case}")

    for case in cases:
        results[case] = run_case(case)

    ratios = []
    pairs = [
        (
            "overlap-vla-encoder-llm-decode",
            "vla-encoder-only",
            "llm-decode-only",
        ),
        (
            "overlap-vla-decoder-llm-prefill",
            "vla-decoder-only",
            "llm-prefill-only",
        ),
    ]
    for overlap, vla_only, llm_only in pairs:
        if overlap in results and vla_only in results and llm_only in results:
            concurrent = results[overlap].median_ms
            vla_iso = results[vla_only].median_ms
            llm_iso = results[llm_only].median_ms
            ratio_values = [
                ("concurrent_over_max_isolated", ratio_or_nan(concurrent, max(vla_iso, llm_iso))),
                ("concurrent_vs_vla_only", ratio_or_nan(concurrent, vla_iso)),
                ("concurrent_vs_llm_only", ratio_or_nan(concurrent, llm_iso)),
            ]
            print(f"Derived overlap ratios for {overlap}:")
            for metric, value in ratio_values:
                print(f"  {metric}: {value:.3f}x")
                ratios.append((overlap, metric, value))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.csv or (ROOT / "results" / f"vla_llm_overlap_{timestamp}.csv")
    write_csv(csv_path, results, ratios)
    print(f"CSV written to {csv_path}")


if __name__ == "__main__":
    main()
