#!/usr/bin/env python3
"""
Profile Pi0.5 with varying action chunk sizes using JAX.

Run:
    uv run examples/libero/profile_pi05_action_chunk_size.py

This script tests how inference latency scales with action chunk size
(action_horizon) from 50 to 1000, keeping the model architecture fixed.
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.pi0_config import Pi0Config
from openpi.policies.policy import Policy


IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def make_dummy_inputs(action_dim: int, prompt_len: int = 0):
    img = np.zeros((224, 224, 3), dtype=np.float32)
    images = {k: img for k in IMAGE_KEYS}
    image_mask = {k: np.ones((), dtype=bool) for k in IMAGE_KEYS}

    inputs = {
        "image": images,
        "image_mask": image_mask,
        "state": np.zeros((action_dim,), dtype=np.float32),
    }
    if prompt_len > 0:
        inputs["tokenized_prompt"] = np.zeros((prompt_len,), dtype=np.int32)
        inputs["tokenized_prompt_mask"] = np.ones((prompt_len,), dtype=bool)
    return inputs


def build_policy(action_horizon: int, seed: int) -> tuple[Policy, Pi0Config]:
    config = Pi0Config(
        pi05=True,
        action_horizon=action_horizon,
        discrete_state_input=False,  # Matches pi05_libero inference setup
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
    )
    rng = jax.random.PRNGKey(seed)
    model = config.create(rng)
    policy = Policy(model, rng=rng, transforms=(), output_transforms=())
    return policy, config


def profile_chunk_size(action_horizon: int, warmup: int, iterations: int, seed: int):
    print(f"\nAction Horizon: {action_horizon}")

    policy, model_config = build_policy(action_horizon, seed)
    inputs = make_dummy_inputs(action_dim=model_config.action_dim, prompt_len=32)

    print(f"  Warmup ({warmup})", end=": ")
    for i in range(warmup):
        _ = policy.infer(inputs)
        print(f"{i + 1}/{warmup} ", end="", flush=True)
    print("✓")

    latencies_ms: list[float] = []
    print(f"  Measure ({iterations})", end=": ")
    for i in range(iterations):
        start = time.perf_counter()
        _ = policy.infer(inputs)
        jax.block_until_ready(jnp.array(0.0))  # Ensure all work finished
        latencies_ms.append((time.perf_counter() - start) * 1000)
        print(f"{i + 1}/{iterations} ", end="", flush=True)
    print("✓")

    mean_ms = float(np.mean(latencies_ms))
    p50_ms = float(np.percentile(latencies_ms, 50))
    p90_ms = float(np.percentile(latencies_ms, 90))

    print(f"  Stats: mean={mean_ms:.2f} ms, p50={p50_ms:.2f} ms, p90={p90_ms:.2f} ms")
    return {
        "action_horizon": action_horizon,
        "mean_ms": mean_ms,
        "p50_ms": p50_ms,
        "p90_ms": p90_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile Pi0.5 action chunk size variants (JAX)")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs")
    parser.add_argument("--iters", type=int, default=5, help="Number of timed iterations")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    args = parser.parse_args()

    chunk_sizes = [50, 100, 200, 500, 1000]

    print("=" * 70)
    print("Pi0.5 Action Chunk Size Profiling (JAX)")
    print("=" * 70)
    print(f"Devices: {[d.platform for d in jax.devices()]}")
    print(f"Warmup={args.warmup}, iters={args.iters}")
    print(f"Model: gemma_2b + gemma_300m action expert\n")

    results = []
    for idx, chunk_size in enumerate(chunk_sizes, 1):
        print(f"[{idx}/{len(chunk_sizes)}] Chunk size: {chunk_size}")
        results.append(profile_chunk_size(chunk_size, args.warmup, args.iters, args.seed))

    print("\n" + "=" * 70)
    print(f"Summary over {args.iters} runs (ms)")
    print("=" * 70)
    print(f"{'Action Horizon':<15} {'mean':>10} {'p50':>10} {'p90':>10}")
    for r in results:
        print(
            f"{r['action_horizon']:<15} {r['mean_ms']:>10.2f} "
            f"{r['p50_ms']:>10.2f} {r['p90_ms']:>10.2f}"
        )


if __name__ == "__main__":
    main()
