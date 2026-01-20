#!/usr/bin/env python3
"""
Profile Pi0.5 action expert width variants using JAX.

Run:
    uv run examples/libero/profile_pi05_action_expert_dim.py

This script mirrors the PyTorch PI05 action-expert sweep but uses the
OpenPI JAX policy. It monkey patches Gemma configs with a few test widths
and times `Policy.infer` over random inputs (no checkpoint load).
"""

import argparse
import time
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import gemma as gemma_module
from openpi.models.pi0_config import Pi0Config
from openpi.policies.policy import Policy

# Extra Gemma variants for the action expert sweep.
# Note: num_heads=8 must match the paligemma backbone for multi-expert attention to work
_CUSTOM_VARIANTS = {
    "gemma_512": dict(width=512, depth=18, mlp_dim=2048, num_heads=8, num_kv_heads=1, head_dim=256),
    "gemma_768": dict(width=768, depth=18, mlp_dim=3072, num_heads=8, num_kv_heads=1, head_dim=256),
    "gemma_1536": dict(width=1536, depth=18, mlp_dim=6144, num_heads=8, num_kv_heads=1, head_dim=256),
    "gemma_2048": dict(width=2048, depth=18, mlp_dim=8192, num_heads=8, num_kv_heads=1, head_dim=256),
}

_original_get_config = gemma_module.get_config


def get_custom_gemma_config(variant: str) -> gemma_module.Config:
    if variant in _CUSTOM_VARIANTS:
        return gemma_module.Config(**_CUSTOM_VARIANTS[variant])
    return _original_get_config(variant)


# Monkey patch Gemma to add the test variants.
gemma_module.get_config = get_custom_gemma_config


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


def build_policy(variant: str, seed: int) -> tuple[Policy, Pi0Config]:
    config = Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,  # Matches pi05_libero inference setup
        paligemma_variant="gemma_2b",
        action_expert_variant=variant,
    )
    rng = jax.random.PRNGKey(seed)
    model = config.create(rng)
    policy = Policy(model, rng=rng, transforms=(), output_transforms=())
    return policy, config


def profile_variant(variant: str, warmup: int, iterations: int, seed: int):
    cfg = gemma_module.get_config(variant)
    print(f"\nVariant {variant}: width={cfg.width}, mlp_dim={cfg.mlp_dim}")

    policy, model_config = build_policy(variant, seed)
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
        "variant": variant,
        "config": asdict(cfg),
        "mean_ms": mean_ms,
        "p50_ms": p50_ms,
        "p90_ms": p90_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile Pi0.5 action expert width variants (JAX)")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs")
    parser.add_argument("--iters", type=int, default=5, help="Number of timed iterations")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    args = parser.parse_args()

    variants = [
        "gemma_512",
        "gemma_300m",  # baseline action expert
        "gemma_2048",
    ]

    print("=" * 70)
    print("Pi0.5 Action Expert Profiling (JAX)")
    print("=" * 70)
    print(f"Devices: {[d.platform for d in jax.devices()]}")
    print(f"Warmup={args.warmup}, iters={args.iters}\n")

    results = []
    for idx, variant in enumerate(variants, 1):
        print(f"[{idx}/{len(variants)}] {variant}")
        results.append(profile_variant(variant, args.warmup, args.iters, args.seed))

    print("\n" + "=" * 70)
    print(f"Summary over {args.iters} runs (ms)")
    print("=" * 70)
    print(f"{'Variant':<12} {'Width':>6} {'MLP':>8} {'mean':>10} {'p50':>10} {'p90':>10}")
    for r in results:
        print(
            f"{r['variant']:<12} {r['config']['width']:>6} {r['config']['mlp_dim']:>8} "
            f"{r['mean_ms']:>10.2f} {r['p50_ms']:>10.2f} {r['p90_ms']:>10.2f}"
        )


if __name__ == "__main__":
    main()
