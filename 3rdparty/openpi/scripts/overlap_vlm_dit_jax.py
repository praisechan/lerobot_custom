"""Benchmark overlap of VLM(prefix/KV-cache) and DiT-like denoising for the JAX PI0/PI05 model on a single GPU.

This benchmark is for the case where you:
- want to use the default OpenPI checkpoints (JAX/orbax, e.g. pi05_libero)
- do NOT want to convert checkpoints to PyTorch
- still want to quantify *overlap* between the VLM(prefix) phase and the denoising loop on one GPU

Key point (same as the PyTorch benchmark):
- For the same sample, denoise depends on the prefix KV-cache. So you cannot overlap prefix and denoise of the *same*
  sample.
- The only safe overlap is **pipelining across samples**: while denoising sample i, enqueue prefix work for sample i+1.

Implementation strategy (JAX):
- Use separate jitted functions for prefix-KV and denoise.
- Use asynchronous dispatch and explicit completion via `.block_until_ready()`.
- Use different samples so there is no data dependency.

This measures overlap at the *device dispatch / execution* level. For deeper kernel-level overlap analysis, profile with
Nsight Systems (CUDA timeline) while running this script.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi.models import model as _model
from openpi.shared import download as _download
from openpi.training import config as _config


def _timeit(fn, *, iters: int) -> float:
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def _fmt_rate(seconds_per_iter: float) -> str:
    return f"{seconds_per_iter * 1e3:.3f} ms/iter ({1.0 / seconds_per_iter:.2f} it/s)"


def _make_fake_obs(cfg: Any, batch_size: int) -> _model.Observation[jax.Array]:
    """Create a synthetic observation matching the config's input spec."""
    obs_spec, _ = cfg.model.inputs_spec(batch_size=batch_size)  # type: ignore[attr-defined]

    def make_from_spec(spec: jax.ShapeDtypeStruct):
        # Remove batch dimension? No: inputs_spec already includes batch dim.
        if spec.dtype == jnp.float32:
            return jax.random.uniform(jax.random.key(0), shape=spec.shape, minval=-1.0, maxval=1.0, dtype=jnp.float32)
        if spec.dtype == jnp.int32:
            return jax.random.randint(jax.random.key(0), shape=spec.shape, minval=0, maxval=2048, dtype=jnp.int32)
        if spec.dtype == jnp.bool_:
            return jnp.ones(spec.shape, dtype=jnp.bool_)
        return jnp.zeros(spec.shape, dtype=spec.dtype)

    images = {k: make_from_spec(spec) for k, spec in obs_spec.images.items()}
    image_masks = {k: jnp.ones(spec.shape[:-3], dtype=jnp.bool_) for k, spec in obs_spec.images.items()}
    state = make_from_spec(obs_spec.state)

    tokenized_prompt = None
    tokenized_prompt_mask = None
    if obs_spec.tokenized_prompt is not None:
        tokenized_prompt = make_from_spec(obs_spec.tokenized_prompt)
        tokenized_prompt_mask = jnp.ones(obs_spec.tokenized_prompt.shape, dtype=jnp.bool_)

    return _model.Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        token_ar_mask=None,
        token_loss_mask=None,
    )


@dataclasses.dataclass
class Args:
    config: str = "pi05_libero"
    checkpoint_dir: str = "gs://openpi-assets/checkpoints/pi05_libero"

    batch_size: int = 1
    num_steps: int = 10
    iters: int = 50
    warmup: int = 10


def main(args: Args) -> None:
    # Ensure we are on GPU (single device). If you have multiple GPUs visible, set CUDA_VISIBLE_DEVICES=0.
    backend = jax.default_backend()
    devices = jax.devices()
    if backend != "gpu":
        raise RuntimeError(f"Expected JAX GPU backend for this benchmark, got backend={backend}, devices={devices}")

    train_cfg = _config.get_config(args.config)
    ckpt_dir = _download.maybe_download(args.checkpoint_dir)
    params = _model.restore_params(ckpt_dir / "params", dtype=jnp.bfloat16)
    model = train_cfg.model.load(params)

    # Build obs list: independent samples for pipelining.
    obs_list = [_make_fake_obs(train_cfg, args.batch_size) for _ in range(args.iters + args.warmup)]

    # Jitted building blocks.
    # Note: kv_cache is a pytree; we keep it explicit to separate prefix and denoise stages.
    def prefix_fn(observation: _model.Observation[jax.Array]):
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        from openpi.models.pi0 import make_attn_mask  # noqa: PLC0415

        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        return kv_cache, prefix_mask, observation

    def denoise_fn(kv_cache, prefix_mask, observation: _model.Observation[jax.Array], *, rng: jax.Array):
        # Minimal reproduction of Pi0.sample_actions denoise loop.
        from openpi.models.pi0 import make_attn_mask  # noqa: PLC0415
        import einops  # noqa: PLC0415

        dt = -1.0 / args.num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, model.action_horizon, model.action_dim))

        def step(carry):
            x_t, time_t = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
                observation, x_t, jnp.broadcast_to(time_t, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (_prefix_out, suffix_out), _ = model.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])
            return x_t + dt * v_t, time_t + dt

        def cond(carry):
            _x_t, time_t = carry
            return time_t >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    prefix_jit = jax.jit(prefix_fn)
    denoise_jit = jax.jit(denoise_fn, static_argnames=())

    # Warmup: compile.
    rng = jax.random.key(0)
    kv_cache, prefix_mask, obs0 = prefix_jit(obs_list[0])
    x0 = denoise_jit(kv_cache, prefix_mask, obs0, rng=rng)
    x0.block_until_ready()

    # Helper runs.
    def run_sequential(obs_seq: list[_model.Observation[jax.Array]]):
        nonlocal rng
        for obs in obs_seq:
            kv_cache, prefix_mask, obs_pp = prefix_jit(obs)
            rng, step_rng = jax.random.split(rng)
            out = denoise_jit(kv_cache, prefix_mask, obs_pp, rng=step_rng)
            out.block_until_ready()

    def run_prefix_only(obs_seq: list[_model.Observation[jax.Array]]):
        for obs in obs_seq:
            kv_cache, prefix_mask, obs_pp = prefix_jit(obs)
            # Ensure device completion for timing.
            jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, kv_cache)
            prefix_mask.block_until_ready()
            obs_pp.state.block_until_ready()

    def run_denoise_only(obs_seq: list[_model.Observation[jax.Array]]):
        """Measure denoise loop in isolation but still using real kv_cache from prefix."""
        nonlocal rng
        for obs in obs_seq:
            kv_cache, prefix_mask, obs_pp = prefix_jit(obs)
            rng, step_rng = jax.random.split(rng)
            out = denoise_jit(kv_cache, prefix_mask, obs_pp, rng=step_rng)
            out.block_until_ready()

    def run_pipelined(obs_seq: list[_model.Observation[jax.Array]]):
        """Pipeline across samples: dispatch denoise(i) and prefix(i+1) before blocking."""
        nonlocal rng

        # Prime the pipeline: compute prefix for first sample.
        kv_cache, prefix_mask, obs_pp = prefix_jit(obs_seq[0])

        pending_denoise = None
        for i in range(len(obs_seq)):
            # Dispatch denoise(i)
            rng, step_rng = jax.random.split(rng)
            pending_denoise = denoise_jit(kv_cache, prefix_mask, obs_pp, rng=step_rng)

            # Dispatch prefix(i+1) concurrently (different sample)
            if i + 1 < len(obs_seq):
                kv_cache, prefix_mask, obs_pp = prefix_jit(obs_seq[i + 1])

            # Ensure denoise(i) is done before moving to next iteration (to avoid unbounded queueing).
            pending_denoise.block_until_ready()

    # Benchmark.
    obs_for_bench = obs_list[args.warmup :]
    run_sequential(obs_list[: args.warmup])

    prefix_s = _timeit(lambda: run_prefix_only(obs_for_bench), iters=args.iters)
    denoise_s = _timeit(lambda: run_denoise_only(obs_for_bench), iters=args.iters)
    seq_s = _timeit(lambda: run_sequential(obs_for_bench), iters=args.iters)
    ovl_s = _timeit(lambda: run_pipelined(obs_for_bench), iters=args.iters)

    print("=== overlap_vlm_dit_jax results ===")
    print(f"config: {args.config}")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"backend: {backend}")
    print(f"devices: {devices}")
    print(f"batch_size: {args.batch_size}")
    print(f"num_steps: {args.num_steps}")
    print(f"iters: {args.iters} (warmup {args.warmup})")
    print(f"prefix_only:  {_fmt_rate(prefix_s)}")
    print(f"denoise_only: {_fmt_rate(denoise_s)}")
    print(f"sequential:   {_fmt_rate(seq_s)}")
    print(f"pipelined:    {_fmt_rate(ovl_s)}")
    print(f"speedup:      {(seq_s / ovl_s):.3f}x")


if __name__ == "__main__":
    main(tyro.cli(Args))
