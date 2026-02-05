#!/usr/bin/env python3
"""
Quick profiling script for π0 model with JAX profiler.
Shows fine-grained profiling inside JIT-compiled code.
"""
import argparse
from click import prompt
import jax
import jax.numpy as jnp
import jax.profiler

# Configure JAX for profiling
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_command_buffer='  # Better profiling

print("Loading π0 model...")
from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import get_config
from openpi.models.tokenizer import PaligemmaTokenizer
import numpy as np
import dataclasses
import torch

parser = argparse.ArgumentParser(description="Profile Pi0.5 (JAX)")
parser.add_argument("--is_GB10_device", type=bool, default=False, help="Use GB10 device")
args = parser.parse_args()


# Load policy using the correct API (same as serve_policy.py)
config_name = "pi05_libero"  # Just the config name, not a file path
checkpoint_dir = "gs://openpi-assets/checkpoints/pi05_libero"
policies = {}

# Initialize tokenizer for token counting

# Create dummy observation for profiling (must be dict format, not Observation object)
print("\nCreating dummy observation...")
# Policy.infer expects a dict with these keys (see examples/libero/main.py)
base_observation = {
    "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/state": np.zeros(7, dtype=np.float32),
    "prompt": "",  # Will be set per prompt
}

# Define prompts of varying lengths to vary input token count
base_prompt = "pick up the red object from the table and place it in the basket carefully. "
prompts = [base_prompt * 1]  # Single prompt for testing

print("\n" + "="*60)
print("Starting profiling...")
print("="*60)

# Profile with JAX profiler (saves trace files locally)
profile_dir = "/tmp/pi0-libero-profile"
print(f"\nProfile will be saved to: {profile_dir}")
# jax.profiler.start_trace(profile_dir)  # Commented out for faster testing

num_inferences = 3  # Reduced for faster testing
print(f"\nRunning {num_inferences} inference steps per prompt...")

# max_len_lists = [128, 512, 1024, 2048, 4096]
max_len_lists = [128]
for max_len in max_len_lists:
    tokenizer = PaligemmaTokenizer(max_len=max_len)
    print("Tokenizer loaded.")
    # Build a prompt that will result in exactly max_len text tokens
    # Start with a base prompt and repeat until we reach or exceed max_len tokens, then truncate
    repeated_prompt = base_prompt
    tokens, mask = tokenizer.tokenize(repeated_prompt)
    while np.sum(mask) < max_len:
        repeated_prompt += base_prompt
        tokens, mask = tokenizer.tokenize(repeated_prompt)
    # Now truncate the prompt so that the number of tokens is exactly max_len
    # Find the minimal substring that gives exactly max_len tokens
    left, right = 0, len(repeated_prompt)
    best_prompt = repeated_prompt
    while left < right:
        mid = (left + right) // 2
        test_prompt = repeated_prompt[:mid]
        tokens, mask = tokenizer.tokenize(test_prompt)
        if np.sum(mask) < max_len:
            left = mid + 1
        else:
            best_prompt = test_prompt
            right = mid
    # Final prompt with exactly max_len tokens
    tokens, mask = tokenizer.tokenize(best_prompt)
    num_text_tokens = np.sum(mask)
    assert num_text_tokens == max_len, f"Prompt tokenization failed to reach max_len={max_len}, got {num_text_tokens}"
    observation = base_observation.copy()
    observation["prompt"] = best_prompt
    if max_len not in policies:
        config = get_config(config_name)
        config = dataclasses.replace(config, model=dataclasses.replace(config.model, max_token_len=max_len))
        print(f"Creating policy with max_token_len={max_len}")
        policies[max_len] = create_trained_policy(config, checkpoint_dir, is_GB10_device=args.is_GB10_device)
    policy = policies[max_len]
    # Assuming 3 images, each with 256 tokens (224/14)^2 for patch size 14
    num_image_tokens = 3 * 256
    total_input_tokens = num_image_tokens + num_text_tokens
    print(f"\n--- Profiling with prompt: '{best_prompt[:50]}...' (length: {len(best_prompt)} chars, text tokens: {num_text_tokens}, total input tokens: {total_input_tokens}, max_len: {max_len}) ---")
    try:
        torch.cuda.nvtx.range_push(f"inference_token_length_{num_text_tokens}")
        nvtx_enabled = True
    except RuntimeError:
        nvtx_enabled = False
    for i in range(num_inferences):
        print(f"  Inference {i+1}/{num_inferences}")
        result = policy.infer(observation)
        # Ensure GPU work completes before moving to next iteration
        if isinstance(result["actions"], jnp.ndarray):
            result["actions"].block_until_ready()
    if nvtx_enabled:
        torch.cuda.nvtx.range_pop()



# jax.profiler.stop_trace()  # Commented out