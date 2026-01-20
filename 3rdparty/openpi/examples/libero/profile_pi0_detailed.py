#!/usr/bin/env python3
"""
Quick profiling script for π0 model with JAX profiler.
Shows fine-grained profiling inside JIT-compiled code.
"""
import jax
import jax.numpy as jnp
import jax.profiler

# Configure JAX for profiling
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_command_buffer='  # Better profiling

print("Loading π0 model...")
from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import get_config
import numpy as np

# Load policy using the correct API (same as serve_policy.py)
config = get_config("pi05_libero")  # Just the config name, not a file path
checkpoint_dir = "gs://openpi-assets/checkpoints/pi05_libero"
policy = create_trained_policy(config, checkpoint_dir)
print(f"Model loaded: {type(policy._model).__name__}")

# Create dummy observation for profiling (must be dict format, not Observation object)
print("\nCreating dummy observation...")
# Policy.infer expects a dict with these keys (see examples/libero/main.py)
observation = {
    "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/state": np.zeros(7, dtype=np.float32),
    "prompt": "pick up the object",
}

# Warmup to compile everything
print("Warming up (compiling JIT functions)...")
for i in range(3):
    print(f"  Warmup {i+1}/3")
    result = policy.infer(observation)
    # result is a dict with "actions" key
    if isinstance(result["actions"], jnp.ndarray):
        result["actions"].block_until_ready()
    print(f"    Action shape: {result['actions'].shape}")

print("\n" + "="*60)
print("Starting profiling...")
print("="*60)

# Profile with JAX profiler (saves trace files locally)
profile_dir = "/tmp/pi0-libero-profile"
print(f"\nProfile will be saved to: {profile_dir}")

num_inferences = 5
print(f"\nRunning {num_inferences} inference steps...")

# Don't use create_perfetto_link=True on remote machines - it requires local HTTP access
with jax.profiler.trace(profile_dir, create_perfetto_link=True):
    for i in range(num_inferences):
        print(f"  Inference {i+1}/{num_inferences}")
        result = policy.infer(observation)
        # Ensure GPU work completes before moving to next iteration
        if isinstance(result["actions"], jnp.ndarray):
            result["actions"].block_until_ready()

print("\n" + "="*60)
print("Profiling complete!")
print("="*60)
print(f"\nProfile saved to: {profile_dir}")

print("\n📊 How to view the trace:")
print("\n1. Using XProf (Recommended for remote machines):")
print(f"   xprof --port 8791 {profile_dir}")
print("   Then open http://localhost:8791/ in your browser")
print("   Select your run from the dropdown, then choose 'trace_viewer' from Tools")

print("\n2. Using Perfetto (requires local file or SSH tunnel):")
print("   a. Download trace file to your local machine:")
print(f"      scp -r atlas2:{profile_dir} .")
print("   b. Go to https://ui.perfetto.dev/")
print("   c. Click 'Open trace file' and select the .pb.gz or .json.gz file")

print("\n3. For SSH tunnel (if you want automatic Perfetto link):")
print("   ssh -L 9001:127.0.0.1:9001 atlas2")
print("   Then re-run this script with create_perfetto_link=True")

print("\n🔍 What to look for in the trace:")
print("  - sample_actions (outer)")
print("    ├─ preprocess_observation")
print("    ├─ embed_prefix")
print("    ├─ compute_kv_cache")
print("    └─ diffusion_loop")
print("        ├─ step_embed_suffix (inside JIT!)")
print("        ├─ step_compute_masks (inside JIT!)")
print("        ├─ step_compute_positions (inside JIT!)")
print("        ├─ step_llm_forward (inside JIT! - expected bottleneck)")
print("        └─ step_action_proj (inside JIT!)")
print("\nThe step_* markers show detailed breakdown of each diffusion step!")
