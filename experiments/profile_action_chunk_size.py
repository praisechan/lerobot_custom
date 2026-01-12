#!/usr/bin/env python
"""
Simple latency profiling for PI05 with different action chunk sizes.
Keeps both VLM (PaliGemma) and action expert architecture constant.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n lerobot python profile_action_chunk_size.py
"""

import torch
import numpy as np
import time
import gc
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.configs.types import FeatureType, PolicyFeature
import torch.cuda.nvtx as nvtx


def create_model(chunk_size, device="cuda"):
    """Create PI05 model with specified chunk_size."""
    config = PI05Config(
        action_expert_variant="gemma_300m",  # Keep constant
        paligemma_variant="gemma_2b",  # Keep constant
        max_action_dim=32,
        chunk_size=chunk_size,  # This is what we vary
        n_action_steps=10,
        num_inference_steps=10,
        device=device,
        dtype="float32",
        
        input_features={
            "observation.images.agentview_image": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 224, 224),
            ),
            "observation.images.eye_in_hand_image": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 224, 224),
            ),
            "observation.language.instruction": PolicyFeature(
                type=FeatureType.LANGUAGE,
                shape=(),
            ),
        },
        output_features={
            "action": PolicyFeature(
                type=FeatureType.ACTION,
                shape=(7,),
            ),
        },
    )
    
    model = PI05Policy(config)
    model.eval()
    return model


def create_batch(device="cuda"):
    """Create dummy input batch."""
    return {
        "observation.images.agentview_image": torch.randn(1, 3, 224, 224, device=device),
        "observation.images.eye_in_hand_image": torch.randn(1, 3, 224, 224, device=device),
        "observation.language.tokens": torch.randint(0, 1000, (1, 50), device=device),
        "observation.language.attention_mask": torch.ones(1, 50, device=device, dtype=torch.bool),
    }


def profile_chunk_size(chunk_size, warmup=3, iterations=10, device="cuda"):
    """Profile a single chunk size configuration."""
    print(f"\nTesting: chunk_size={chunk_size}")
    print("-" * 60)
    
    # Print GPU memory before
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1024**3
        print(f"  GPU memory before: {mem_before:.2f} GB")
    
    try:
        # Create model
        print(f"  Creating model (chunk_size={chunk_size})...", end='', flush=True)
        model = create_model(chunk_size, device)
        print(" ✓")
        
        # Create batch
        batch = create_batch(device)
        
        nvtx.range_push(f"warmup_chunk{chunk_size}")
        # Warmup
        print(f"  Warmup: ", end='', flush=True)
        for i in range(warmup):
            with torch.no_grad():
                _ = model.predict_action_chunk(batch)
            print(f"{i+1}/{warmup} ", end='', flush=True)
        print("✓")
        nvtx.range_pop()
        
        # Measure
        print(f"  Measuring: ", end='', flush=True)
        
        nvtx.range_push(f"measure_chunk{chunk_size}")
        for i in range(iterations):
            with torch.no_grad():
                _ = model.predict_action_chunk(batch)
            print(f"{i+1}/{iterations} ", end='', flush=True)
        print("✓")
        nvtx.range_pop()
        
        # Cleanup - aggressive memory freeing
        print(f"  Cleaning up...", end='', flush=True)
        # Move model to CPU first to release GPU memory
        model.cpu()
        del model, batch
        # Force garbage collection
        gc.collect()
        # Clear CUDA cache
        torch.cuda.empty_cache()
        # Synchronize to ensure all operations are complete
        torch.cuda.synchronize()
        
        # Print GPU memory after cleanup
        if torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated() / 1024**3
            print(f" ✓ (GPU memory after: {mem_after:.2f} GB)")
        else:
            print(" ✓")
        
        return
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        # Cleanup on error
        if 'model' in locals():
            model.cpu()
            del model
        if 'batch' in locals():
            del batch
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return None


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    print("="*70)
    print("PI05 Action Chunk Size Latency Profiling")
    print("="*70)
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"\nAction Expert: gemma_300m (constant)")
    print(f"VLM: gemma_2b (constant)")
    
    # Test different chunk sizes
    chunk_sizes = [
        10,   # Small
        50,   # Default/baseline
        200,        
        1000,  # Extra Large
        
    ]
    
    print(f"\nTesting {len(chunk_sizes)} chunk sizes with 3 warmup + 10 measurement iterations each")
    print("="*70)
    
    for i, chunk_size in enumerate(chunk_sizes, 1):
        nvtx.range_push(f"Profile_chunk{chunk_size}")
        print(f"\n[{i}/{len(chunk_sizes)}] Progress: {'█' * i}{'░' * (len(chunk_sizes)-i)} {i*100//len(chunk_sizes)}%")
        profile_chunk_size(chunk_size, warmup=3, iterations=10, device=device)
        nvtx.range_pop()
    
    print("\n" + "="*70)
    print("✓ Profiling complete!")
    print("="*70)
    print("\nTo analyze results, check the nsys report:")
    print("  nsys profile -o chunk_size_profile python profile_action_chunk_size.py")
    print("="*70)


if __name__ == "__main__":
    main()
