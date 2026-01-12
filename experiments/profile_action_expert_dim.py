#!/usr/bin/env python
"""
Simple latency profiling for PI05 action expert with different width dimensions.
Only modifies the action expert, keeps VLM (PaliGemma) constant.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n lerobot python profile_action_expert_simple.py
"""

import torch
import numpy as np
import time
import gc
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, get_gemma_config, GemmaConfig
from lerobot.configs.types import FeatureType, PolicyFeature
import torch.cuda.nvtx as nvtx

# Extend get_gemma_config with test variants
def get_custom_gemma_config(variant: str) -> GemmaConfig:
    if variant == "gemma_300m":
        return GemmaConfig(width=1024, depth=18, mlp_dim=4096,
                          num_heads=8, num_kv_heads=1, head_dim=256)
    elif variant == "gemma_2b":
        return GemmaConfig(width=2048, depth=18, mlp_dim=16_384,
                          num_heads=8, num_kv_heads=1, head_dim=256)
    # Test variants - only change width and mlp_dim
    elif variant == "gemma_512":
        return GemmaConfig(width=512, depth=18, mlp_dim=2048,
                          num_heads=4, num_kv_heads=1, head_dim=256)
    elif variant == "gemma_768":
        return GemmaConfig(width=768, depth=18, mlp_dim=3072,
                          num_heads=6, num_kv_heads=1, head_dim=256)
    elif variant == "gemma_1536":
        return GemmaConfig(width=1536, depth=18, mlp_dim=6144,
                          num_heads=8, num_kv_heads=1, head_dim=256)
    elif variant == "gemma_2048":
        return GemmaConfig(width=2048, depth=18, mlp_dim=8192,
                          num_heads=8, num_kv_heads=1, head_dim=256)
    else:
        raise ValueError(f"Unknown variant: {variant}")


import lerobot.policies.pi05.modeling_pi05 as pi05_module
pi05_module.get_gemma_config = get_custom_gemma_config


def create_model(expert_variant, device="cuda"):
    config = PI05Config(
        action_expert_variant=expert_variant,
        paligemma_variant="gemma_2b",  # Keep VLM constant
        max_action_dim=32,
        chunk_size=50,
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
    return {
        "observation.images.agentview_image": torch.randn(1, 3, 224, 224, device=device),
        "observation.images.eye_in_hand_image": torch.randn(1, 3, 224, 224, device=device),
        "observation.language.tokens": torch.randint(0, 1000, (1, 50), device=device),
        "observation.language.attention_mask": torch.ones(1, 50, device=device, dtype=torch.bool),
    }


def profile_variant(variant, warmup=3, iterations=10, device="cuda"):
    """Profile a single variant."""
    print(f"\nTesting: {variant}")
    print("-" * 60)
    
    # Print GPU memory before
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1024**3
        print(f"  GPU memory before: {mem_before:.2f} GB")
    
    try:
        # Get config
        expert_config = get_custom_gemma_config(variant)
        print(f"  Expert: width={expert_config.width}, mlp_dim={expert_config.mlp_dim}")
        
        # Create model
        print(f"  Creating model...", end='', flush=True)
        model = create_model(variant, device)
        print(" ✓")
        
        # Create batch
        batch = create_batch(device)
        
        nvtx.range_push(f"warmup")
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
        
        nvtx.range_push(f"measure")
        for i in range(iterations):
            with torch.no_grad():
                _ = model.predict_action_chunk(batch)
            print(f"{i+1}/{iterations} ", end='', flush=True)
        print("✓")
        nvtx.range_pop()
        
        # # Print results
        # vlm_mean = stats['vlm_prefix']['mean'] if stats['vlm_prefix'] else 0
        # expert_mean = stats['action_expert']['mean'] if stats['action_expert'] else 0
        # print(f"  ✓ VLM: {vlm_mean:.2f} ms, Expert: {expert_mean:.2f} ms")
        
        # result = {
        #     "variant": variant,
        #     "width": expert_config.width,
        #     "mlp_dim": expert_config.mlp_dim,
        #     "vlm_ms": vlm_mean,
        #     "expert_ms": expert_mean,
        #     "total_ms": vlm_mean + expert_mean,
        #     "stats": stats,
        # }
        
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
    
    # Enable CUDA optimizations (same as lerobot_eval.py)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    print("="*70)
    print("PI05 Action Expert Latency Profiling")
    print("="*70)
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
    
    # Test only valid variants
    variants = [
        "gemma_512",
        # "gemma_768",
        "gemma_300m",  # baseline
        # "gemma_1536",
        "gemma_2048",
    ]
    # variants = [
    #     "gemma_512",
    #     "gemma_768",
    #     "gemma_300m",  # baseline
    #     "gemma_1536",
    #     "gemma_2048",
    # ]
    
    print(f"\nTesting {len(variants)} variants with 3 warmup + 10 measurement iterations each")
    print("="*70)
    
    results = []
    for i, variant in enumerate(variants, 1):
        nvtx.range_push(f"Profile_{variant}")
        print(f"\n[{i}/{len(variants)}] Progress: {'█' * i}{'░' * (len(variants)-i)} {i*100//len(variants)}%")
        profile_variant(variant, warmup=3, iterations=10, device=device)
        nvtx.range_pop()
    
    # Print summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"{'Variant':<15} {'Width':<7} {'MLP':<7} {'VLM (ms)':<10} {'Expert (ms)':<12} {'Total (ms)'}")
    print("-"*70)
    
    for r in results:
        print(f"{r['variant']:<15} {r['width']:<7} {r['mlp_dim']:<7} {r['vlm_ms']:>8.2f}   {r['expert_ms']:>10.2f}   {r['total_ms']:>8.2f}")
    
    print("-"*70)
    
    # Baseline comparison
    baseline = next((r for r in results if r['variant'] == 'gemma_300m'), None)
    if baseline:
        print(f"\nBaseline (gemma_300m): Expert={baseline['expert_ms']:.2f}ms, Total={baseline['total_ms']:.2f}ms")
        print("\nSpeedup vs Baseline:")
        for r in results:
            if r['variant'] != 'gemma_300m':
                speedup = baseline['expert_ms'] / r['expert_ms']
                print(f"  {r['variant']:<15} Expert: {speedup:>5.2f}x ({r['expert_ms']:>6.2f}ms)")
        
    print("\n" + "="*70)
    print("✓ Results saved to action_expert_latency.csv")
    print("="*70)


if __name__ == "__main__":
    main()
