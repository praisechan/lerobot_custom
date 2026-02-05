"""
Example usage of Pi0Inference with CUDA MPS support.

This script demonstrates how to use the new forward_mps method
with different resource allocation strategies.
"""

import torch
from pi0_infer import Pi0Inference
import argparse
import os


def run_mps_experiments():
    """
    Run Pi0Inference with different CUDA MPS configurations.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_views', type=int, default=2, help='Number of views')
    parser.add_argument('--prompt_len', type=int, default=0, help='Prompt length')
    parser.add_argument('--chunk_size', type=int, default=63, help='Chunk size')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Checkpoint directory')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations')
    args = parser.parse_args()

    # Load checkpoint
    if args.checkpoint_dir:
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'checkpoint.pth'))
    else:
        checkpoint = {'language_embeds': torch.empty(args.prompt_len, 2048, dtype=torch.bfloat16)}

    # Initialize inference engine
    infer = Pi0Inference(checkpoint, num_views=args.num_views, chunk_size=args.chunk_size)

    # Prepare input tensors
    input_image = torch.empty(args.num_views, 224, 224, 3, dtype=torch.bfloat16).cuda()
    input_state = torch.empty(32, dtype=torch.bfloat16).cuda()
    input_noise = torch.empty(args.chunk_size, 32, dtype=torch.bfloat16).cuda()

    # ===== Test 1: Sequential execution with MPS (baseline) =====
    print("\n" + "="*80)
    print("Test 1: Sequential execution with MPS")
    print("="*80)
    for _ in range(args.iterations):
        torch.cuda.nvtx.range_push("MPS Sequential")
        _ = infer.forward_mps(
            input_image, 
            input_state, 
            input_noise, 
            mps_encoder_percentage=100,  # Encoder gets all resources
            mps_decoder_percentage=100,  # Then decoder gets all resources
            concurrent=False
        )
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print("Sequential MPS execution completed")

    # ===== Test 2: Concurrent MPS with 50-50 resource split =====
    print("\n" + "="*80)
    print("Test 2: Concurrent MPS with 50-50 resource split")
    print("="*80)
    for _ in range(args.iterations):
        torch.cuda.nvtx.range_push("MPS Concurrent 50-50")
        _ = infer.forward_mps(
            input_image, 
            input_state, 
            input_noise, 
            mps_encoder_percentage=50,   # Encoder gets 50% of SMs
            mps_decoder_percentage=50,   # Decoder gets 50% of SMs
            concurrent=True
        )
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print("Concurrent MPS 50-50 execution completed")

    # ===== Test 3: Concurrent MPS with encoder-heavy allocation =====
    print("\n" + "="*80)
    print("Test 3: Concurrent MPS with encoder-heavy allocation (70-30)")
    print("="*80)
    for _ in range(args.iterations):
        torch.cuda.nvtx.range_push("MPS Concurrent 70-30")
        _ = infer.forward_mps(
            input_image, 
            input_state, 
            input_noise, 
            mps_encoder_percentage=70,   # Encoder gets 70% of SMs
            mps_decoder_percentage=30,   # Decoder gets 30% of SMs
            concurrent=True
        )
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print("Concurrent MPS 70-30 execution completed")

    # ===== Test 4: Concurrent MPS with decoder-heavy allocation =====
    print("\n" + "="*80)
    print("Test 4: Concurrent MPS with decoder-heavy allocation (30-70)")
    print("="*80)
    for _ in range(args.iterations):
        torch.cuda.nvtx.range_push("MPS Concurrent 30-70")
        _ = infer.forward_mps(
            input_image, 
            input_state, 
            input_noise, 
            mps_encoder_percentage=30,   # Encoder gets 30% of SMs
            mps_decoder_percentage=70,   # Decoder gets 70% of SMs
            concurrent=True
        )
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print("Concurrent MPS 30-70 execution completed")

    # ===== Test 5: Compare with original concurrent streams (no MPS) =====
    print("\n" + "="*80)
    print("Test 5: Original concurrent streams (no MPS) for comparison")
    print("="*80)
    for _ in range(args.iterations):
        torch.cuda.nvtx.range_push("Original Concurrent Streams")
        _ = infer.forward(input_image, input_state, input_noise, concurrent=True)
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print("Original concurrent streams execution completed")

    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)
    print("\nNote: Use NSys (NVIDIA Nsight Systems) to analyze the timeline:")
    print("  nsys profile -o output python pi0_infer_mps_example.py")


if __name__ == "__main__":
    run_mps_experiments()
