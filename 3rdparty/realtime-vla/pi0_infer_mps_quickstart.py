#!/usr/bin/env python3
"""
Quick Start Guide for Pi0Inference CUDA MPS Support

This script shows the simplest way to use the new forward_mps() method.
"""

import torch
from pi0_infer import Pi0Inference

# Initialize model
checkpoint = {'language_embeds': torch.empty(0, 2048, dtype=torch.bfloat16)}
infer = Pi0Inference(checkpoint, num_views=2, chunk_size=63)

# Prepare inputs
input_image = torch.empty(2, 224, 224, 3, dtype=torch.bfloat16).cuda()
input_state = torch.empty(32, dtype=torch.bfloat16).cuda()
input_noise = torch.empty(63, 32, dtype=torch.bfloat16).cuda()

# ============================================================================
# QUICK EXAMPLES
# ============================================================================

# ✅ Example 1: Simplest - Balanced 50-50 concurrent execution
print("Running: Balanced 50-50 concurrent MPS")
output = infer.forward_mps(input_image, input_state, input_noise, concurrent=True)

# ✅ Example 2: Encoder-heavy (70% encoder, 30% decoder)
print("Running: Encoder-heavy 70-30 concurrent MPS")
output = infer.forward_mps(
    input_image, input_state, input_noise,
    mps_encoder_percentage=70,
    mps_decoder_percentage=30,
    concurrent=True
)

# ✅ Example 3: Decoder-heavy (30% encoder, 70% decoder)
print("Running: Decoder-heavy 30-70 concurrent MPS")
output = infer.forward_mps(
    input_image, input_state, input_noise,
    mps_encoder_percentage=30,
    mps_decoder_percentage=70,
    concurrent=True
)

# ✅ Example 4: Sequential execution
print("Running: Sequential MPS (encoder first, then decoder)")
output = infer.forward_mps(
    input_image, input_state, input_noise,
    concurrent=False
)

# ✅ Example 5: Using the original concurrent method (for comparison)
print("Running: Original concurrent streams (no MPS)")
output = infer.forward(input_image, input_state, input_noise, concurrent=True)

print("\n✅ All examples completed successfully!")
print("\nFor detailed documentation, see: CUDA_MPS_GUIDE.md")
