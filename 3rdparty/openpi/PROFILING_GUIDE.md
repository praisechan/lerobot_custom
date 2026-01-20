# π0 Profiling Guide with JAX TraceAnnotation

## Overview
This guide explains the detailed layer-by-layer profiling annotations added to the π0 model to identify performance bottlenecks.

## Changes Made

### 1. Added TraceAnnotation to Gemma Transformer (`src/openpi/models/gemma.py`)

We added detailed `jax.profiler.TraceAnnotation` markers throughout the transformer layers:

#### **LLM Main Forward Pass**
- `llm_embed_prepare` - Input embedding and mask preparation
- `llm_transformer_layers` - All transformer blocks (main computation)
- `llm_final_norm` - Final layer normalization

#### **Transformer Block Level**
Each transformer block now has these annotations:
- `block_pre_attention` - Pre-attention normalization and setup
- `block_attention` - Self-attention computation (see detailed breakdown below)
- `block_post_attention` - Post-attention residual connection
- `block_pre_ffn` - Pre-FFN normalization
- `block_post_ffn` - Post-FFN residual connection

#### **Attention Layer Breakdown**
The attention mechanism is broken down into:
- `attn_qkv_proj` - Query/Key/Value projection (matrix multiplications)
- `attn_rope` - Rotary Position Embedding application
- `attn_kv_cache` - KV cache concatenation (for autoregressive generation)
- `attn_compute_logits` - Compute attention scores (Q @ K^T)
- `attn_softmax` - Softmax with attention masking
- `attn_weighted_sum` - Weighted sum of values (Attention @ V)
- `attn_output_proj` - Output projection

### 2. Existing Annotations in π0 Model (`src/openpi/models/pi0.py`)

The π0 model already had high-level annotations:
- `sample_actions` - Outer wrapper
- `preprocess_observation` - Input preprocessing
- `embed_prefix` - Vision + state embedding
- `compute_kv_cache` - Initial KV cache computation
- `diffusion_loop` - Diffusion sampling loop
  - `step_embed_suffix` - Action embedding at each diffusion step
  - `step_compute_masks` - Attention mask computation
  - `step_compute_positions` - Position computation
  - `step_llm_forward` - LLM forward pass (this is where the transformer layers run)
  - `step_action_proj` - Action projection

## How to View Profiling Results

### Option 1: XProf (Recommended for Remote Machines)
```bash
# On remote machine:
xprof --port 8791 /tmp/pi0-libero-profile

# Then open in browser:
http://localhost:8791/
# Select your run from dropdown → Tools → trace_viewer
```

### Option 2: Perfetto (Local or Downloaded Traces)
```bash
# Download trace from remote:
scp -r user@remote:/tmp/pi0-libero-profile .

# Go to https://ui.perfetto.dev/
# Click "Open trace file" and select the .json.gz or .pb.gz file
```

## What to Look For in the Trace

### 1. **Top-Level Structure**
Look for the diffusion loop iterations. Each iteration should show:
```
step_llm_forward (largest time consumer)
  ├─ llm_embed_prepare
  ├─ llm_transformer_layers (main bottleneck)
  │   ├─ block_attention (repeated for each layer)
  │   └─ block_pre_ffn + block_post_ffn (repeated for each layer)
  └─ llm_final_norm
```

### 2. **Expected Bottlenecks**
Based on typical transformer behavior:
- **Most time**: `llm_transformer_layers` (this contains all 18 transformer blocks for Gemma-2B)
- **Within each block**:
  - `block_attention` - Usually 30-40% of block time
    - `attn_compute_logits` - Matrix multiplication (Q @ K^T)
    - `attn_softmax` - Softmax operation
    - `attn_weighted_sum` - Another matrix multiplication (Attention @ V)
  - `block_pre_ffn` + `block_post_ffn` - Usually 50-60% of block time
    - Contains 2 large matrix multiplications (up projection and down projection)

### 3. **Performance Analysis**
To identify bottlenecks:
1. **Total Time Distribution**:
   - Find which annotation takes the most time
   - For Gemma-2B with 18 layers, `llm_transformer_layers` should dominate

2. **Per-Layer Analysis**:
   - Compare time spent in attention vs FFN
   - FFN is typically 1.5-2x slower than attention (has 4x parameters)

3. **Attention Breakdown**:
   - Check if `attn_qkv_proj` or `attn_output_proj` are slow (could indicate memory bandwidth issues)
   - Check if `attn_compute_logits` is slow (compute-bound)
   - Check if `attn_softmax` is slow (memory-bound)

4. **Memory Operations**:
   - `attn_kv_cache` - Should be fast for short sequences
   - If slow, might indicate memory bandwidth bottleneck

### 4. **Optimization Opportunities**
Based on profiling results:
- **If attention is slow**: Consider FlashAttention or other fused kernels
- **If FFN is slow**: Consider fused FFN kernels or quantization
- **If KV cache operations are slow**: Consider better memory layout or caching strategy
- **If rope is slow**: Consider precomputing rope frequencies

## Example Trace Interpretation

```
sample_actions: 500ms (total)
├─ preprocess_observation: 10ms
├─ embed_prefix: 20ms
├─ compute_kv_cache: 100ms
│   └─ llm_transformer_layers: 95ms
└─ diffusion_loop: 370ms (10 steps)
    └─ Per step (~37ms each):
        ├─ step_embed_suffix: 2ms
        ├─ step_compute_masks: 1ms
        ├─ step_llm_forward: 33ms ← BOTTLENECK
        │   ├─ llm_embed_prepare: 0.5ms
        │   ├─ llm_transformer_layers: 32ms ← MAIN BOTTLENECK
        │   │   └─ Per layer (~1.8ms × 18 layers):
        │   │       ├─ block_attention: 0.7ms (39%)
        │   │       │   ├─ attn_qkv_proj: 0.2ms
        │   │       │   ├─ attn_compute_logits: 0.2ms
        │   │       │   ├─ attn_softmax: 0.1ms
        │   │       │   └─ attn_weighted_sum: 0.2ms
        │   │       └─ block_pre_ffn + block_post_ffn: 1.1ms (61%)
        │   └─ llm_final_norm: 0.5ms
        └─ step_action_proj: 1ms
```

In this example:
- The LLM forward pass takes 89% of time per diffusion step
- Within LLM, the transformer layers take 97% of time
- FFN (61%) takes more time than attention (39%), which is expected
- Optimization should focus on `llm_transformer_layers`, especially FFN operations

## Running the Profiler

```bash
cd /home/juchanlee/lerobot_custom/openpi
uv run examples/libero/profile_pi0_detailed.py
```

The trace will be saved to `/tmp/pi0-libero-profile`.

## Technical Notes

- **JAX JIT Compatibility**: `jax.profiler.TraceAnnotation` works inside JIT-compiled code, unlike NVTX markers
- **Overhead**: TraceAnnotation has minimal overhead (~1-2% typically)
- **Nesting**: Annotations are properly nested to show the call hierarchy
- **Multiple Devices**: For multi-GPU setups, annotations will show per-device breakdown

## Further Reading

- [JAX Profiling Guide](https://docs.jax.dev/en/latest/profiling.html)
- [Perfetto UI Documentation](https://perfetto.dev/docs/)
- [XLA Performance Profiling](https://www.tensorflow.org/xla/tfprof)
