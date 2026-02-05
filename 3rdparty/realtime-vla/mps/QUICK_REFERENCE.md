# Pi0Inference CUDA MPS - Quick Reference Card

## 🎯 One-Liner Examples

```python
# Basic (default 50-50 concurrent)
output = infer.forward_mps(image, state, noise, concurrent=True)

# Encoder-heavy
output = infer.forward_mps(image, state, noise, mps_encoder_percentage=70, concurrent=True)

# Decoder-heavy  
output = infer.forward_mps(image, state, noise, mps_encoder_percentage=30, concurrent=True)

# Sequential
output = infer.forward_mps(image, state, noise, concurrent=False)

# Original (CUDA graphs)
output = infer.forward(image, state, noise, concurrent=True)
```

## 📊 Method Comparison Matrix

```
┌──────────────────────┬────────────────────┬──────────────────────┐
│ Method               │ Execution Mode     │ SM Control           │
├──────────────────────┼────────────────────┼──────────────────────┤
│ forward()            │ CUDA Graphs        │ Implicit             │
│ forward_mps()        │ Live code          │ Explicit (50-50 def) │
└──────────────────────┴────────────────────┴──────────────────────┘

Use forward() for:          Use forward_mps() for:
├─ Maximum performance      ├─ Resource tuning
├─ Real-time latency        ├─ Load balancing
└─ CUDA graph replay        └─ Experimentation
```

## 🔧 Parameter Reference

```python
def forward_mps(
    self, 
    observation_images_normalized,        # Tensor [num_views, 224, 224, 3]
    observation_state_normalized,         # Tensor [32]
    diffusion_noise,                      # Tensor [chunk_size, 32]
    mps_encoder_percentage=50,            # 1-100, default 50
    mps_decoder_percentage=50,            # 1-100, default 50
    concurrent=False                      # bool, default False
)
```

## ⚡ Performance Tuning

```
Balanced (50-50) ────────────────────────────────────────
  ├─ Good starting point
  └─ Equal load assumption

Encoder-Heavy (70-30) ────────────────────────────────────
  ├─ When encoder is bottleneck
  └─ Try: 60-40, 70-30, 80-20

Decoder-Heavy (30-70) ────────────────────────────────────
  ├─ When decoder is bottleneck
  └─ Try: 40-60, 30-70, 20-80

Sequential (100-100) ──────────────────────────────────────
  ├─ Baseline measurement
  └─ Each stage gets full resources
```

## 📈 Quick Benchmark

```bash
# Run all configurations
python benchmark_forward_methods.py --iterations 20

# Output shows:
# - Fastest method
# - Performance difference vs baseline
# - Recommendations
```

## 🎓 Learning Path

```
1. Read README_MPS.md (5 min)
   └─ Get overview

2. Run pi0_infer_mps_quickstart.py (2 min)
   └─ See it work

3. Read CUDA_MPS_GUIDE.md (15 min)
   └─ Understand details

4. Run pi0_infer_mps_example.py (10 min)
   └─ Explore configurations

5. Run benchmark_forward_methods.py (20 min)
   └─ Optimize for your hardware

6. Profile with NSys (5 min)
   └─ Visualize execution timeline
```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| MPS allocation not taking effect | Add `torch.cuda.synchronize()` before/after change |
| Concurrent slower than sequential | Try different split (70-30, 80-20) |
| GPU not supported | Check GPU is Maxwell or newer (GTX 750+) |
| Import errors | Ensure pi0_infer.py is in Python path |

## 📊 Benchmark Configurations

```
1. forward() - Sequential
2. forward() - Concurrent (CUDA Graphs)  ← Baseline
3. forward_mps() - Sequential (100-100)
4. forward_mps() - Concurrent (50-50)     ← Balanced
5. forward_mps() - Concurrent (70-30)     ← Encoder-heavy
6. forward_mps() - Concurrent (30-70)     ← Decoder-heavy
```

## 🎯 Decision Tree

```
Do you need maximum performance?
├─ YES → Use forward(concurrent=True)
└─ NO → Continue

Do you want to tune SM allocation?
├─ YES → Use forward_mps(concurrent=True)
└─ NO → Use forward(concurrent=True)

Is one component slower?
├─ YES → Use forward_mps() with custom split
└─ NO → Use balanced 50-50 split
```

## 🔗 File Quick Links

| File | Purpose | Read Time |
|------|---------|-----------|
| [README_MPS.md](README_MPS.md) | Package overview | 5 min |
| [MPS_IMPLEMENTATION_SUMMARY.md](MPS_IMPLEMENTATION_SUMMARY.md) | How it works | 10 min |
| [CUDA_MPS_GUIDE.md](CUDA_MPS_GUIDE.md) | Complete reference | 20 min |
| [COMPARISON_forward_vs_forward_mps.py](COMPARISON_forward_vs_forward_mps.py) | Comparison guide | 10 min |
| [pi0_infer_mps_quickstart.py](pi0_infer_mps_quickstart.py) | Minimal examples | 1 min |
| [pi0_infer_mps_example.py](pi0_infer_mps_example.py) | Full examples | 5 min |
| [benchmark_forward_methods.py](benchmark_forward_methods.py) | Benchmarks | 20 min |

## 🚀 Common Workflows

### Workflow 1: Get Started (5 minutes)
```bash
python pi0_infer_mps_quickstart.py
# Done! You've seen all basic patterns
```

### Workflow 2: Understand Differences (20 minutes)
```bash
python pi0_infer_mps_example.py --iterations 10
# Saw different configurations
# Then read CUDA_MPS_GUIDE.md
```

### Workflow 3: Optimize for Hardware (30 minutes)
```bash
python benchmark_forward_methods.py --iterations 50
# Got detailed performance comparison
# Ready to choose best configuration
```

### Workflow 4: Profile Execution (15 minutes)
```bash
nsys profile -o output python pi0_infer_mps_example.py
nsys-ui output.nsys-rep
# Visualized execution timeline
# Identified optimization opportunities
```

## 💡 Pro Tips

1. **Always warmup first**: GPU needs a few iterations to reach optimal clocks
2. **Use NVTX markers**: Built into examples for easy NSys profiling
3. **Try different splits**: What works on your GPU may differ from others
4. **Monitor memory**: Check memory bandwidth for contention
5. **Save optimal config**: Once tuned, hardcode the percentages

## 📈 Expected Performance

```
Configuration          | Latency vs forward()  | Flexibility
────────────────────────────────────────────────────────────
forward() concurrent   | 0% (baseline)        | None
forward_mps() 50-50    | +5-15%               | High
forward_mps() 70-30    | +5-12% (tuned)       | High
forward_mps() 30-70    | +8-15% (tuned)       | High
```

## ✅ Verification Checklist

- [ ] Read README_MPS.md
- [ ] Run quickstart.py
- [ ] Run example.py
- [ ] Run benchmark.py
- [ ] Read CUDA_MPS_GUIDE.md
- [ ] Profile with NSys
- [ ] Choose optimal configuration
- [ ] Integrate into production

---

**Status**: Ready to Use ✅  
**Last Updated**: 2026-02-05  
**Questions?** Check CUDA_MPS_GUIDE.md or run the examples!
