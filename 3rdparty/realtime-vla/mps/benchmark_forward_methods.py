#!/usr/bin/env python3
"""
Benchmarking Script: Compare forward() vs forward_mps() with different allocations

This script measures execution time and helps you find the optimal configuration
for your specific hardware and workload.
"""

import torch
import time
from pi0_infer import Pi0Inference
import argparse


def benchmark_method(infer, input_image, input_state, input_noise, 
                     method_name, iterations=10, **kwargs):
    """
    Benchmark a specific forward method configuration.
    
    Returns:
        Dictionary with timing statistics
    """
    # Warmup runs
    print(f"  Warming up {method_name}...", end=" ", flush=True)
    for _ in range(3):
        if "forward_mps" in method_name:
            infer.forward_mps(input_image, input_state, input_noise, **kwargs)
        else:
            infer.forward(input_image, input_state, input_noise, **kwargs)
    torch.cuda.synchronize()
    print("✓")
    
    # Timed runs
    times = []
    print(f"  Running {iterations} iterations...", end=" ", flush=True)
    
    start_total = time.perf_counter()
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        if "forward_mps" in method_name:
            _ = infer.forward_mps(input_image, input_state, input_noise, **kwargs)
        else:
            _ = infer.forward(input_image, input_state, input_noise, **kwargs)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    total_time = (time.perf_counter() - start_total) * 1000
    
    print("✓")
    
    # Calculate statistics
    times_sorted = sorted(times)
    results = {
        'method': method_name,
        'iterations': iterations,
        'mean': sum(times) / len(times),
        'median': times_sorted[len(times) // 2],
        'min': min(times),
        'max': max(times),
        'p95': times_sorted[int(len(times) * 0.95)],
        'p99': times_sorted[int(len(times) * 0.99)],
        'total_time': total_time,
    }
    
    return results


def print_results(results_list):
    """Pretty print benchmark results."""
    print("\n" + "="*100)
    print("BENCHMARK RESULTS")
    print("="*100)
    
    # Header
    print(f"{'Method':<40} {'Mean (ms)':<12} {'Median (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
    print("-"*100)
    
    # Results
    for r in results_list:
        method = r['method']
        if len(method) > 39:
            method = method[:36] + "..."
        
        print(f"{method:<40} {r['mean']:<12.2f} {r['median']:<12.2f} {r['min']:<12.2f} {r['max']:<12.2f}")
    
    # Summary
    print("-"*100)
    print("\nDetailed Statistics:")
    for r in results_list:
        print(f"\n{r['method']}:")
        print(f"  Mean:      {r['mean']:.2f} ms")
        print(f"  Median:    {r['median']:.2f} ms")
        print(f"  Min:       {r['min']:.2f} ms")
        print(f"  Max:       {r['max']:.2f} ms")
        print(f"  P95:       {r['p95']:.2f} ms")
        print(f"  P99:       {r['p99']:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Pi0Inference forward methods")
    parser.add_argument('--num_views', type=int, default=2, help='Number of views')
    parser.add_argument('--prompt_len', type=int, default=0, help='Prompt length')
    parser.add_argument('--chunk_size', type=int, default=63, help='Chunk size')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Checkpoint directory')
    parser.add_argument('--iterations', type=int, default=20, help='Number of iterations per test')
    args = parser.parse_args()

    print("\n" + "="*100)
    print("Pi0Inference Benchmarking Suite")
    print("="*100)
    
    # Initialize model
    print("\nInitializing model...", end=" ", flush=True)
    if args.checkpoint_dir:
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'checkpoint.pth'))
    else:
        checkpoint = {'language_embeds': torch.empty(args.prompt_len, 2048, dtype=torch.bfloat16)}
    
    infer = Pi0Inference(checkpoint, num_views=args.num_views, chunk_size=args.chunk_size)
    print("✓")
    
    # Prepare inputs
    print("Preparing input tensors...", end=" ", flush=True)
    input_image = torch.empty(args.num_views, 224, 224, 3, dtype=torch.bfloat16).cuda()
    input_state = torch.empty(32, dtype=torch.bfloat16).cuda()
    input_noise = torch.empty(args.chunk_size, 32, dtype=torch.bfloat16).cuda()
    print("✓")
    
    print(f"\nConfiguration:")
    print(f"  num_views: {args.num_views}")
    print(f"  chunk_size: {args.chunk_size}")
    print(f"  iterations per test: {args.iterations}")
    
    results = []
    
    # Test 1: Original forward (sequential)
    print("\n[1/6] Benchmarking forward() - Sequential")
    results.append(benchmark_method(
        infer, input_image, input_state, input_noise,
        "forward() - Sequential",
        iterations=args.iterations,
        concurrent=False
    ))
    
    # Test 2: Original forward (concurrent)
    print("\n[2/6] Benchmarking forward() - Concurrent (CUDA Graphs)")
    results.append(benchmark_method(
        infer, input_image, input_state, input_noise,
        "forward() - Concurrent (CUDA Graphs)",
        iterations=args.iterations,
        concurrent=True
    ))
    
    # Test 3: forward_mps sequential
    print("\n[3/6] Benchmarking forward_mps() - Sequential")
    results.append(benchmark_method(
        infer, input_image, input_state, input_noise,
        "forward_mps() - Sequential (100-100)",
        iterations=args.iterations,
        concurrent=False
    ))
    
    # Test 4: forward_mps concurrent 50-50
    print("\n[4/6] Benchmarking forward_mps() - Concurrent (50-50 split)")
    results.append(benchmark_method(
        infer, input_image, input_state, input_noise,
        "forward_mps() - Concurrent (50-50)",
        iterations=args.iterations,
        mps_encoder_percentage=50,
        mps_decoder_percentage=50,
        concurrent=True
    ))
    
    # Test 5: forward_mps concurrent 70-30
    print("\n[5/6] Benchmarking forward_mps() - Concurrent (70-30 split)")
    results.append(benchmark_method(
        infer, input_image, input_state, input_noise,
        "forward_mps() - Concurrent (70-30)",
        iterations=args.iterations,
        mps_encoder_percentage=70,
        mps_decoder_percentage=30,
        concurrent=True
    ))
    
    # Test 6: forward_mps concurrent 30-70
    print("\n[6/6] Benchmarking forward_mps() - Concurrent (30-70 split)")
    results.append(benchmark_method(
        infer, input_image, input_state, input_noise,
        "forward_mps() - Concurrent (30-70)",
        iterations=args.iterations,
        mps_encoder_percentage=30,
        mps_decoder_percentage=70,
        concurrent=True
    ))
    
    # Print results
    print_results(results)
    
    # Performance analysis
    print("\n" + "="*100)
    print("PERFORMANCE ANALYSIS")
    print("="*100)
    
    # Find fastest method
    fastest = min(results, key=lambda x: x['mean'])
    print(f"\nFastest method: {fastest['method']}")
    print(f"  Mean latency: {fastest['mean']:.2f} ms")
    
    # Compare baseline (forward concurrent)
    baseline = next((r for r in results if "forward() - Concurrent" in r['method']), None)
    if baseline:
        print(f"\nBaseline (forward() - Concurrent): {baseline['mean']:.2f} ms")
        for r in results:
            if r != baseline:
                diff = ((r['mean'] - baseline['mean']) / baseline['mean']) * 100
                symbol = "↑" if diff > 0 else "↓"
                print(f"  {r['method']:<50} {symbol} {abs(diff):+.1f}%")
    
    print("\n" + "="*100)
    print("\nRECOMMENDATIONS:")
    print("-"*100)
    print("1. Use forward() if:")
    print("   - You need maximum performance")
    print("   - Latency is critical (real-time applications)")
    print("   - You don't need to adjust resource allocation")
    print()
    print("2. Use forward_mps() if:")
    print("   - You want to tune resource allocation")
    print("   - Performance difference is < 5%")
    print("   - You need flexibility for different workloads")
    print()


if __name__ == "__main__":
    import os
    main()
