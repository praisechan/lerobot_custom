#!/usr/bin/env python3
"""
Plot compute-memory contention benchmark results

Usage:
    python3 plot.py results.csv [output_prefix]
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_contention_results(csv_path, output_prefix=None):
    """
    Generate four plots analyzing compute-memory contention
    
    Args:
        csv_path: Path to CSV file with benchmark results
        output_prefix: Prefix for output files (default: based on csv_path)
    """
    # Read CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    
    # Validate columns
    required_cols = ['num_sms', 'mode', 'mem_bw_gb_s', 'gemm_gflops', 
                     'mem_time_ms', 'gemm_time_ms', 'mem_slowdown', 
                     'gemm_slowdown', 'overlap_pct']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV must contain columns: {required_cols}")
        print(f"Found columns: {df.columns.tolist()}")
        sys.exit(1)
    
    # Set output prefix
    if output_prefix is None:
        output_prefix = csv_path.replace('.csv', '')
    
    # Separate by mode
    isolated_mem = df[df['mode'] == 'isolated_mem'].sort_values('num_sms')
    isolated_gemm = df[df['mode'] == 'isolated_gemm'].sort_values('num_sms')
    concurrent = df[df['mode'] == 'concurrent'].sort_values('num_sms')
    
    # Create merged dataframe with explicit column naming
    merged = concurrent[['num_sms', 'mem_bw_gb_s', 'gemm_gflops', 'mem_slowdown', 
                         'gemm_slowdown', 'overlap_pct']].copy()
    merged.rename(columns={
        'mem_bw_gb_s': 'mem_bw_gb_s_concurrent',
        'gemm_gflops': 'gemm_gflops_concurrent'
    }, inplace=True)
    
    # Merge with isolated measurements
    merged = pd.merge(
        merged,
        isolated_mem[['num_sms', 'mem_bw_gb_s']].rename(columns={'mem_bw_gb_s': 'mem_bw_gb_s_isolated_mem'}),
        on='num_sms'
    )
    merged = pd.merge(
        merged,
        isolated_gemm[['num_sms', 'gemm_gflops']].rename(columns={'gemm_gflops': 'gemm_gflops_isolated_gemm'}),
        on='num_sms'
    )
    
    # Calculate retention percentages
    merged['mem_retention_pct'] = (merged['mem_bw_gb_s_concurrent'] / 
                                    merged['mem_bw_gb_s_isolated_mem']) * 100
    merged['gemm_retention_pct'] = (merged['gemm_gflops_concurrent'] / 
                                     merged['gemm_gflops_isolated_gemm']) * 100
    
    # =========================================================================
    # Plot 1: Performance Retention vs SM Count
    # =========================================================================
    
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    
    ax1.plot(merged['num_sms'], merged['mem_retention_pct'], 
             marker='o', markersize=8, linewidth=2.5, 
             label='Memory Bandwidth Retention', color='#2E86AB')
    
    ax1.plot(merged['num_sms'], merged['gemm_retention_pct'], 
             marker='s', markersize=8, linewidth=2.5, 
             label='GEMM Throughput Retention', color='#A23B72')
    
    ax1.axhline(100, color='gray', linestyle='--', linewidth=1.5, 
                alpha=0.7, label='100% (No Degradation)')
    
    ax1.set_xlabel('Number of SMs', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Performance Retention (%)', fontsize=14, fontweight='bold')
    ax1.set_title('Concurrent Execution Performance Retention vs SM Count', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 110)
    
    plt.tight_layout()
    output_file = f"{output_prefix}_performance_retention.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # =========================================================================
    # Plot 2: Absolute Performance vs SM Count (Dual Y-axis)
    # =========================================================================
    
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    
    # Left Y-axis: Bandwidth (GB/s)
    color_mem = '#2E86AB'
    ax2.set_xlabel('Number of SMs', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Memory Bandwidth (GB/s)', fontsize=14, fontweight='bold', color=color_mem)
    
    line1 = ax2.plot(isolated_mem['num_sms'], isolated_mem['mem_bw_gb_s'], 
                     marker='o', markersize=8, linewidth=2.5, linestyle='--',
                     label='Isolated Mem BW', color=color_mem, alpha=0.7)
    
    line2 = ax2.plot(concurrent['num_sms'], concurrent['mem_bw_gb_s'], 
                     marker='o', markersize=8, linewidth=2.5,
                     label='Concurrent Mem BW', color=color_mem)
    
    ax2.tick_params(axis='y', labelcolor=color_mem)
    ax2.fill_between(concurrent['num_sms'], 
                      concurrent['mem_bw_gb_s'], 
                      isolated_mem['mem_bw_gb_s'],
                      alpha=0.2, color=color_mem, label='Mem BW Degradation')
    
    # Right Y-axis: Throughput (GFLOPS)
    ax2_right = ax2.twinx()
    color_gemm = '#A23B72'
    ax2_right.set_ylabel('GEMM Throughput (GFLOPS)', fontsize=14, 
                         fontweight='bold', color=color_gemm)
    
    line3 = ax2_right.plot(isolated_gemm['num_sms'], isolated_gemm['gemm_gflops'], 
                           marker='s', markersize=8, linewidth=2.5, linestyle='--',
                           label='Isolated GEMM', color=color_gemm, alpha=0.7)
    
    line4 = ax2_right.plot(concurrent['num_sms'], concurrent['gemm_gflops'], 
                           marker='s', markersize=8, linewidth=2.5,
                           label='Concurrent GEMM', color=color_gemm)
    
    ax2_right.tick_params(axis='y', labelcolor=color_gemm)
    
    # Combine legends
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, fontsize=11, loc='upper left')
    
    ax2.set_title('Absolute Performance: Isolated vs Concurrent Execution', 
                  fontsize=16, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = f"{output_prefix}_absolute_performance.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # =========================================================================
    # Plot 3: Slowdown Ratio vs SM Count
    # =========================================================================
    
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    
    ax3.plot(concurrent['num_sms'], concurrent['mem_slowdown'], 
             marker='o', markersize=8, linewidth=2.5, 
             label='Memory Copy Slowdown', color='#2E86AB')
    
    ax3.plot(concurrent['num_sms'], concurrent['gemm_slowdown'], 
             marker='s', markersize=8, linewidth=2.5, 
             label='GEMM Slowdown', color='#A23B72')
    
    ax3.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, 
                alpha=0.7, label='1.0x (No Slowdown)')
    
    ax3.set_xlabel('Number of SMs', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Slowdown Ratio', fontsize=14, fontweight='bold')
    ax3.set_title('Contention-Induced Slowdown vs SM Count', 
                  fontsize=16, fontweight='bold', pad=20)
    ax3.legend(fontsize=12, loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.9, max(concurrent['mem_slowdown'].max(), 
                          concurrent['gemm_slowdown'].max()) * 1.1)
    
    plt.tight_layout()
    output_file = f"{output_prefix}_slowdown_ratios.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # =========================================================================
    # Plot 4: Concurrent Execution Efficiency
    # =========================================================================
    
    fig4, ax4 = plt.subplots(figsize=(12, 7))
    
    ax4.plot(concurrent['num_sms'], concurrent['overlap_pct'], 
             marker='D', markersize=8, linewidth=2.5, 
             label='Time Overlap Percentage', color='#F18F01')
    
    ax4.axhline(0, color='gray', linestyle='--', linewidth=1.5, 
                alpha=0.7, label='0% (Sequential Execution)')
    
    ax4.set_xlabel('Number of SMs', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Overlap Percentage (%)', fontsize=14, fontweight='bold')
    ax4.set_title('Concurrent Execution Efficiency (Time Overlap)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax4.legend(fontsize=12, loc='best')
    ax4.grid(True, alpha=0.3)
    
    # Add informative text
    avg_overlap = concurrent['overlap_pct'].mean()
    ax4.text(0.02, 0.98, f'Average Overlap: {avg_overlap:.1f}%',
             transform=ax4.transAxes, fontsize=12,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_file = f"{output_prefix}_concurrent_efficiency.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # =========================================================================
    # Summary Statistics
    # =========================================================================
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    print(f"\nMemory Bandwidth Retention:")
    print(f"  Min:  {merged['mem_retention_pct'].min():.1f}%")
    print(f"  Max:  {merged['mem_retention_pct'].max():.1f}%")
    print(f"  Mean: {merged['mem_retention_pct'].mean():.1f}%")
    
    print(f"\nGEMM Throughput Retention:")
    print(f"  Min:  {merged['gemm_retention_pct'].min():.1f}%")
    print(f"  Max:  {merged['gemm_retention_pct'].max():.1f}%")
    print(f"  Mean: {merged['gemm_retention_pct'].mean():.1f}%")
    
    print(f"\nMemory Copy Slowdown:")
    print(f"  Min:  {concurrent['mem_slowdown'].min():.2f}x")
    print(f"  Max:  {concurrent['mem_slowdown'].max():.2f}x")
    print(f"  Mean: {concurrent['mem_slowdown'].mean():.2f}x")
    
    print(f"\nGEMM Slowdown:")
    print(f"  Min:  {concurrent['gemm_slowdown'].min():.2f}x")
    print(f"  Max:  {concurrent['gemm_slowdown'].max():.2f}x")
    print(f"  Mean: {concurrent['gemm_slowdown'].mean():.2f}x")
    
    print(f"\nConcurrent Execution Overlap:")
    print(f"  Min:  {concurrent['overlap_pct'].min():.1f}%")
    print(f"  Max:  {concurrent['overlap_pct'].max():.1f}%")
    print(f"  Mean: {concurrent['overlap_pct'].mean():.1f}%")
    
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("""
Performance Retention: Higher is better (100% = no degradation)
  - >90%: Minimal contention
  - 70-90%: Moderate contention
  - <70%: Significant contention

Slowdown Ratio: Closer to 1.0 is better (1.0x = no slowdown)
  - 1.0-1.2x: Minimal impact
  - 1.2-1.5x: Moderate impact
  - >1.5x: Significant impact

Overlap Percentage: Higher is better (indicates good concurrency)
  - >60%: Kernels execute mostly concurrently
  - 30-60%: Partial concurrent execution
  - <30%: Mostly sequential execution
    """)
    print("="*70 + "\n")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else None
    
    plot_contention_results(csv_path, output_prefix)
    
    print("\nAll plots generated successfully!")
