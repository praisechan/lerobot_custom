#!/usr/bin/env python3
"""
Plot size sweep experiment results showing kernel slowdown behavior.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def read_csv_with_metadata(csv_path):
    """
    Read CSV file and extract relevant columns.
    Expected columns: num_sms, gemv_M, gemv_N, gemm_size, mode,
                     gemv_gflops, gemm_gflops, gemv_time_ms, gemm_time_ms,
                     gemv_slowdown, gemm_slowdown, overlap_pct
    
    Each configuration has 3 rows (isolated_gemv, isolated_gemm, concurrent).
    We need to extract data from all three modes and combine them.
    """
    df = pd.read_csv(csv_path)
    
    # Get unique configurations
    configs = df[['num_sms', 'gemv_M', 'gemv_N', 'gemm_size']].drop_duplicates()
    
    # Build result dataframe
    results = []
    for _, config in configs.iterrows():
        mask = (df['num_sms'] == config['num_sms']) & \
               (df['gemv_M'] == config['gemv_M']) & \
               (df['gemv_N'] == config['gemv_N']) & \
               (df['gemm_size'] == config['gemm_size'])
        
        subset = df[mask]
        
        gemv_only = subset[subset['mode'] == 'isolated_gemv'].iloc[0]
        gemm_only = subset[subset['mode'] == 'isolated_gemm'].iloc[0]
        concurrent = subset[subset['mode'] == 'concurrent'].iloc[0]
        
        results.append({
            'num_sms': config['num_sms'],
            'gemv_M': config['gemv_M'],
            'gemv_N': config['gemv_N'],
            'gemm_size': config['gemm_size'],
            'gemv_only_time_ms': gemv_only['gemv_time_ms'],
            'gemm_only_time_ms': gemm_only['gemm_time_ms'],
            'concurrent_gemv_time_ms': concurrent['gemv_time_ms'],
            'concurrent_gemm_time_ms': concurrent['gemm_time_ms'],
            'gemv_slowdown': concurrent['gemv_slowdown'],
            'gemm_slowdown': concurrent['gemm_slowdown'],
            'overlap_pct': concurrent['overlap_pct'],
        })
    
    return pd.DataFrame(results)


def plot_case1(df, output_dir, show_plot=False):
    """
    Plot Case 1: Fixed GEMV, Varying GEMM
    Shows how GEMV and GEMM slowdown changes as GEMM size increases
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Case 1: Fixed GEMV (32768x32768), Varying GEMM Size', fontsize=16, fontweight='bold')
    
    gemm_sizes = df['gemm_size'].values
    gemv_slowdown = df['gemv_slowdown'].values
    gemm_slowdown = df['gemm_slowdown'].values
    gemv_only_time = df['gemv_only_time_ms'].values
    gemm_only_time = df['gemm_only_time_ms'].values
    concurrent_gemv_time = df['concurrent_gemv_time_ms'].values
    concurrent_gemm_time = df['concurrent_gemm_time_ms'].values
    
    # Plot 1: Slowdown vs GEMM size
    ax1 = axes[0]
    ax1.plot(gemm_sizes, gemv_slowdown, 'o-', linewidth=2, markersize=8, label='GEMV Slowdown', color='#1f77b4')
    ax1.plot(gemm_sizes, gemm_slowdown, 's-', linewidth=2, markersize=8, label='GEMM Slowdown', color='#ff7f0e')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No Slowdown')
    ax1.set_xlabel('GEMM Size (N×N×N)', fontsize=12)
    ax1.set_ylabel('Slowdown Factor', fontsize=12)
    ax1.set_title('Kernel Slowdown vs GEMM Size', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Combined kernel times (Isolated + Concurrent)
    ax2 = axes[1]
    ax2.plot(gemm_sizes, gemv_only_time, 'o-', linewidth=2, markersize=8, label='GEMV Isolated', color='#1f77b4', alpha=0.7)
    ax2.plot(gemm_sizes, gemm_only_time, 's-', linewidth=2, markersize=8, label='GEMM Isolated', color='#ff7f0e', alpha=0.7)
    ax2.plot(gemm_sizes, concurrent_gemv_time, 'o--', linewidth=2, markersize=8, label='GEMV Concurrent', color='#1f77b4')
    ax2.plot(gemm_sizes, concurrent_gemm_time, 's--', linewidth=2, markersize=8, label='GEMM Concurrent', color='#ff7f0e')
    ax2.set_xlabel('GEMM Size (N×N×N)', fontsize=12)
    ax2.set_ylabel('Time (ms/iteration)', fontsize=12)
    ax2.set_title('Kernel Times (Isolated vs Concurrent)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'case1_fixed_gemv_vary_gemm.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    
    if show_plot:
        plt.show()
    plt.close()


def plot_case2(df, output_dir, show_plot=False):
    """
    Plot Case 2: Fixed GEMM, Varying GEMV
    Shows how GEMV and GEMM slowdown changes as GEMV size increases
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Case 2: Fixed GEMM (4096³), Varying GEMV Size', fontsize=16, fontweight='bold')
    
    gemv_sizes = df['gemv_M'].values  # Using M as the size indicator (M=N)
    gemv_slowdown = df['gemv_slowdown'].values
    gemm_slowdown = df['gemm_slowdown'].values
    gemv_only_time = df['gemv_only_time_ms'].values
    gemm_only_time = df['gemm_only_time_ms'].values
    concurrent_gemv_time = df['concurrent_gemv_time_ms'].values
    concurrent_gemm_time = df['concurrent_gemm_time_ms'].values
    
    # Plot 1: Slowdown vs GEMV size
    ax1 = axes[0]
    ax1.plot(gemv_sizes, gemv_slowdown, 'o-', linewidth=2, markersize=8, label='GEMV Slowdown', color='#1f77b4')
    ax1.plot(gemv_sizes, gemm_slowdown, 's-', linewidth=2, markersize=8, label='GEMM Slowdown', color='#ff7f0e')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No Slowdown')
    ax1.set_xlabel('GEMV Size (M×N)', fontsize=12)
    ax1.set_ylabel('Slowdown Factor', fontsize=12)
    ax1.set_title('Kernel Slowdown vs GEMV Size', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Combined kernel times (Isolated + Concurrent)
    ax2 = axes[1]
    ax2.plot(gemv_sizes, gemv_only_time, 'o-', linewidth=2, markersize=8, label='GEMV Isolated', color='#1f77b4', alpha=0.7)
    ax2.plot(gemv_sizes, gemm_only_time, 's-', linewidth=2, markersize=8, label='GEMM Isolated', color='#ff7f0e', alpha=0.7)
    ax2.plot(gemv_sizes, concurrent_gemv_time, 'o--', linewidth=2, markersize=8, label='GEMV Concurrent', color='#1f77b4')
    ax2.plot(gemv_sizes, concurrent_gemm_time, 's--', linewidth=2, markersize=8, label='GEMM Concurrent', color='#ff7f0e')
    ax2.set_xlabel('GEMV Size (M×N)', fontsize=12)
    ax2.set_ylabel('Time (ms/iteration)', fontsize=12)
    ax2.set_title('Kernel Times (Isolated vs Concurrent)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'case2_fixed_gemm_vary_gemv.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    
    if show_plot:
        plt.show()
    plt.close()


def create_summary_plot(df_case1, df_case2, output_dir, show_plot=False):
    """
    Create a summary comparison plot showing slowdown patterns from both cases
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Slowdown Comparison: Both Cases', fontsize=16, fontweight='bold')
    
    # Case 1: Fixed GEMV, Varying GEMM
    ax1 = axes[0]
    gemm_sizes = df_case1['gemm_size'].values
    ax1.plot(gemm_sizes, df_case1['gemv_slowdown'].values, 'o-', linewidth=2, markersize=8, 
             label='GEMV (Fixed) Slowdown', color='#1f77b4')
    ax1.plot(gemm_sizes, df_case1['gemm_slowdown'].values, 's-', linewidth=2, markersize=8, 
             label='GEMM (Varying) Slowdown', color='#ff7f0e')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('GEMM Size (N×N×N)', fontsize=12)
    ax1.set_ylabel('Slowdown Factor', fontsize=12)
    ax1.set_title('Case 1: Fixed GEMV, Varying GEMM', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Case 2: Fixed GEMM, Varying GEMV
    ax2 = axes[1]
    gemv_sizes = df_case2['gemv_M'].values
    ax2.plot(gemv_sizes, df_case2['gemv_slowdown'].values, 'o-', linewidth=2, markersize=8, 
             label='GEMV (Varying) Slowdown', color='#1f77b4')
    ax2.plot(gemv_sizes, df_case2['gemm_slowdown'].values, 's-', linewidth=2, markersize=8, 
             label='GEMM (Fixed) Slowdown', color='#ff7f0e')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('GEMV Size (M×N)', fontsize=12)
    ax2.set_ylabel('Slowdown Factor', fontsize=12)
    ax2.set_title('Case 2: Fixed GEMM, Varying GEMV', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'summary_slowdown_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    
    if show_plot:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot size sweep experiment results')
    parser.add_argument('--input_dir', type=str, default='size_sweep_results',
                       help='Directory containing CSV result files (default: size_sweep_results)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: same as input_dir)')
    parser.add_argument('--show', action='store_true',
                       help='Display plots interactively')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read CSV files
    case1_csv = input_dir / 'case1_fixed_gemv_vary_gemm.csv'
    case2_csv = input_dir / 'case2_fixed_gemm_vary_gemv.csv'
    
    if not case1_csv.exists() or not case2_csv.exists():
        print(f"Error: CSV files not found in {input_dir}")
        print(f"Expected files:")
        print(f"  - {case1_csv}")
        print(f"  - {case2_csv}")
        return 1
    
    print(f"Reading results from: {input_dir}")
    df_case1 = read_csv_with_metadata(case1_csv)
    df_case2 = read_csv_with_metadata(case2_csv)
    
    print(f"\nCase 1 data points: {len(df_case1)}")
    print(f"Case 2 data points: {len(df_case2)}")
    
    # Generate plots
    print(f"\nGenerating plots to: {output_dir}")
    plot_case1(df_case1, output_dir, args.show)
    plot_case2(df_case2, output_dir, args.show)
    create_summary_plot(df_case1, df_case2, output_dir, args.show)
    
    print("\n" + "="*60)
    print("Plotting complete!")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Generated plots:")
    print(f"  - case1_fixed_gemv_vary_gemm.png")
    print(f"  - case2_fixed_gemm_vary_gemv.png")
    print(f"  - summary_slowdown_comparison.png")
    
    return 0


if __name__ == '__main__':
    exit(main())
