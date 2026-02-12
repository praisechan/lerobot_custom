#!/usr/bin/env python3
"""
Plot read bandwidth vs. SM/TPC count from sm_bw_sweep results.

Usage:
    python3 plot.py --csv results.csv --out bandwidth_plot.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_bandwidth(csv_path, output_path, title=None, figsize=(10, 6)):
    """
    Generate a line plot of read bandwidth vs TPC count with error bars.
    
    Args:
        csv_path: Path to CSV file from sm_bw_sweep
        output_path: Output path for plot image
        title: Optional custom title
        figsize: Figure size tuple (width, height)
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Extract columns
    tpc_count = df['tpc_count'].values
    bw_mean = df['read_GBps_mean'].values
    bw_std = df['read_GBps_std'].values
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bandwidth with shaded error region
    ax.plot(tpc_count, bw_mean, 'o-', linewidth=2, markersize=4, 
            color='#2E86AB', label='Read Bandwidth')
    ax.fill_between(tpc_count, bw_mean - bw_std, bw_mean + bw_std,
                     alpha=0.3, color='#2E86AB', label='±1 Std Dev')
    
    # Formatting
    ax.set_xlabel('TPC Count', fontsize=12, fontweight='bold')
    ax.set_ylabel('Read Bandwidth (GB/s)', fontsize=12, fontweight='bold')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title('DRAM Read Bandwidth vs. Active TPC Count', 
                     fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    # Add text box with max bandwidth
    max_bw = bw_mean.max()
    max_bw_tpc = tpc_count[bw_mean.argmax()]
    
    # Find saturation point (where bandwidth reaches 95% of max)
    saturation_threshold = 0.95 * max_bw
    saturation_idx = np.where(bw_mean >= saturation_threshold)[0]
    saturation_tpc = tpc_count[saturation_idx[0]] if len(saturation_idx) > 0 else max_bw_tpc
    
    textstr = f'Peak BW: {max_bw:.1f} GB/s\n'
    textstr += f'Saturation: {saturation_tpc} TPCs\n'
    textstr += f'(≥{saturation_threshold:.1f} GB/s)'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"TPC Range: {tpc_count.min()} - {tpc_count.max()}")
    print(f"Bandwidth Range: {bw_mean.min():.2f} - {bw_mean.max():.2f} GB/s")
    print(f"Peak Bandwidth: {max_bw:.2f} GB/s at {max_bw_tpc} TPCs")
    print(f"Saturation Point: ~{saturation_tpc} TPCs (≥95% of peak)")
    
    # Calculate bandwidth efficiency vs TPC count
    bw_per_tpc = bw_mean / tpc_count
    print(f"\nBandwidth per TPC:")
    print(f"  Min: {bw_per_tpc.min():.2f} GB/s/TPC")
    print(f"  Max: {bw_per_tpc.max():.2f} GB/s/TPC (at {tpc_count[bw_per_tpc.argmax()]} TPCs)")
    print(f"  At Peak: {bw_mean.max() / max_bw_tpc:.2f} GB/s/TPC")


def plot_time_comparison(csv_path, output_path):
    """
    Plot execution time vs TPC count (inverse relationship).
    
    Args:
        csv_path: Path to CSV file
        output_path: Output path for plot
    """
    df = pd.read_csv(csv_path)
    
    tpc_count = df['tpc_count'].values
    time_mean = df['time_ms_mean'].values
    time_std = df['time_ms_std'].values
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(tpc_count, time_mean, 'o-', linewidth=2, markersize=4, color='#A23B72')
    ax.fill_between(tpc_count, time_mean - time_std, time_mean + time_std,
                     alpha=0.3, color='#A23B72')
    
    ax.set_xlabel('TPC Count', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Kernel Execution Time vs. Active TPC Count', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Time comparison plot saved to {output_path}")


def plot_dual_axis(csv_path, output_path):
    """
    Create a dual-axis plot with both bandwidth and time.
    
    Args:
        csv_path: Path to CSV file
        output_path: Output path for plot
    """
    df = pd.read_csv(csv_path)
    
    tpc_count = df['tpc_count'].values
    bw_mean = df['read_GBps_mean'].values
    time_mean = df['time_ms_mean'].values
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bandwidth on left axis
    color1 = '#2E86AB'
    ax1.set_xlabel('TPC Count', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Read Bandwidth (GB/s)', fontsize=12, fontweight='bold', color=color1)
    ax1.plot(tpc_count, bw_mean, 'o-', linewidth=2, markersize=4, color=color1, label='Bandwidth')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Time on right axis
    ax2 = ax1.twinx()
    color2 = '#A23B72'
    ax2.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold', color=color2)
    ax2.plot(tpc_count, time_mean, 's--', linewidth=2, markersize=4, color=color2, label='Time')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Title
    ax1.set_title('Bandwidth and Time vs. TPC Count', fontsize=14, fontweight='bold')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Dual-axis plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot bandwidth results from sm_bw_sweep',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 plot.py --csv results.csv --out bandwidth.png
  python3 plot.py --csv results.csv --out bw.png --title "H100 Read Bandwidth"
  python3 plot.py --csv results.csv --out time.png --plot-type time
  python3 plot.py --csv results.csv --out dual.png --plot-type dual
        """
    )
    
    parser.add_argument('--csv', type=str, required=True,
                        help='Input CSV file from sm_bw_sweep')
    parser.add_argument('--out', type=str, required=True,
                        help='Output plot file path (e.g., plot.png)')
    parser.add_argument('--title', type=str, default=None,
                        help='Custom plot title')
    parser.add_argument('--plot-type', type=str, default='bandwidth',
                        choices=['bandwidth', 'time', 'dual'],
                        help='Type of plot to generate (default: bandwidth)')
    parser.add_argument('--figsize', type=str, default='10,6',
                        help='Figure size as "width,height" (default: 10,6)')
    
    args = parser.parse_args()
    
    # Check if CSV exists
    if not Path(args.csv).exists():
        print(f"Error: CSV file not found: {args.csv}")
        return 1
    
    # Parse figsize
    try:
        figsize = tuple(map(float, args.figsize.split(',')))
        if len(figsize) != 2:
            raise ValueError
    except ValueError:
        print("Error: --figsize must be 'width,height' (e.g., '10,6')")
        return 1
    
    # Generate requested plot
    if args.plot_type == 'bandwidth':
        plot_bandwidth(args.csv, args.out, title=args.title, figsize=figsize)
    elif args.plot_type == 'time':
        plot_time_comparison(args.csv, args.out)
    elif args.plot_type == 'dual':
        plot_dual_axis(args.csv, args.out)
    
    return 0


if __name__ == '__main__':
    exit(main())
