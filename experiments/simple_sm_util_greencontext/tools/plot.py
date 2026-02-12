#!/usr/bin/env python3
"""
Plot bandwidth vs SM count from CSV results

Usage:
    python3 plot.py results.csv [output.png]
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_bandwidth_sweep(csv_path, output_path=None):
    """
    Plot bandwidth vs SM count with error bars
    
    Args:
        csv_path: Path to CSV file with columns (num_sms, mean_bw_gb_s, stdev_bw_gb_s)
        output_path: Path to save plot (default: replace .csv with .png)
    """
    # Read CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    
    # Validate columns
    required_cols = ['num_sms', 'mean_bw_gb_s', 'stdev_bw_gb_s']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV must contain columns: {required_cols}")
        print(f"Found columns: {df.columns.tolist()}")
        sys.exit(1)
    
    # Sort by SM count
    df = df.sort_values('num_sms')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot with error bars
    ax.errorbar(
        df['num_sms'],
        df['mean_bw_gb_s'],
        yerr=df['stdev_bw_gb_s'],
        marker='o',
        markersize=6,
        linestyle='-',
        linewidth=2,
        capsize=5,
        capthick=2,
        label='Read Bandwidth',
        color='#2E86AB',
        ecolor='#A23B72'
    )
    
    # Find saturation point (where derivative drops below threshold)
    if len(df) > 2:
        derivatives = np.diff(df['mean_bw_gb_s'].values) / np.diff(df['num_sms'].values)
        # Saturation: where slope drops to < 20% of initial slope
        initial_slope = np.mean(derivatives[:3]) if len(derivatives) >= 3 else derivatives[0]
        saturation_threshold = 0.2 * initial_slope
        
        for i, deriv in enumerate(derivatives):
            if deriv < saturation_threshold:
                saturation_idx = i + 1  # +1 because diff reduces length by 1
                ax.axvline(
                    df['num_sms'].iloc[saturation_idx],
                    color='red',
                    linestyle='--',
                    linewidth=2,
                    alpha=0.7,
                    label=f'Saturation (~{df["num_sms"].iloc[saturation_idx]} SMs)'
                )
                break
    
    # Labels and title
    ax.set_xlabel('Number of SMs', fontsize=14, fontweight='bold')
    ax.set_ylabel('Read Bandwidth (GB/s)', fontsize=14, fontweight='bold')
    ax.set_title('DRAM Read Bandwidth vs SM Count\n(CUDA Green Contexts Partitioning)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    
    # Formatting
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Tight layout
    plt.tight_layout()
    
    # Determine output path
    if output_path is None:
        output_path = csv_path.replace('.csv', '.png')
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Show summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    print(f"SM Count Range: {df['num_sms'].min()} - {df['num_sms'].max()}")
    print(f"Bandwidth Range: {df['mean_bw_gb_s'].min():.2f} - {df['mean_bw_gb_s'].max():.2f} GB/s")
    print(f"Peak Bandwidth: {df['mean_bw_gb_s'].max():.2f} ± {df.loc[df['mean_bw_gb_s'].idxmax(), 'stdev_bw_gb_s']:.2f} GB/s")
    print(f"Peak at: {df.loc[df['mean_bw_gb_s'].idxmax(), 'num_sms']} SMs")
    
    # Efficiency: bandwidth per SM at different points
    df['bw_per_sm'] = df['mean_bw_gb_s'] / df['num_sms']
    print(f"\nBandwidth per SM:")
    print(f"  At min SMs: {df.iloc[0]['bw_per_sm']:.2f} GB/s/SM")
    print(f"  At max SMs: {df.iloc[-1]['bw_per_sm']:.2f} GB/s/SM")
    print(f"  Peak efficiency: {df['bw_per_sm'].max():.2f} GB/s/SM at {df.loc[df['bw_per_sm'].idxmax(), 'num_sms']} SMs")
    print("="*60)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot.py <results.csv> [output.png]")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    plot_bandwidth_sweep(csv_path, output_path)

if __name__ == '__main__':
    main()
