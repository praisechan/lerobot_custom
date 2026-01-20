import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data
df = pd.read_csv('eval_output/results.csv')

# Create output directory
output_dir = Path('eval_output/plots')
output_dir.mkdir(exist_ok=True)

print(f"Loaded {len(df)} rows")
print(f"Methods: {df['method'].unique()}")
print(f"Delays: {sorted(df['delay'].unique())}")
print(f"Levels: {len(df['level'].unique())}")

# ============================================================================
# Figure 1: Average Success Rate vs Inference Delay (for each method)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Group by method and delay, average across all levels and horizons
method_delay_avg = df.groupby(['method', 'delay'])['returned_episode_solved'].mean().reset_index()

for method in df['method'].unique():
    method_data = method_delay_avg[method_delay_avg['method'] == method]
    ax.plot(method_data['delay'], method_data['returned_episode_solved'], 
            marker='o', linewidth=2, markersize=8, label=method)

ax.set_xlabel('Inference Delay (timesteps)', fontsize=12)
ax.set_ylabel('Average Success Rate', fontsize=12)
ax.set_title('RTC Method Performance vs Inference Delay\n(Averaged across all levels and execution horizons)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(sorted(df['delay'].unique()))

plt.tight_layout()
plt.savefig(output_dir / 'success_vs_delay.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'success_vs_delay.png'}")
plt.close()

# ============================================================================
# Figure 2: Heatmaps for each method (Delay x Execute Horizon)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

methods = df['method'].unique()
for idx, method in enumerate(methods):
    method_data = df[df['method'] == method]
    pivot_data = method_data.pivot_table(
        values='returned_episode_solved',
        index='execute_horizon',
        columns='delay',
        aggfunc='mean'
    )
    
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0, vmax=1, ax=axes[idx], cbar_kws={'label': 'Success Rate'})
    axes[idx].set_title(f'{method.upper()} Method', fontsize=13, fontweight='bold')
    axes[idx].set_xlabel('Inference Delay', fontsize=11)
    axes[idx].set_ylabel('Execute Horizon', fontsize=11)

plt.suptitle('Success Rate Heatmaps: Delay × Execute Horizon\n(Averaged across all levels)', 
             fontsize=15, fontweight='bold', y=1.0)
plt.tight_layout()
plt.savefig(output_dir / 'heatmaps_delay_horizon.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'heatmaps_delay_horizon.png'}")
plt.close()

# ============================================================================
# Figure 2.5: Line Graphs for Naive vs RTC (in milliseconds)
# ============================================================================
# Convert timesteps to milliseconds (1 timestep = 33ms)
TIMESTEP_MS = 33

# Filter for only naive and realtime methods
rtc_comparison_df = df[df['method'].isin(['naive', 'realtime'])].copy()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Graph 1: Success Rate vs Inference Delay (ms) - Fixed execute_horizon = 4
ax1 = axes[0]
FIXED_EXECUTE_HORIZON = 4
delay_comparison = rtc_comparison_df[rtc_comparison_df['execute_horizon'] == FIXED_EXECUTE_HORIZON].copy()
delay_comparison = delay_comparison.groupby(['method', 'delay'])['returned_episode_solved'].mean().reset_index()
delay_comparison['delay_ms'] = delay_comparison['delay'] * TIMESTEP_MS

for method in ['naive', 'realtime']:
    method_data = delay_comparison[delay_comparison['method'] == method]
    ax1.plot(method_data['delay_ms'], method_data['returned_episode_solved'], 
            marker='o', linewidth=2.5, markersize=10, label=method.upper())

ax1.set_xlabel('Inference Delay (ms)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Average Success Rate', fontsize=13, fontweight='bold')
ax1.set_title(f'Success Rate vs Inference Delay\n(Execute Horizon fixed at {FIXED_EXECUTE_HORIZON} = {FIXED_EXECUTE_HORIZON * TIMESTEP_MS}ms)', 
             fontsize=13, fontweight='bold')
ax1.legend(fontsize=12, loc='best')
ax1.grid(True, alpha=0.4, linestyle='--')
ax1.set_ylim([0.4, 1.0])

# Graph 2: Success Rate vs Execute Horizon (ms) - Fixed delay = 1
ax2 = axes[1]
FIXED_DELAY = 1
horizon_comparison = rtc_comparison_df[rtc_comparison_df['delay'] == FIXED_DELAY].copy()
horizon_comparison = horizon_comparison.groupby(['method', 'execute_horizon'])['returned_episode_solved'].mean().reset_index()
horizon_comparison['execute_horizon_ms'] = horizon_comparison['execute_horizon'] * TIMESTEP_MS

for method in ['naive', 'realtime']:
    method_data = horizon_comparison[horizon_comparison['method'] == method]
    ax2.plot(method_data['execute_horizon_ms'], method_data['returned_episode_solved'], 
            marker='s', linewidth=2.5, markersize=10, label=method.upper())

ax2.set_xlabel('Execute Horizon (ms)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Average Success Rate', fontsize=13, fontweight='bold')
ax2.set_title(f'Success Rate vs Execute Horizon\n(Inference Delay fixed at {FIXED_DELAY} = {FIXED_DELAY * TIMESTEP_MS}ms)', 
             fontsize=13, fontweight='bold')
ax2.legend(fontsize=12, loc='best')
ax2.grid(True, alpha=0.4, linestyle='--')
ax2.set_ylim([0.7, 1.0])

plt.tight_layout()
plt.savefig(output_dir / 'naive_vs_rtc_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'naive_vs_rtc_comparison.png'}")
plt.close()

# ============================================================================
# Figure 3: Per-Level Comparison (Bar plot)
# ============================================================================
# Average across delays and horizons for each method and level
level_method_avg = df.groupby(['level', 'method'])['returned_episode_solved'].mean().reset_index()

# Clean level names for readability
level_method_avg['level_name'] = level_method_avg['level'].str.replace('worlds/l/', '').str.replace('.json', '')

fig, ax = plt.subplots(figsize=(16, 8))

levels = sorted(level_method_avg['level_name'].unique())
x = np.arange(len(levels))
width = 0.2

for idx, method in enumerate(methods):
    method_data = level_method_avg[level_method_avg['method'] == method]
    method_data = method_data.sort_values('level_name')
    
    offset = width * (idx - len(methods)/2 + 0.5)
    ax.bar(x + offset, method_data['returned_episode_solved'], width, 
           label=method, alpha=0.8)

ax.set_xlabel('Kinetix Level', fontsize=12)
ax.set_ylabel('Average Success Rate', fontsize=12)
ax.set_title('Method Comparison Across Kinetix Levels\n(Averaged over all delays and execute horizons)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(levels, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig(output_dir / 'method_comparison_by_level.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'method_comparison_by_level.png'}")
plt.close()

# ============================================================================
# Figure 4: Best Configuration Analysis
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, method in enumerate(methods):
    method_data = df[df['method'] == method]
    
    # Find best configuration for each delay
    best_configs = method_data.groupby('delay').apply(
        lambda x: x.nlargest(1, 'returned_episode_solved')
    ).reset_index(drop=True)
    
    ax = axes[idx]
    bars = ax.bar(best_configs['delay'], best_configs['returned_episode_solved'], 
                  alpha=0.7, width=0.6)
    
    # Add execute_horizon labels on top of bars
    for i, (delay, horizon, success) in enumerate(zip(
        best_configs['delay'], 
        best_configs['execute_horizon'],
        best_configs['returned_episode_solved']
    )):
        ax.text(delay, success + 0.02, f'h={int(horizon)}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Inference Delay', fontsize=11)
    ax.set_ylabel('Best Success Rate', fontsize=11)
    ax.set_title(f'{method.upper()}: Best Execute Horizon per Delay', 
                 fontsize=12, fontweight='bold')
    ax.set_xticks(sorted(df['delay'].unique()))
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Optimal Execute Horizon Selection for Each Delay\n(h=horizon shown above bars)', 
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'best_configs_per_delay.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'best_configs_per_delay.png'}")
plt.close()

# ============================================================================
# Print Summary Statistics
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print("\nOverall Average Success Rate by Method:")
print(df.groupby('method')['returned_episode_solved'].mean().sort_values(ascending=False))

print("\nBest Configuration Overall (by success rate):")
best_overall = df.nlargest(5, 'returned_episode_solved')[
    ['method', 'delay', 'execute_horizon', 'returned_episode_solved', 'level']
]
print(best_overall.to_string(index=False))

print(f"\n✓ All plots saved to: {output_dir}/")
