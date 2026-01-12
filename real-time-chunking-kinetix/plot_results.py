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
