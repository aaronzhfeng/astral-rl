#!/usr/bin/env python3
"""Generate ASTRAL vs Baseline training comparison plot."""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_tensorboard_data(log_dir):
    """Load scalar data from TensorBoard logs."""
    ea = EventAccumulator(log_dir)
    ea.Reload()
    
    data = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = (steps, values)
    return data

def smooth(values, weight=0.9):
    """Exponential moving average smoothing."""
    smoothed = []
    last = values[0]
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def main():
    results_dir = "results/runs"
    
    # Find ASTRAL and Baseline runs
    astral_dirs = sorted(glob.glob(f"{results_dir}/astral_astral_*"))
    baseline_dirs = sorted(glob.glob(f"{results_dir}/baseline_baseline_*"))
    
    print(f"Found {len(astral_dirs)} ASTRAL runs")
    print(f"Found {len(baseline_dirs)} Baseline runs")
    
    # Load data
    astral_data = []
    baseline_data = []
    
    for d in astral_dirs:
        try:
            data = load_tensorboard_data(d)
            if 'charts/mean_return' in data:
                astral_data.append(data['charts/mean_return'])
            elif 'train/episode_return' in data:
                astral_data.append(data['train/episode_return'])
        except Exception as e:
            print(f"Error loading {d}: {e}")
    
    for d in baseline_dirs:
        try:
            data = load_tensorboard_data(d)
            if 'charts/mean_return' in data:
                baseline_data.append(data['charts/mean_return'])
            elif 'train/episode_return' in data:
                baseline_data.append(data['train/episode_return'])
        except Exception as e:
            print(f"Error loading {d}: {e}")
    
    print(f"Loaded {len(astral_data)} ASTRAL, {len(baseline_data)} Baseline curves")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Learning curves
    ax1 = axes[0]
    
    colors = {'astral': '#2ecc71', 'baseline': '#e74c3c'}
    
    # Plot individual runs with transparency
    for steps, values in astral_data:
        ax1.plot(steps, smooth(values, 0.9), color=colors['astral'], alpha=0.3, linewidth=1)
    for steps, values in baseline_data:
        ax1.plot(steps, smooth(values, 0.9), color=colors['baseline'], alpha=0.3, linewidth=1)
    
    # Plot mean with thicker line
    if astral_data:
        # Interpolate to common x-axis
        max_steps = max(max(s) for s, v in astral_data)
        common_steps = np.linspace(0, max_steps, 200)
        astral_interp = []
        for steps, values in astral_data:
            interp_vals = np.interp(common_steps, steps, values)
            astral_interp.append(interp_vals)
        astral_mean = np.mean(astral_interp, axis=0)
        astral_std = np.std(astral_interp, axis=0)
        ax1.plot(common_steps, smooth(list(astral_mean), 0.9), color=colors['astral'], 
                linewidth=2.5, label=f'ASTRAL (n={len(astral_data)})')
    
    if baseline_data:
        max_steps = max(max(s) for s, v in baseline_data)
        common_steps = np.linspace(0, max_steps, 200)
        baseline_interp = []
        for steps, values in baseline_data:
            interp_vals = np.interp(common_steps, steps, values)
            baseline_interp.append(interp_vals)
        baseline_mean = np.mean(baseline_interp, axis=0)
        ax1.plot(common_steps, smooth(list(baseline_mean), 0.9), color=colors['baseline'], 
                linewidth=2.5, label=f'Baseline (n={len(baseline_data)})')
    
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Episode Return', fontsize=12)
    ax1.set_title('Learning Curves: ASTRAL vs Baseline', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 550)
    ax1.axhline(y=500, color='gray', linestyle='--', alpha=0.5, label='Max Return')
    
    # Plot 2: Final performance bar chart
    ax2 = axes[1]
    
    # Get final performance for each run
    astral_final = [values[-50:] for steps, values in astral_data]  # Last 50 episodes
    baseline_final = [values[-50:] for steps, values in baseline_data]
    
    astral_final_mean = np.mean([np.mean(v) for v in astral_final]) if astral_final else 0
    astral_final_std = np.std([np.mean(v) for v in astral_final]) if astral_final else 0
    baseline_final_mean = np.mean([np.mean(v) for v in baseline_final]) if baseline_final else 0
    baseline_final_std = np.std([np.mean(v) for v in baseline_final]) if baseline_final else 0
    
    x = [0, 1]
    heights = [astral_final_mean, baseline_final_mean]
    errors = [astral_final_std, baseline_final_std]
    bar_colors = [colors['astral'], colors['baseline']]
    
    bars = ax2.bar(x, heights, yerr=errors, capsize=8, color=bar_colors, 
                   edgecolor='black', linewidth=1.5, width=0.6)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(['ASTRAL\n(with Abstraction)', 'Baseline\n(GRU only)'], fontsize=12)
    ax2.set_ylabel('Final Episode Return', fontsize=12)
    ax2.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 550)
    ax2.axhline(y=500, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, height, err in zip(bars, heights, errors):
        ax2.text(bar.get_x() + bar.get_width()/2, height + err + 15, 
                f'{height:.1f}±{err:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add improvement annotation
    if baseline_final_mean > 0:
        improvement = (astral_final_mean - baseline_final_mean) / baseline_final_mean * 100
        ax2.annotate(f'+{improvement:.0f}%', xy=(0.5, max(heights)/2), fontsize=14, 
                    fontweight='bold', color='darkgreen', ha='center')
    
    plt.tight_layout()
    
    # Save
    output_path = "report/asset/figure/training_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved to {output_path}")
    
    # Also save to results for reference
    plt.savefig("results/training_comparison.png", dpi=150, bbox_inches='tight', facecolor='white')
    
    plt.show()

if __name__ == "__main__":
    main()

