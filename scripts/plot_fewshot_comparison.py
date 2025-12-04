#!/usr/bin/env python3
"""Generate few-shot comparison plot for the paper."""

import json
import matplotlib.pyplot as plt
import numpy as np

# Data from Experiment C (fair_comparison)
budgets = [1, 3, 5, 10, 20, 30]
gating = [12.1, 11.7, 4.2, 15.2, 9.0, -6.3]
policy_head = [3.5, 50.6, 155.7, -19.0, -58.1, -122.7]
full = [-50.4, 21.3, -10.4, -4.0, -78.0, 1.1]

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(budgets))
width = 0.25

bars1 = ax.bar(x - width, gating, width, label='Gating (ASTRAL)', color='#2E86AB', edgecolor='black')
bars2 = ax.bar(x, policy_head, width, label='Policy Head', color='#F6AE2D', edgecolor='black')
bars3 = ax.bar(x + width, full, width, label='Full Fine-Tune', color='#E94F37', edgecolor='black')

ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

ax.set_xlabel('Adaptation Episodes', fontsize=12)
ax.set_ylabel('Improvement over Baseline', fontsize=12)
ax.set_title('Few-Shot Adaptation: Gating vs Policy-Head vs Full', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(budgets)
ax.legend(loc='upper right')

# Add value labels on bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if abs(height) > 10:
            ax.annotate(f'{height:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height > 0 else -10),
                       textcoords="offset points",
                       ha='center', va='bottom' if height > 0 else 'top',
                       fontsize=8)

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

ax.set_ylim(-150, 180)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report/version_2/asset/figure/fewshot_comparison.png', dpi=150, bbox_inches='tight')
print("Saved to report/version_2/asset/figure/fewshot_comparison.png")
plt.close()

