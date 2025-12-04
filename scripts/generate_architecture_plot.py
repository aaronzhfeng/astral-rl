#!/usr/bin/env python3
"""
Generate ASTRAL Architecture Diagram.
This script creates both horizontal and vertical versions of the architecture.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Style setup
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

OUTPUT_DIR = Path("report/version_2/asset/figure")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Colors
COLORS = {
    'slot0': '#E76F51',   # Coral
    'slot1': '#2A9D8F',   # Teal
    'slot2': '#E9C46A',   # Gold
}
FROZEN = '#E3F2FD'        # Light blue
ADAPT = '#C8E6C9'         # Light green
BORDER = '#37474F'        # Dark gray
MATH_BG = '#F5F5F5'       # Light gray for math boxes
BANK_BG = '#FFF8E1'       # Light yellow for abstraction bank

# Standard sizes
BOX_W = 1.6
BOX_H = 0.9
MATH_H = 0.55
ARROW_GAP = 0.4


def draw_box(ax, x, y, w, h, label, color, fontsize=14, zorder=1):
    """Draw a rounded box with label."""
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, 
        boxstyle='round,pad=0.02',
        facecolor=color, 
        edgecolor=BORDER, 
        linewidth=2,
        zorder=zorder
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', 
           fontsize=fontsize, fontweight='bold', zorder=zorder+1)


def draw_math_box(ax, x, y, w, h, math_text, fontsize=14, zorder=1):
    """Draw a dashed math annotation box."""
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, 
        boxstyle='round,pad=0.02',
        facecolor=MATH_BG, 
        edgecolor='#888', 
        linewidth=1.5, 
        linestyle='--',
        zorder=zorder
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, math_text, ha='center', va='center', 
           fontsize=fontsize, fontweight='bold', family='serif', style='italic',
           zorder=zorder+1)


def draw_arrow(ax, x1, y1, x2, y2, style='-', color=BORDER, zorder=0):
    """Draw an arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
               arrowprops=dict(arrowstyle='->', color=color, lw=2, linestyle=style),
               zorder=zorder)


def generate_horizontal():
    """Generate horizontal architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(-0.2, 14)
    ax.set_ylim(0.3, 5.7)
    ax.axis('off')
    
    mid_y = 3.0
    
    # ==================== 1. Input ====================
    draw_box(ax, 0.2, mid_y - BOX_H/2, BOX_W, BOX_H, 'Input', FROZEN, fontsize=15)
    draw_math_box(ax, 0.2, mid_y - BOX_H/2 - 0.75, BOX_W, MATH_H, '[o,a,r]', fontsize=13)
    draw_arrow(ax, 0.2 + BOX_W, mid_y, 0.2 + BOX_W + ARROW_GAP, mid_y)
    
    # ==================== 2. GRU ====================
    gru_x = 2.2
    draw_box(ax, gru_x, mid_y - BOX_H/2, BOX_W, BOX_H, 'GRU', FROZEN, fontsize=15)
    draw_math_box(ax, gru_x, mid_y - BOX_H/2 - 0.75, BOX_W, MATH_H, 'h_t', fontsize=15)
    draw_arrow(ax, gru_x + BOX_W, mid_y, gru_x + BOX_W + ARROW_GAP, mid_y)
    
    # ==================== 3. Abstraction Bank ====================
    bank_x = 4.2
    bank_w = 5.6
    
    # Bank container
    bank_rect = mpatches.FancyBboxPatch(
        (bank_x, 0.5), bank_w, 4.8, 
        boxstyle='round,pad=0.03',
        facecolor=BANK_BG, 
        edgecolor=BORDER, 
        linewidth=2.5,
        zorder=0
    )
    ax.add_patch(bank_rect)
    ax.text(bank_x + bank_w/2, 5.05, 'Abstraction Bank', 
           ha='center', fontsize=18, fontweight='bold')
    
    # Gating Network (green, adapted)
    gating_w = 2.0
    gating_x = bank_x + 0.2
    gating_y = mid_y - BOX_H/2
    draw_box(ax, gating_x, gating_y, gating_w, BOX_H, 'Gating Network', ADAPT, fontsize=13)
    ax.text(gating_x + gating_w/2, gating_y + BOX_H + 0.15, '~4.3k', 
           ha='center', fontsize=11, color='#2E7D32', fontweight='bold')
    
    # Weight vector math box
    draw_math_box(ax, gating_x, gating_y - 0.75, gating_w, MATH_H, 'w=[w₀,w₁,w₂]', fontsize=13)
    
    # Circle (weighted combination)
    circle_r = 0.4
    circle_x = bank_x + bank_w/2
    circle_y = mid_y
    circle = plt.Circle((circle_x, circle_y), circle_r, 
                        facecolor='white', edgecolor=BORDER, linewidth=2, zorder=2)
    ax.add_patch(circle)
    ax.text(circle_x, circle_y, '⊗', ha='center', va='center', 
           fontsize=22, fontweight='bold', zorder=3)
    
    # Summation label on TOP of circle
    ax.text(circle_x, circle_y + circle_r + 0.3, 'Σ wₖEₖ', 
           ha='center', fontsize=16, fontweight='bold', style='italic', color='#333')
    
    # Arrow from Gating to circle
    draw_arrow(ax, gating_x + gating_w, mid_y, circle_x - circle_r, mid_y)
    
    # Learnable Embeddings label (stacked vertically)
    slot_x = bank_x + 4.0
    ax.text(slot_x + 0.65, 4.65, 'Learnable', 
           ha='center', fontsize=13, color='#444', fontweight='bold')
    ax.text(slot_x + 0.65, 4.35, 'Embeddings', 
           ha='center', fontsize=13, color='#444', fontweight='bold')
    
    # Slot boxes
    slot_w = 1.3
    slot_h = 0.65
    slot_spacing = 0.85
    
    e1_y = circle_y - slot_h/2
    e0_y = e1_y + slot_spacing
    e2_y = e1_y - slot_spacing
    
    for name, color, sy in [('E₀', COLORS['slot0'], e0_y), 
                             ('E₁', COLORS['slot1'], e1_y), 
                             ('E₂', COLORS['slot2'], e2_y)]:
        slot_rect = mpatches.FancyBboxPatch(
            (slot_x, sy), slot_w, slot_h, 
            boxstyle='round,pad=0.02',
            facecolor=color, 
            edgecolor=BORDER, 
            linewidth=1.5, 
            zorder=2
        )
        ax.add_patch(slot_rect)
        ax.text(slot_x + slot_w/2, sy + slot_h/2, name, 
               ha='center', va='center', fontsize=15, fontweight='bold', zorder=3)
    
    # Arrows from slots to circle
    for sy in [e0_y, e1_y, e2_y]:
        draw_arrow(ax, slot_x, sy + slot_h/2, circle_x + circle_r, circle_y, zorder=1)
    
    # Output z
    z_x = circle_x - 0.45
    z_y = 0.7
    draw_math_box(ax, z_x, z_y, 0.9, 0.6, 'z', fontsize=16)
    draw_arrow(ax, circle_x, circle_y - circle_r, circle_x, z_y + 0.6)
    
    # Arrow from bank to FiLM (no arrow from circle)
    draw_arrow(ax, bank_x + bank_w, mid_y, bank_x + bank_w + ARROW_GAP, mid_y)
    
    # ==================== 4. FiLM ====================
    film_x = bank_x + bank_w + ARROW_GAP
    draw_box(ax, film_x, mid_y - BOX_H/2, BOX_W, BOX_H, 'FiLM', FROZEN, fontsize=15)
    draw_math_box(ax, film_x - 0.1, mid_y - BOX_H/2 - 0.75, BOX_W + 0.2, MATH_H, 'γ⊙h_t+β', fontsize=13)
    
    draw_arrow(ax, film_x + BOX_W, mid_y + 0.2, film_x + BOX_W + ARROW_GAP, mid_y + 0.7)
    draw_arrow(ax, film_x + BOX_W, mid_y - 0.2, film_x + BOX_W + ARROW_GAP, mid_y - 0.7)
    
    # ==================== 5. Policy Head ====================
    head_x = film_x + BOX_W + ARROW_GAP
    draw_box(ax, head_x, mid_y + 0.35, BOX_W, BOX_H, 'Policy', FROZEN, fontsize=15)
    draw_math_box(ax, head_x, mid_y + 0.35 + BOX_H + 0.1, BOX_W, MATH_H, 'π(a|s)', fontsize=14)
    
    # ==================== 6. Value Head ====================
    draw_box(ax, head_x, mid_y - 0.35 - BOX_H, BOX_W, BOX_H, 'Value', FROZEN, fontsize=15)
    draw_math_box(ax, head_x, mid_y - 0.35 - BOX_H - 0.65, BOX_W, MATH_H, 'V(s)', fontsize=14)
    
    # ==================== Legend (top center, above the figure) ====================
    legend_elements = [
        mpatches.Patch(facecolor=FROZEN, edgecolor=BORDER, label='Frozen at test time'),
        mpatches.Patch(facecolor=ADAPT, edgecolor=BORDER, label='Adapted at test time'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1.02),
              framealpha=0.95, fontsize=12, ncol=1,
              handlelength=1.5, handleheight=1.2, borderpad=0.5)
    
    plt.tight_layout(pad=0.1)
    plt.savefig(OUTPUT_DIR / 'architecture_horizontal.png', bbox_inches='tight', pad_inches=0.05, dpi=150, facecolor='white')
    plt.savefig(OUTPUT_DIR / 'architecture.png', bbox_inches='tight', pad_inches=0.05, dpi=150, facecolor='white')
    plt.close()
    print("✓ Horizontal architecture saved")


def generate_vertical():
    """Generate vertical architecture diagram."""
    fig, ax = plt.subplots(figsize=(6, 9))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 11)
    ax.axis('off')
    
    def varrow(x1, y1, x2, y2, style='-'):
        draw_arrow(ax, x1, y1, x2, y2, style)
    
    # 1. Input at top
    draw_box(ax, 1.5, 9.8, 3, 0.6, 'Input Projection', FROZEN)
    draw_math_box(ax, 1.8, 9.25, 2.4, 0.45, '[obs, a, r]', fontsize=9)
    varrow(3, 9.25, 3, 8.95)
    
    # 2. GRU
    draw_box(ax, 1.5, 8.1, 3, 0.7, 'GRU Backbone', FROZEN)
    draw_math_box(ax, 2.0, 7.5, 2.0, 0.5, 'h_t', fontsize=11)
    varrow(3, 7.5, 3, 7.2)
    
    # 3. Abstraction Bank
    bank_rect = mpatches.FancyBboxPatch((0.3, 2.8), 5.4, 4.3, boxstyle='round,pad=0.03',
                                        facecolor=BANK_BG, edgecolor=BORDER, linewidth=2)
    ax.add_patch(bank_rect)
    ax.text(3, 6.85, 'Abstraction Bank', ha='center', fontsize=11, fontweight='bold')
    
    # Gating Network
    draw_box(ax, 0.5, 4.6, 1.8, 1.2, 'Gating', ADAPT, fontsize=9)
    ax.text(1.4, 4.85, 'Network', ha='center', fontsize=9, fontweight='bold')
    ax.text(1.4, 5.55, '~4.3k', ha='center', fontsize=8, color='#2E7D32', fontweight='bold')
    
    draw_math_box(ax, 0.55, 3.55, 1.7, 0.55, 'w = [w₀,w₁,w₂]', fontsize=8)
    
    # Slots
    ax.text(4.5, 6.35, 'Learnable', ha='center', fontsize=8, color='#666', fontweight='bold')
    ax.text(4.5, 6.1, 'Embeddings', ha='center', fontsize=8, color='#666', fontweight='bold')
    
    for i, (name, color) in enumerate([('E₀', COLORS['slot0']), 
                                        ('E₁', COLORS['slot1']), 
                                        ('E₂', COLORS['slot2'])]):
        slot_rect = mpatches.FancyBboxPatch((3.8, 5.5 - i*0.6), 1.4, 0.45, 
                                            boxstyle='round,pad=0.02',
                                            facecolor=color, edgecolor=BORDER, 
                                            linewidth=1.2)
        ax.add_patch(slot_rect)
        ax.text(4.5, 5.72 - i*0.6, name, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Circle
    circle = plt.Circle((3, 4.5), 0.35, facecolor='white', edgecolor=BORDER, linewidth=1.5)
    ax.add_patch(circle)
    ax.text(3, 4.5, '⊗', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(3, 4.95, 'Σ wₖEₖ', ha='center', fontsize=10, fontweight='bold', style='italic', color='#333')
    
    # Arrows
    varrow(2.25, 3.82, 2.7, 4.25, style='--')
    for i in range(3):
        ax.annotate('', xy=(3.35, 4.5), xytext=(3.8, 5.72 - i*0.6),
                   arrowprops=dict(arrowstyle='->', color=BORDER, lw=1))
    
    # Output z
    draw_math_box(ax, 2.4, 3.0, 1.2, 0.5, 'z', fontsize=12)
    varrow(3, 4.15, 3, 3.55)
    
    varrow(3, 2.8, 3, 2.5)
    
    # FiLM
    draw_box(ax, 1.2, 1.7, 3.6, 0.7, 'FiLM Modulation', FROZEN)
    draw_math_box(ax, 1.5, 1.1, 3.0, 0.5, 'γ ⊙ h_t + β', fontsize=11)
    
    varrow(2.0, 1.1, 1.3, 0.7)
    varrow(4.0, 1.1, 4.7, 0.7)
    
    # Heads
    draw_box(ax, 0.3, 0.05, 2.0, 0.6, 'Policy Head', FROZEN, fontsize=9)
    draw_box(ax, 3.7, 0.05, 2.0, 0.6, 'Value Head', FROZEN, fontsize=9)
    draw_math_box(ax, 0.5, -0.55, 1.6, 0.45, 'π(a|s)', fontsize=10)
    draw_math_box(ax, 3.9, -0.55, 1.6, 0.45, 'V(s)', fontsize=10)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=FROZEN, edgecolor=BORDER, label='Frozen at test time'),
        mpatches.Patch(facecolor=ADAPT, edgecolor=BORDER, label='Adapted at test time'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'architecture_vertical.png', bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()
    print("✓ Vertical architecture saved")


def main():
    print("\n" + "="*50)
    print("Generating ASTRAL Architecture Diagrams")
    print("="*50 + "\n")
    
    generate_horizontal()
    generate_vertical()
    
    print("\n" + "="*50)
    print(f"Saved to: {OUTPUT_DIR}")
    print("  - architecture.png (horizontal, main)")
    print("  - architecture_horizontal.png")
    print("  - architecture_vertical.png")
    print("="*50 + "\n")


if __name__ == '__main__':
    main()

