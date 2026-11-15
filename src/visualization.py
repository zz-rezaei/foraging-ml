"""
Shared plotting settings so that every figure in the project uses the
same colors, category order and style.

Colors come from a colorblind-checked categorical palette. Each color
always means the same thing in every figure:
  - groups:    Control = blue, AD = orange
  - behaviors: Eat = green, Carry = magenta, NosePoke = yellow, NoLeave = aqua
"""

from pathlib import Path

import matplotlib.pyplot as plt

GROUP_ORDER = ["Control", "AD"]
GROUP_COLORS = {"Control": "#2a78d6", "AD": "#eb6834"}

BEHAVIOR_ORDER = ["Eat", "Carry", "NosePoke", "NoLeave"]
BEHAVIOR_COLORS = {
    "Eat": "#008300",
    "Carry": "#e87ba4",
    "NosePoke": "#eda100",
    "NoLeave": "#1baf7a",
}

CONDITION_ORDER = ["short-soft", "long-soft", "short-hard", "long-hard"]
SIZE_ORDER = ["small", "large"]

NEUTRAL_DARK = "#52514e"   # mean markers, reference lines
NEUTRAL_LIGHT = "#d9d8d4"  # grid lines


def set_plot_style():
    """Apply one consistent, clean style to all matplotlib figures."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": NEUTRAL_LIGHT,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
    })


def save_figure(fig, name, folder="../results/figures"):
    """Save a figure as PNG. Call this BEFORE plt.show(); after show()
    the figure is closed and an empty image would be saved."""
    Path(folder).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(folder) / f"{name}.png", bbox_inches="tight")
