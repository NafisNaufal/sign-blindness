"""Figure style for IEEE two-column conference papers.

IEEE conference format is 8.5x11 with two 3.5in columns. Figures sized in inches here
so they drop into LaTeX at scale 1.0 with no resizing: rescaling in LaTeX is the usual
reason conference figures end up with mismatched font sizes.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL_WIDTH = 3.487      # inches, single IEEE column
FULL_WIDTH = 7.16      # inches, both columns

# Colour-blind-safe (Okabe-Ito). Reviewers print in greyscale more often than expected,
# so these are also chosen to separate by luminance.
C = {
    "oracle":  "#000000",
    "ours":    "#0072B2",
    "uniform": "#D55E00",
    "random":  "#999999",
    "none":    "#CC79A7",
    "accent":  "#009E73",
    "warn":    "#E69F00",
}


def use_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.2,
        "lines.markersize": 3,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": None,
        "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42,      # embed TrueType so the PDF is editable/searchable
        "ps.fonttype": 42,
    })


def save(fig, path, also_png=True):
    """Write PDF for LaTeX plus a PNG for quick viewing."""
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    if also_png:
        fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return path.with_suffix(".pdf")


def panel_label(ax, text, dx=-0.18, dy=1.04):
    ax.text(dx, dy, text, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")


def bootstrap_ci(x, stat=np.mean, n=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI. Every headline number in the paper gets one."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    boot = np.array([stat(rng.choice(x, len(x), replace=True)) for _ in range(n)])
    return stat(x), np.percentile(boot, 100 * alpha / 2), np.percentile(boot, 100 * (1 - alpha / 2))
