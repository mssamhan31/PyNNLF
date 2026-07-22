from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

PALETTE = {
    "orange": "#eb932c",
    "dark_blue": "#22303d",
    "grey": "#2F4D67",
    "light_grey": "#5C7D99",
    "light_white": "#ebe3e3",
}

MODEL_COLORS = {
    "m1_naive_hp1": PALETTE["grey"],
    "m6_lr_hp1": PALETTE["light_grey"],
    "m17_xgb_hp1": PALETTE["orange"],
}


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.18,
            "grid.color": "#c7d0d8",
            "axes.edgecolor": "#22303d",
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#c7d0d8",
        }
    )


def save_figure(fig, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
