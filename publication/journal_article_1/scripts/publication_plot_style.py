"""Shared matplotlib style for the journal article figures.

Every paper figure is drawn through this module so that colour, typography and
output resolution stay consistent across sections. Import ``PALETTE`` for series
colours, call :func:`apply_publication_style` once before plotting, and write the
result with :func:`save_figure`.

The colours are the Okabe-Ito qualitative palette, which stays distinguishable
under deuteranopia, protanopia and tritanopia, and separates in greyscale. The
earlier palette used three closely spaced blues that were hard to tell apart in
print; hue alone is still not enough where series overlap, so pair a colour with
an entry from :data:`LINESTYLES` whenever traces share an axis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Page sizing
# ---------------------------------------------------------------------------
#
# Figures are drawn at the width they occupy on the page, so a point size set
# here is the point size the reader sees. Previously figures were drawn 14-18
# inches wide and shrunk to fit, which scaled a nominal 15 pt label down to
# about 6 pt in print.
#
# PAGE_WIDTH_IN is the text width of a single-column A4 page with the
# manuscript's margins. Body text alongside is 11 pt.

PAGE_WIDTH_IN = 6.8

BASE_PT = 7.5
AXIS_LABEL_PT = 9.0
TICK_PT = 8.0
TITLE_PT = 9.5
ANNOTATION_PT = 6.5

# Ticks per axis. Fewer, larger, well-spaced labels read better once the figure
# is at its printed size.
TICK_BINS = 4


def page_figsize(aspect_width: float, aspect_height: float) -> tuple[float, float]:
    """Scale an authored aspect ratio to the printed page width.

    Call sites keep their original proportions, expressed as any two numbers in
    the intended ratio, and this returns the matching size in inches at
    :data:`PAGE_WIDTH_IN` wide.

    Args:
        aspect_width (float): the width of the intended aspect ratio.
        aspect_height (float): the height of the intended aspect ratio.

    Returns:
        tuple[float, float]: width and height in inches.
    """
    return (PAGE_WIDTH_IN, PAGE_WIDTH_IN * aspect_height / aspect_width)


# Okabe-Ito qualitative palette. Keys are semantic rather than descriptive of the
# hue, so a colour can be changed here without every call site reading falsely.
PALETTE = {
    "series_a": "#0072B2",  # blue
    "series_b": "#E69F00",  # orange
    "series_c": "#009E73",  # bluish green
    "series_d": "#CC79A7",  # reddish purple
    "series_e": "#D55E00",  # vermillion
    "neutral": "#6E7B85",  # grey, for reference lines and background points
    "fill": "#EBE3E3",  # pale fill, never a data colour
}

# Line styles paired with PALETTE entries so overlapping traces stay separable in
# greyscale and for readers who cannot distinguish the hues.
LINESTYLES = {
    "series_a": "-",
    "series_b": "-",
    "series_c": "--",
    "series_d": ":",
    "series_e": "-.",
}

MODEL_COLORS = {
    "m1_naive_hp1": PALETTE["series_a"],
    "m6_lr_hp1": PALETTE["series_c"],
    "m17_xgb_hp1": PALETTE["series_b"],
}

# Twelve-model categorical sequence, used by the grouped-bar and scatter figures
# that show the full model library. Twelve categories cannot all be mutually
# distinguishable under colour-blindness, so these figures also carry a direct
# label on every bar group or point; the ordering below keeps adjacent bars as
# far apart in hue and lightness as the set allows.
MODEL_SEQUENCE = [
    "m1_naive_hp1",
    "m2_snaive_hp2",
    "m3_ets_hp1",
    "m4_arima_hp1",
    "m6_lr_hp1",
    "m7_ann_hp1",
    "m8_dnn_hp1",
    "m9_rt_hp3",
    "m10_rf_hp1",
    "m13_lstm_hp2",
    "m16_prophet_hp1",
    "m17_xgb_hp1",
]

MODEL_SEQUENCE_COLORS = {
    "m1_naive_hp1": "#0072B2",  # blue
    "m2_snaive_hp2": "#56B4E9",  # sky blue
    "m3_ets_hp1": "#009E73",  # bluish green
    "m4_arima_hp1": "#8FD744",  # light green
    "m6_lr_hp1": "#E69F00",  # orange
    "m7_ann_hp1": "#F0E442",  # yellow
    "m8_dnn_hp1": "#D55E00",  # vermillion
    "m9_rt_hp3": "#CC79A7",  # reddish purple
    "m10_rf_hp1": "#7B3294",  # purple
    "m13_lstm_hp2": "#994F00",  # brown
    "m16_prophet_hp1": "#00539C",  # deep blue
    "m17_xgb_hp1": "#40B0A6",  # teal
}

# Short labels used on axes and legends, so figures do not carry the internal
# model numbering that means nothing to a reader.
MODEL_SHORT_LABELS = {
    "m1_naive_hp1": "naive_hp1",
    "m2_snaive_hp2": "snaive_hp2",
    "m3_ets_hp1": "ets_hp1",
    "m4_arima_hp1": "arima_hp1",
    "m6_lr_hp1": "lr_hp1",
    "m7_ann_hp1": "ann_hp1",
    "m8_dnn_hp1": "dnn_hp1",
    "m9_rt_hp3": "rt_hp3",
    "m10_rf_hp1": "rf_hp1",
    "m13_lstm_hp2": "lstm_hp2",
    "m16_prophet_hp1": "prophet_hp1",
    "m17_xgb_hp1": "xgb_hp1",
}


def apply_publication_style() -> None:
    """Set the matplotlib rcParams shared by every paper figure.

    Call once before building a figure. Safe to call repeatedly.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "Arial",
            "font.size": BASE_PT,
            "figure.titlesize": TITLE_PT + 1.0,
            "axes.titlesize": TITLE_PT,
            "axes.labelsize": AXIS_LABEL_PT,
            "xtick.labelsize": TICK_PT,
            "ytick.labelsize": TICK_PT,
            "legend.fontsize": TICK_PT,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.18,
            "grid.color": "#c7d0d8",
            "axes.edgecolor": "#22303d",
            "axes.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#c7d0d8",
        }
    )


def save_figure(fig, output_path: Path) -> Path:
    """Write a figure as a 300 dpi PNG and close it.

    Args:
        fig (matplotlib.figure.Figure): the figure to write.
        output_path (Path): destination path; parent directories are created.

    Returns:
        Path: the path written, so callers can collect it for the manifest.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
