from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
import generate_supervisor_revision_outputs as pub
from publication_plot_style import PALETTE, save_figure

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

recap = pd.read_csv(ROOT / "experiment_result" / "a1_experiment_result.csv")
rows = {}
for ds_no, label in [("ds20", "ASHD"), ("ds11", "AEDP")]:
    subset = recap.loc[
        recap["dataset_no"].astype(str).eq(ds_no)
        & pd.to_numeric(recap["forecast_horizon_min"], errors="coerce").eq(1440)
        & recap["model_name"].astype(str).eq("m17_xgb_hp1")
    ].sort_values(["exp_date", "experiment_no"])
    if subset.empty:
        raise ValueError(f"Missing {ds_no} row")
    rows[label] = subset.iloc[-1]

ashd_path = pub._pick_cv1_file(str(rows["ASHD"]["experiment_folder"]))
aedp_path = pub._pick_cv1_file(str(rows["AEDP"]["experiment_folder"]))
ashd_df = pub._read_forecast_frame(ashd_path).sort_values("datetime").reset_index(drop=True)
aedp_df = pub._read_forecast_frame(aedp_path).sort_values("datetime").reset_index(drop=True)

ashd_peak = float(
    pd.to_numeric(ashd_df["observation"], errors="coerce")
    .loc[pd.to_numeric(ashd_df["observation"], errors="coerce").gt(0)]
    .max()
)
aedp_peak = float(
    pd.to_numeric(aedp_df["observation"], errors="coerce")
    .loc[pd.to_numeric(aedp_df["observation"], errors="coerce").gt(0)]
    .max()
)

# Exact same calendar positions as AEDP reference window: 2024-02-04 to 2024-02-11
# We keep the same week length, but shift the ASHD window to the same month/day positions in 2013.
aedp_window = (pd.Timestamp("2024-02-04"), pd.Timestamp("2024-02-11"))
aedp_week = aedp_df.loc[
    (aedp_df["datetime"] >= aedp_window[0]) & (aedp_df["datetime"] < aedp_window[1])
].copy().reset_index(drop=True)

ashd_start = pd.Timestamp("2013-02-04")
ashd_end = ashd_start + pd.Timedelta(days=7)
ashd_week = ashd_df.loc[
    (ashd_df["datetime"] >= ashd_start) & (ashd_df["datetime"] < ashd_end)
].copy().reset_index(drop=True)

if len(ashd_week) < 336:
    raise ValueError(f"ASHD alt4 window too short: {len(ashd_week)} points")

limits = (-1.3, 1.3)
yticks = [-1.0, -0.5, 0.0, 0.5, 1.0]

out_dir = RESULTS_DIR / "01_ashd_aedp_148hh_comparison" / "figures" / "fig30_same_season_alternatives"
out_dir.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(2, 1, figsize=(14, 11), sharex=False, sharey=True)
for ax, label, week_df, peak in zip(axes, ["ASHD", "AEDP"], [ashd_week, aedp_week], [ashd_peak, aedp_peak]):
    ax.set_axisbelow(True)
    ax.plot(week_df["datetime"], week_df["observation"] / peak, color=PALETTE["dark_blue"], linewidth=1.8, label="Actual")
    ax.plot(week_df["datetime"], week_df["forecast"] / peak, color=PALETTE["orange"], linewidth=1.4, label="Forecast")
    ax.axhline(1.0, color=PALETTE["grey"], linewidth=1.1, linestyle="--", label=f"CV1 actual peak: {peak:.2f} kW")
    label_name = "ASHD" if label == "ASHD" else "AEDP"
    ax.set_title(f'{label_name} - representative week from {week_df["datetime"].dt.date.iloc[0]}')
    ax.set_ylabel("Load / positive peak")
    ax.set_ylim(*limits)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=3)
    pub._format_datetime_axis(ax)
    pub._format_numeric_axis(ax, y=False)
    ax.set_yticks(yticks)
axes[-1].set_xlabel("Date")
fig.suptitle("Alt4: same calendar positions, peak-normalized shared scale", y=1.02)
fig.tight_layout(h_pad=3.0)
out_path = out_dir / "fig30_alt4_same_calendar_positions.png"
save_figure(fig, out_path)
print(f"generated {out_path}")
