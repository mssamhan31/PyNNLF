from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import generate_supervisor_revision_outputs as pub
from publication_plot_style import PALETTE, apply_publication_style, save_figure


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"


def _load_netload(csv_name: str) -> pd.DataFrame:
    path = DATA_DIR / csv_name
    df = pd.read_csv(path)
    out = df[["datetime", "netload_kW"]].copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out["value"] = pd.to_numeric(out["netload_kW"], errors="coerce")
    out = out.dropna(subset=["datetime", "value"]).sort_values("datetime").reset_index(drop=True)
    return out[["datetime", "value"]]


def _hourly_distribution(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    d = df.copy()
    d["hour"] = d["datetime"].dt.hour
    d = d.dropna(subset=[value_col, "hour"])
    summary = (
        d.groupby("hour")[value_col]
        .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        .unstack()
        .rename(columns={0.10: "p10", 0.25: "p25", 0.50: "p50", 0.75: "p75", 0.90: "p90"})
        .reset_index()
    )
    all_hours = pd.DataFrame({"hour": list(range(24))})
    return all_hours.merge(summary, on="hour", how="left")


def _hourly_ramp_distribution(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    d = df.copy()
    d["ramp"] = d[value_col].diff()
    d = d.dropna(subset=["ramp"])
    d["hour"] = d["datetime"].dt.hour
    d = d.dropna(subset=["hour"])
    summary = (
        d.groupby("hour")["ramp"]
        .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        .unstack()
        .rename(columns={0.10: "p10", 0.25: "p25", 0.50: "p50", 0.75: "p75", 0.90: "p90"})
        .reset_index()
    )
    all_hours = pd.DataFrame({"hour": list(range(24))})
    return all_hours.merge(summary, on="hour", how="left")


def _plot_band_panel(ax, s: pd.DataFrame, ylabel: str, title: str, zero_line: bool = False) -> None:
    x = s["hour"].to_numpy(dtype=float)
    p10 = s["p10"].to_numpy(dtype=float)
    p25 = s["p25"].to_numpy(dtype=float)
    p50 = s["p50"].to_numpy(dtype=float)
    p75 = s["p75"].to_numpy(dtype=float)
    p90 = s["p90"].to_numpy(dtype=float)

    ax.fill_between(x, p10, p90, color=PALETTE["light_grey"], alpha=0.20, label="10-90 percentile")
    ax.fill_between(x, p25, p75, color=PALETTE["grey"], alpha=0.25, label="25-75 percentile")
    ax.plot(x, p50, color=PALETTE["dark_blue"], linewidth=2.0, label="Median")
    if zero_line:
        ax.axhline(0.0, color=PALETTE["orange"], linewidth=1.2, linestyle="--")

    ax.set_xlim(0, 23)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _prepare_sa_bess_week() -> tuple[pd.Timestamp, dict[str, pd.DataFrame], str]:
    recap_sa = pd.read_csv(RESULTS_DIR / "04_sa_bess_clean_44hh" / "sa_bess_44hh_fh8_combined_recap.csv")

    label_order = ["underlying_load", "net_load_with_pv", "net_load_with_pv_battery"]
    dataset_files = {
        "underlying_load": DATA_DIR / "ds22_sa_bess_44hh_pos_underlying_load_30min.csv",
        "net_load_with_pv": DATA_DIR / "ds23_sa_bess_44hh_pos_net_load_with_pv_30min.csv",
        "net_load_with_pv_battery": DATA_DIR / "ds24_sa_bess_44hh_pos_net_load_with_pv_battery_30min.csv",
    }

    actual_frames: dict[str, pd.DataFrame] = {}
    for lbl in label_order:
        d = pd.read_csv(dataset_files[lbl])
        d = d[["datetime", "netload_kW"]].copy()
        d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce")
        d["observation"] = pd.to_numeric(d["netload_kW"], errors="coerce")
        d = d.dropna(subset=["datetime", "observation"]).sort_values("datetime").reset_index(drop=True)
        actual_frames[lbl] = d[["datetime", "observation"]]

    frames: dict[str, pd.DataFrame] = {}
    for lbl in label_order:
        subset = recap_sa.loc[
            recap_sa["dataset_label"].astype(str).eq(lbl)
            & recap_sa["model_name"].astype(str).eq("m17_xgb_hp1")
            & recap_sa["forecast_horizon_min"].astype(int).eq(1440)
        ].sort_values(["exp_date", "experiment_no"])
        row = subset.iloc[-1]
        cv1 = pub._pick_cv1_file(str(row["experiment_folder"]))
        f = pub._read_forecast_frame(cv1)
        aligned = actual_frames[lbl].merge(f[["datetime", "forecast"]], on="datetime", how="inner")
        frames[lbl] = aligned[["datetime", "observation", "forecast"]].sort_values("datetime").reset_index(drop=True)

    return pub._select_composition_week(frames, points_per_day=48)


def _generate_combined_fig20_21() -> Path:
    week_start, week_frames, week_mode = _prepare_sa_bess_week()

    label_order = ["underlying_load", "net_load_with_pv", "net_load_with_pv_battery"]
    pretty = {
        "underlying_load": "Underlying load",
        "net_load_with_pv": "Net load with PV",
        "net_load_with_pv_battery": "Net load with PV and battery",
    }

    pv_generation_obs = (
        week_frames["underlying_load"]["observation"].to_numpy()
        - week_frames["net_load_with_pv"]["observation"].to_numpy()
    )
    battery_charging_obs = (
        week_frames["net_load_with_pv_battery"]["observation"].to_numpy()
        - week_frames["net_load_with_pv"]["observation"].to_numpy()
    )

    y_values = []
    for lbl in label_order:
        y_values.append(week_frames[lbl]["observation"].to_numpy())
        y_values.append(week_frames[lbl]["forecast"].to_numpy())
    y_values.extend([pv_generation_obs, battery_charging_obs])
    y_all = np.concatenate(y_values)
    y_min = float(np.nanmin(y_all))
    y_max = float(np.nanmax(y_all))
    y_pad = 0.04 * (y_max - y_min if y_max > y_min else 1.0)
    y_limits = (y_min - y_pad, y_max + y_pad)

    errors = {
        lbl: week_frames[lbl]["forecast"] - week_frames[lbl]["observation"]
        for lbl in label_order
    }
    roughness = {
        lbl: float(np.std(np.diff(week_frames[lbl]["observation"].to_numpy())))
        for lbl in label_order
    }
    max_abs_error = max(float(np.nanmax(np.abs(error.to_numpy(dtype=float)))) for error in errors.values())
    error_limit = max_abs_error * 1.05

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=False)
    ts_layout = [
        (axes[0, 0], "underlying_load", "(a)"),
        (axes[1, 0], "net_load_with_pv", "(b)"),
        (axes[0, 1], "net_load_with_pv_battery", "(c)"),
    ]

    for ax, lbl, panel_id in ts_layout:
        d = week_frames[lbl]
        x = d["datetime"]
        ax.set_axisbelow(True)
        ax.plot(x, d["observation"], color=PALETTE["dark_blue"], linewidth=1.6, label="Actual")
        ax.plot(x, d["forecast"], color=PALETTE["orange"], linewidth=1.2, label="Forecast")
        if lbl == "net_load_with_pv":
            ax.plot(
                x,
                pv_generation_obs,
                color=PALETTE["light_grey"],
                linewidth=1.1,
                linestyle="--",
                label="PV generation",
            )
        elif lbl == "net_load_with_pv_battery":
            ax.plot(
                x,
                battery_charging_obs,
                color=PALETTE["grey"],
                linewidth=1.1,
                linestyle=":",
                label="Battery net charge",
            )
        ax.set_title(f"{panel_id} {pretty[lbl]}")
        ax.set_ylabel("kW")
        ax.set_ylim(*y_limits)
        pub._format_datetime_axis(ax)
        pub._format_numeric_axis(ax)
        ax.text(
            0.01,
            -0.18,
            f"Roughness: {roughness[lbl]:.2f}",
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=13,
        )
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)

    ax_err = axes[1, 1]
    colors = {
        "underlying_load": PALETTE["dark_blue"],
        "net_load_with_pv": PALETTE["light_grey"],
        "net_load_with_pv_battery": PALETTE["orange"],
    }
    labels = {
        "underlying_load": "Underlying load",
        "net_load_with_pv": "Net load with PV",
        "net_load_with_pv_battery": "Net load with PV and battery",
    }
    for lbl in label_order:
        d = week_frames[lbl]
        ax_err.plot(d["datetime"], errors[lbl], color=colors[lbl], linewidth=1.4, label=labels[lbl])
    ax_err.axhline(0.0, color=PALETTE["grey"], linewidth=1.0, linestyle="--")
    ax_err.set_title("(d) Forecast error comparison across compositions")
    ax_err.set_ylabel("Error (kW)")
    ax_err.set_ylim(-error_limit, error_limit)
    pub._format_datetime_axis(ax_err)
    pub._format_numeric_axis(ax_err)
    ax_err.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)

    axes[1, 0].set_xlabel("Date")
    axes[1, 1].set_xlabel("Date")
    suffix = "(week where underlying is not the roughest)" if week_mode == "underlying_not_most_volatile" else "(best available representative week)"
    fig.suptitle(f"SA BESS composition: timeseries and error comparison {suffix}, from {week_start.date()}", y=1.01)
    fig.tight_layout(h_pad=3.8, w_pad=2.0)

    out_path = RESULTS_DIR / "04_sa_bess_clean_44hh" / "figures" / "fig20_sa_bess_composition_timeseries_error_combined.png"
    return save_figure(fig, out_path)


def _generate_combined_fig22_23() -> Path:
    underlying = _load_netload("ds22_sa_bess_44hh_pos_underlying_load_30min.csv")
    net_pv = _load_netload("ds23_sa_bess_44hh_pos_net_load_with_pv_30min.csv")
    net_pv_batt = _load_netload("ds24_sa_bess_44hh_pos_net_load_with_pv_battery_30min.csv")

    series = [
        ("Underlying load", underlying),
        ("Net load with PV", net_pv),
        ("Net load with PV and battery", net_pv_batt),
    ]

    load_summaries = [(name, _hourly_distribution(df, value_col="value")) for name, df in series]
    ramp_summaries = [(name, _hourly_ramp_distribution(df, value_col="value")) for name, df in series]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11.5), sharex=False)

    load_vals = np.concatenate([s[["p10", "p25", "p50", "p75", "p90"]].to_numpy().ravel() for _, s in load_summaries])
    load_vals = load_vals[np.isfinite(load_vals)]
    load_lim = max(abs(float(np.min(load_vals))), abs(float(np.max(load_vals)))) * 1.05

    ramp_vals = np.concatenate([s[["p10", "p25", "p50", "p75", "p90"]].to_numpy().ravel() for _, s in ramp_summaries])
    ramp_vals = ramp_vals[np.isfinite(ramp_vals)]
    ramp_lim = max(abs(float(np.min(ramp_vals))), abs(float(np.max(ramp_vals)))) * 1.05

    for i in range(3):
        load_name, load_s = load_summaries[i]
        ramp_name, ramp_s = ramp_summaries[i]

        ax_l = axes[0, i]
        row1_label = f"({chr(97 + i)}) {load_name}"
        _plot_band_panel(ax_l, load_s, "Load (kW)", "")
        ax_l.text(0.01, 0.98, row1_label, transform=ax_l.transAxes, va="top", ha="left", fontsize=20)
        ax_l.set_ylim(-load_lim, load_lim)
        ax_l.tick_params(axis="x", labelbottom=True)

        ax_r = axes[1, i]
        row2_label = f"({chr(100 + i)}) {ramp_name}"
        _plot_band_panel(ax_r, ramp_s, "Ramp (kW/30 min)", "", zero_line=True)
        ax_r.text(0.01, 0.98, row2_label, transform=ax_r.transAxes, va="top", ha="left", fontsize=20)
        ax_r.set_ylim(-ramp_lim, ramp_lim)

    for ax in axes[1, :]:
        ax.set_xlabel("Hour of day")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("SA BESS hourly distributions by composition", y=1.01)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93), h_pad=3.4, w_pad=1.8)

    top_row_top = max(ax.get_position().y1 for ax in axes[0, :])
    top_row_bottom = min(ax.get_position().y0 for ax in axes[0, :])
    bottom_row_top = max(ax.get_position().y1 for ax in axes[1, :])

    # Keep the load heading very close to row-1 while staying clear of panel labels.
    load_heading_y = min(0.92, top_row_top + 0.035)
    # Keep ramp heading lower in the inter-row gap to increase separation from row-1.
    ramp_heading_y = bottom_row_top + 0.025

    fig.text(
        0.5,
        load_heading_y,
        "Hourly load distribution (median and percentile bands)",
        ha="center",
        va="center",
        fontsize=20,
    )
    fig.text(
        0.5,
        ramp_heading_y,
        "Hourly ramp distribution (median and percentile bands)",
        ha="center",
        va="center",
        fontsize=20,
    )

    out_path = RESULTS_DIR / "04_sa_bess_clean_44hh" / "figures" / "fig21_sa_bess_hourly_load_ramp_combined.png"
    return save_figure(fig, out_path)


def _generate_combined_fig24_25() -> Path:
    underlying = _load_netload("ds22_sa_bess_44hh_pos_underlying_load_30min.csv").rename(columns={"value": "underlying"})
    net_pv = _load_netload("ds23_sa_bess_44hh_pos_net_load_with_pv_30min.csv").rename(columns={"value": "net_pv"})
    net_pv_batt = _load_netload("ds24_sa_bess_44hh_pos_net_load_with_pv_battery_30min.csv").rename(columns={"value": "net_pv_batt"})

    merged = underlying.merge(net_pv, on="datetime", how="inner").merge(net_pv_batt, on="datetime", how="inner")
    merged["pv_generation"] = merged["underlying"] - merged["net_pv"]
    merged["battery_net_charge"] = merged["net_pv_batt"] - merged["net_pv"]

    pv_summary = _hourly_distribution(merged[["datetime", "pv_generation"]].rename(columns={"pv_generation": "value"}))
    batt_summary = _hourly_distribution(merged[["datetime", "battery_net_charge"]].rename(columns={"battery_net_charge": "value"}))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharex=True)
    _plot_band_panel(
        axes[0],
        pv_summary,
        "PV generation (kW)",
        "(a) Hourly PV generation distribution",
        zero_line=True,
    )
    _plot_band_panel(
        axes[1],
        batt_summary,
        "Battery net charge (kW)",
        "(b) Hourly battery net charge distribution",
        zero_line=True,
    )
    axes[0].set_xlabel("Hour of day")
    axes[1].set_xlabel("Hour of day")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("SA BESS hourly PV generation and battery net charge distributions", y=1.04)
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    out_path = RESULTS_DIR / "04_sa_bess_clean_44hh" / "figures" / "fig22_sa_bess_pv_battery_distribution_combined.png"
    return save_figure(fig, out_path)


def main() -> None:
    apply_publication_style()

    outputs = [
        _generate_combined_fig20_21(),
        _generate_combined_fig22_23(),
        _generate_combined_fig24_25(),
    ]

    for p in outputs:
        print(f"generated {p}")


if __name__ == "__main__":
    main()
