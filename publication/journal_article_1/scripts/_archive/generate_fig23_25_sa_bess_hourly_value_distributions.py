from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from publication_plot_style import PALETTE, apply_publication_style, save_figure


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def _load_netload(csv_name: str) -> pd.DataFrame:
    path = DATA_DIR / csv_name
    df = pd.read_csv(path)
    if "datetime" not in df.columns or "netload_kW" not in df.columns:
        raise ValueError(f"Missing datetime/netload_kW in {path}")

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
    summary = all_hours.merge(summary, on="hour", how="left")
    return summary


def _plot_distribution_panel(ax, s: pd.DataFrame, title: str, ylabel: str) -> None:
    x = s["hour"].to_numpy(dtype=float)
    p10 = s["p10"].to_numpy(dtype=float)
    p25 = s["p25"].to_numpy(dtype=float)
    p50 = s["p50"].to_numpy(dtype=float)
    p75 = s["p75"].to_numpy(dtype=float)
    p90 = s["p90"].to_numpy(dtype=float)

    ax.fill_between(x, p10, p90, color=PALETTE["light_grey"], alpha=0.20, label="10-90 percentile")
    ax.fill_between(x, p25, p75, color=PALETTE["grey"], alpha=0.25, label="25-75 percentile")
    ax.plot(x, p50, color=PALETTE["dark_blue"], linewidth=2.0, label="Median")
    ax.set_xlim(0, 23)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 23])


def main() -> None:
    apply_publication_style()

    underlying = _load_netload("ds22_sa_bess_44hh_pos_underlying_load_30min.csv")
    net_pv = _load_netload("ds23_sa_bess_44hh_pos_net_load_with_pv_30min.csv")
    net_pv_batt = _load_netload("ds24_sa_bess_44hh_pos_net_load_with_pv_battery_30min.csv")

    merged = underlying.rename(columns={"value": "underlying"})
    merged = merged.merge(net_pv.rename(columns={"value": "net_pv"}), on="datetime", how="inner")
    merged = merged.merge(net_pv_batt.rename(columns={"value": "net_pv_batt"}), on="datetime", how="inner")

    merged["pv"] = merged["underlying"] - merged["net_pv"]
    merged["battery"] = merged["net_pv_batt"] - merged["net_pv"]

    actual_summaries = [
        ("(a) Underlying load", _hourly_distribution(merged[["datetime", "underlying"]].rename(columns={"underlying": "value"}))),
        ("(b) Net load with PV", _hourly_distribution(merged[["datetime", "net_pv"]].rename(columns={"net_pv": "value"}))),
        ("(c) Net load with PV and battery", _hourly_distribution(merged[["datetime", "net_pv_batt"]].rename(columns={"net_pv_batt": "value"}))),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    for ax, (title, summary) in zip(axes, actual_summaries):
        _plot_distribution_panel(ax, summary, title, "Load (kW)")
        ax.legend(loc="upper right", ncol=1)
    axes[-1].set_xlabel("Hour of day")
    fig.suptitle("SA BESS hourly load distributions (median and percentile bands)", y=1.01)
    fig.tight_layout()
    out1 = RESULTS_DIR / "04_sa_bess_clean_44hh" / "figures" / "fig23_sa_bess_hourly_load_distribution.png"
    save_figure(fig, out1)
    print(f"generated {out1}")

    pv_summary = _hourly_distribution(merged[["datetime", "pv"]].rename(columns={"pv": "value"}))
    fig, ax = plt.subplots(figsize=(14, 4.8))
    _plot_distribution_panel(ax, pv_summary, "PV contribution (underlying - net load with PV)", "PV proxy (kW)")
    ax.axhline(0.0, color=PALETTE["orange"], linewidth=1.2, linestyle="--")
    ax.legend(loc="upper right", ncol=1)
    ax.set_xlabel("Hour of day")
    fig.suptitle("SA BESS hourly PV contribution distribution", y=1.02)
    fig.tight_layout()
    out2 = RESULTS_DIR / "04_sa_bess_clean_44hh" / "figures" / "fig24_sa_bess_hourly_pv_distribution.png"
    save_figure(fig, out2)
    print(f"generated {out2}")

    batt_summary = _hourly_distribution(merged[["datetime", "battery"]].rename(columns={"battery": "value"}))
    fig, ax = plt.subplots(figsize=(14, 4.8))
    _plot_distribution_panel(ax, batt_summary, "Battery contribution (net load with PV+battery - net load with PV)", "Battery proxy (kW)")
    ax.axhline(0.0, color=PALETTE["orange"], linewidth=1.2, linestyle="--")
    ax.legend(loc="upper right", ncol=1)
    ax.set_xlabel("Hour of day")
    fig.suptitle("SA BESS hourly battery contribution distribution", y=1.02)
    fig.tight_layout()
    out3 = RESULTS_DIR / "04_sa_bess_clean_44hh" / "figures" / "fig25_sa_bess_hourly_battery_distribution.png"
    save_figure(fig, out3)
    print(f"generated {out3}")


if __name__ == "__main__":
    main()
