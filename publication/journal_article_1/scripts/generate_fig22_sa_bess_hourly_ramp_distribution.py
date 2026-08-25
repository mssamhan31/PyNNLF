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


def _load_series(csv_name: str) -> pd.DataFrame:
    path = DATA_DIR / csv_name
    df = pd.read_csv(path)
    if "datetime" not in df.columns or "netload_kW" not in df.columns:
        raise ValueError(f"Missing datetime/netload_kW in {path}")

    out = df[["datetime", "netload_kW"]].copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out["value"] = pd.to_numeric(out["netload_kW"], errors="coerce")
    out = out.dropna(subset=["datetime", "value"]).sort_values("datetime").reset_index(drop=True)
    return out[["datetime", "value"]]


def _hourly_ramp_distribution(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ramp"] = d["value"].diff()
    d["hour"] = d["datetime"].dt.hour
    d = d.dropna(subset=["ramp", "hour"])

    summary = (
        d.groupby("hour")["ramp"]
        .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        .unstack()
        .rename(columns={0.10: "p10", 0.25: "p25", 0.50: "p50", 0.75: "p75", 0.90: "p90"})
        .reset_index()
    )

    all_hours = pd.DataFrame({"hour": list(range(24))})
    summary = all_hours.merge(summary, on="hour", how="left")
    return summary


def main() -> None:
    apply_publication_style()

    series_info = [
        ("Underlying load", "ds22_sa_bess_44hh_pos_underlying_load_30min.csv"),
        ("Net load with PV", "ds23_sa_bess_44hh_pos_net_load_with_pv_30min.csv"),
        ("Net load with PV and battery", "ds24_sa_bess_44hh_pos_net_load_with_pv_battery_30min.csv"),
    ]

    summaries = []
    all_vals = []
    for label, csv_name in series_info:
        df = _load_series(csv_name)
        summary = _hourly_ramp_distribution(df)
        summaries.append((label, summary))
        all_vals.extend(summary[["p10", "p25", "p50", "p75", "p90"]].to_numpy().ravel().tolist())

    finite_vals = [v for v in all_vals if pd.notna(v)]
    if not finite_vals:
        raise ValueError("No finite ramp distribution values computed")
    y_abs = max(abs(min(finite_vals)), abs(max(finite_vals)))
    y_lim = y_abs * 1.05

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, sharey=True)
    for panel_idx, (ax, (label, s)) in enumerate(zip(axes, summaries)):
        x = s["hour"].to_numpy(dtype=float)
        p10 = s["p10"].to_numpy(dtype=float)
        p25 = s["p25"].to_numpy(dtype=float)
        p50 = s["p50"].to_numpy(dtype=float)
        p75 = s["p75"].to_numpy(dtype=float)
        p90 = s["p90"].to_numpy(dtype=float)

        ax.fill_between(x, p10, p90, color=PALETTE["light_grey"], alpha=0.20, label="10-90 percentile")
        ax.fill_between(x, p25, p75, color=PALETTE["grey"], alpha=0.25, label="25-75 percentile")
        ax.plot(x, p50, color=PALETTE["dark_blue"], linewidth=2.0, label="Median")
        ax.axhline(0.0, color=PALETTE["orange"], linewidth=1.2, linestyle="--")

        ax.set_xlim(0, 23)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_ylabel("Ramp (kW/30 min)")
        ax.set_title(f"({chr(97 + panel_idx)}) {label}")
        ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
        ax.legend(loc="upper right", ncol=1)

    axes[-1].set_xlabel("Hour of day")
    fig.suptitle("SA BESS hourly ramp-rate distributions (median and percentile bands)", y=1.01)
    fig.tight_layout()

    out_path = (
        RESULTS_DIR
        / "04_sa_bess_clean_44hh"
        / "figures"
        / "fig22_sa_bess_hourly_ramp_rate_distribution.png"
    )
    save_figure(fig, out_path)
    print(f"generated {out_path}")


if __name__ == "__main__":
    main()
