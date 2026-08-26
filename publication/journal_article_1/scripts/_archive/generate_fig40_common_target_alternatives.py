from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import generate_supervisor_revision_outputs as pub


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def _load_horizon_frame(recap: pd.DataFrame, horizon_min: int, horizon_label: str) -> pd.DataFrame:
    subset = recap.loc[
        recap["dataset_no"].astype(str).eq("ds20")
        & recap["forecast_horizon_min"].astype(int).eq(horizon_min)
        & recap["model_name"].astype(str).eq("m17_xgb_hp1")
    ].sort_values(["exp_date", "experiment_no"])
    if subset.empty:
        raise ValueError(f"Missing ds20 horizon {horizon_min} xgb experiment row")

    row = subset.iloc[-1]
    cv1 = pub._pick_cv1_file(str(row["experiment_folder"]))
    df = pub._read_forecast_frame(cv1).sort_values("datetime").reset_index(drop=True)
    if "datetime" not in df.columns:
        raise ValueError(f"Datetime column missing for horizon {horizon_label}")

    out = df[["datetime", "observation", "forecast"]].copy()
    out = out.dropna(subset=["datetime", "observation", "forecast"]).reset_index(drop=True)
    out = out.rename(
        columns={
            "observation": f"observation_{horizon_label}",
            "forecast": f"forecast_{horizon_label}",
        }
    )
    return out


def _find_contiguous_runs(common_df: pd.DataFrame, step: pd.Timedelta) -> list[tuple[int, int]]:
    if common_df.empty:
        return []

    diffs = common_df["datetime"].diff()
    breaks = diffs.ne(step).fillna(True)
    run_id = breaks.cumsum()

    runs: list[tuple[int, int]] = []
    for _, grp in common_df.groupby(run_id):
        start_idx = int(grp.index.min())
        end_idx = int(grp.index.max())
        runs.append((start_idx, end_idx))
    return runs


def _select_three_windows(common_df: pd.DataFrame, window_len: int) -> list[tuple[str, int]]:
    step = pd.Timedelta(minutes=30)
    runs = _find_contiguous_runs(common_df, step)

    valid_starts: list[int] = []
    for start, end in runs:
        run_len = end - start + 1
        if run_len >= window_len:
            valid_starts.extend(range(start, end - window_len + 2))

    if not valid_starts:
        raise ValueError("No contiguous common-target window found with required length")

    n = len(valid_starts)
    picks = [
        ("alt1_earliest", valid_starts[0]),
        ("alt2_middle", valid_starts[n // 2]),
        ("alt3_latest", valid_starts[-1]),
    ]
    return picks


def main() -> None:
    pub.apply_publication_style()
    recap = pub._load_publication_recap()

    horizons = [
        (30, "30m", "30-minute horizon"),
        (1440, "1d", "1-day horizon"),
        (10080, "1w", "1-week horizon"),
    ]

    frames = {short: _load_horizon_frame(recap, h, short) for h, short, _ in horizons}

    common = frames["30m"][["datetime", "observation_30m"]].copy()
    for _, short, _ in horizons:
        if short == "30m":
            common = common.merge(frames[short][["datetime", f"forecast_{short}"]], on="datetime", how="inner")
        else:
            common = common.merge(
                frames[short][["datetime", f"observation_{short}", f"forecast_{short}"]],
                on="datetime",
                how="inner",
            )

    if common.empty:
        raise ValueError("No common forecast target timestamps across all three horizons")

    common = common.sort_values("datetime").reset_index(drop=True)

    window_len = 336
    alternatives = _select_three_windows(common, window_len)
    out_dir = RESULTS_DIR / "02_ashd_148hh_forecast_horizon" / "figures" / "fig40_common_target_alternatives"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_limits = (-100.0, 180.0)

    for alt_name, start_idx in alternatives:
        window = common.iloc[start_idx : start_idx + window_len].copy().reset_index(drop=True)
        start_date = window["datetime"].iloc[0].date()
        end_date = window["datetime"].iloc[-1].date()

        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
        for panel_index, (ax, (_, short, pretty)) in enumerate(zip(axes, horizons)):
            actual = window["observation_30m"]
            forecast = window[f"forecast_{short}"]

            ax.plot(window["datetime"], actual, color=pub.PALETTE["dark_blue"], linewidth=1.6, label="Actual")
            ax.plot(window["datetime"], forecast, color=pub.PALETTE["orange"], linewidth=1.3, label="Forecast")
            ax.set_title(f"({chr(97 + panel_index)}) {pretty}")
            ax.set_ylabel("kW")
            ax.set_ylim(*y_limits)
            pub._format_datetime_axis(ax)
            pub._format_numeric_axis(ax)

        axes[0].legend(loc="upper right", ncol=2)
        axes[-1].set_xlabel("Date")
        fig.suptitle(
            f"ASHD XGBoost actual vs forecast by horizon (common targets) - {alt_name} [{start_date} to {end_date}]",
            y=1.01,
        )
        fig.tight_layout()

        out_path = out_dir / f"fig40_common_target_{alt_name}.png"
        pub.save_figure(fig, out_path)
        print(f"generated {out_path}")


if __name__ == "__main__":
    main()