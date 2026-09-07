from __future__ import annotations

import argparse
import math
import uuid
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import yaml

import pynnlf

from publication_plot_style import (
    ANNOTATION_PT,
    AXIS_LABEL_PT,
    TICK_BINS,
    TICK_PT,
    TITLE_PT,
    MODEL_COLORS,
    MODEL_SEQUENCE,
    MODEL_SEQUENCE_COLORS,
    MODEL_SHORT_LABELS,
    PALETTE,
    apply_publication_style,
    page_figsize,
    save_figure,
)


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = WORKSPACE_DIR / "results"
DATA_DIR = WORKSPACE_DIR / "data"
EXP_ROOT = WORKSPACE_DIR / "experiment_result"
EXP_DATABRICKS_ROOT = WORKSPACE_DIR / "experiment_result_databricks"

MODEL_ORDER = ["m1_naive_hp1", "m6_lr_hp1", "m17_xgb_hp1"]


def _format_numeric_axis(ax, *, x: bool = False, y: bool = True) -> None:
    if x:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=TICK_BINS))
    if y:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=TICK_BINS))


def _format_datetime_axis(ax, *, rotation: float = 0, month_year_format: str = "%b-%Y") -> None:
    locator = mdates.DayLocator(interval=1)
    formatter = mdates.ConciseDateFormatter(locator)
    formatter.offset_formats[2] = month_year_format
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", labelrotation=rotation)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("center" if rotation == 0 else "right")


def _print(msg: str) -> None:
    print(f"[revision] {msg}")


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _find_cv_test_files(experiment_folder: str) -> list[Path]:
    hits: list[Path] = []
    for root in [EXP_ROOT, EXP_DATABRICKS_ROOT]:
        exp_dir = root / experiment_folder
        if exp_dir.exists():
            hits.extend(sorted(exp_dir.glob("*_cv_test/*_test_result.csv")))
    return sorted(hits)


def _pick_cv1_file(experiment_folder: str) -> Path:
    files = _find_cv_test_files(experiment_folder)
    if not files:
        raise FileNotFoundError(f"No cv test files found for {experiment_folder}")
    for f in files:
        if "_cv1_test_result.csv" in f.name:
            return f
    return files[0]


def _find_a3_file(experiment_folder: str) -> Path | None:
    """Locate an experiment's cross-validation result file.

    The file is named by experiment number, not by the full folder name, e.g.
    ``E00085_260525_ds25_.../E00085_a3_cross_validation_result.csv``.

    Args:
        experiment_folder (str): the folder name as recorded in the recap.

    Returns:
        Path | None: the a3 file, or None if neither results store holds it.
    """
    experiment_number = experiment_folder.split("_")[0]
    for root in [EXP_ROOT, EXP_DATABRICKS_ROOT]:
        path = root / experiment_folder / f"{experiment_number}_a3_cross_validation_result.csv"
        if path.exists():
            return path
    return None


def _read_fold_metrics(experiment_folder: str) -> pd.DataFrame | None:
    """Read an experiment's per-fold metrics, excluding its summary rows.

    The a3 file carries one row per cross-validation fold plus ``mean`` and
    ``stddev`` summary rows. Callers wanting folds must drop the summaries, or
    the summaries are treated as extra folds.

    Args:
        experiment_folder (str): the folder name as recorded in the recap.

    Returns:
        pd.DataFrame | None: the fold rows, or None if the file is absent.
    """
    path = _find_a3_file(experiment_folder)
    if path is None:
        return None
    frame = pd.read_csv(path, index_col=0)
    return frame.drop(index=["mean", "stddev"], errors="ignore")


def _read_forecast_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "observation" not in df.columns:
        raise ValueError(f"Missing observation column in {path}")
    if "forecast" in df.columns:
        forecast_col = "forecast"
    else:
        candidates = [c for c in df.columns if c not in {"datetime", "observation", "residual"}]
        if not candidates:
            raise ValueError(f"No forecast column found in {path}")
        forecast_col = candidates[0]
    out = df[["observation", forecast_col]].copy()
    out = out.rename(columns={forecast_col: "forecast"})
    if "datetime" in df.columns:
        out["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return out


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _ensure_ds11_xgb_exists(allow_rerun: bool = False) -> bool:
    recap_path = EXP_ROOT / "a1_experiment_result.csv"
    recap = _load_csv(recap_path)
    mask = (
        recap["dataset_no"].astype(str).eq("ds11")
        & recap["forecast_horizon_min"].astype(int).eq(1440)
        & recap["model_name"].astype(str).eq("m17_xgb_hp1")
    )
    if mask.any():
        _print("Found existing ds11 fh8 xgb experiment in publication experiment_result.")
        return True

    if not allow_rerun:
        _print("Missing ds11 fh8 xgb experiment. Skipping rerun in fast mode.")
        return False

    _print("Missing ds11 fh8 xgb experiment. Running required long step now.")
    payload = {
        "datasets": ["ds11"],
        "forecast_horizons": ["fh8"],
        "model_and_hp": [["m17", "hp1"]],
    }
    specs_dir = WORKSPACE_DIR / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = specs_dir / f"_tmp_ds11_xgb_{uuid.uuid4().hex[:8]}.yaml"
    tmp_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    try:
        pynnlf.run_experiment_batch(tmp_path, plot_enabled=False)
        pynnlf.recap_experiments(EXP_ROOT)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    recap = _load_csv(recap_path)
    mask = (
        recap["dataset_no"].astype(str).eq("ds11")
        & recap["forecast_horizon_min"].astype(int).eq(1440)
        & recap["model_name"].astype(str).eq("m17_xgb_hp1")
    )
    ok = bool(mask.any())
    _print("Long step completed." if ok else "Long step finished but ds11 fh8 xgb still missing.")
    return ok


def _replace_legacy_section_figures() -> None:
    section_dirs = [
        RESULTS_DIR / "01_ashd_aedp_148hh_comparison" / "figures",
        RESULTS_DIR / "02_ashd_148hh_forecast_horizon" / "figures",
        RESULTS_DIR / "03_aedp_aggregation_level" / "figures",
        RESULTS_DIR / "04_sa_bess_clean_44hh" / "figures",
    ]
    for section_dir in section_dirs:
        if not section_dir.exists():
            continue
        for png in section_dir.glob("*.png"):
            png.unlink()
    _print("Removed legacy section figures for full replacement.")


def _build_experiment_parameters_table() -> Path:
    rows = [
        {
            "experiment_part": "ASHD vs AEDP comparison",
            "dataset": "ds20 (ASHD_148hh_weather), ds11 (AEDP_148hh_weather)",
            "target_variable": "netload_kW",
            "predictor_groups": "calendar, lag net load, weather",
            "lag_construction": "up to 7 days lag, horizon-aware",
            "forecast_horizon": "1 day (fh8=1440 min)",
            "cv_configuration": "10-fold CV",
            "aggregation_levels": "N/A",
            "model_set": "12 models",
            "hyperparameter_set": "hp1/hp2/hp3 as configured",
            "evaluation_metrics": "train/test RMSE, train/test nRMSE, stddev, runtime",
        },
        {
            "experiment_part": "ASHD forecast horizon comparison",
            "dataset": "ds20 (ASHD_148hh_weather)",
            "target_variable": "netload_kW",
            "predictor_groups": "calendar, lag net load, weather",
            "lag_construction": "up to 7 days lag, horizon-aware",
            "forecast_horizon": "30-min (fh1), 1-day (fh8), 1-week (fh10)",
            "cv_configuration": "10-fold CV",
            "aggregation_levels": "N/A",
            "model_set": "12 models",
            "hyperparameter_set": "hp1/hp2/hp3 as configured",
            "evaluation_metrics": "train/test RMSE, train/test nRMSE, stddev, runtime",
        },
        {
            "experiment_part": "AEDP aggregation level comparison",
            "dataset": "ds25-ds36 (AEDP sample sets)",
            "target_variable": "netload_kW",
            "predictor_groups": "calendar, lag net load, weather",
            "lag_construction": "up to 7 days lag, horizon-aware",
            "forecast_horizon": "1 day (fh8=1440 min)",
            "cv_configuration": "10-fold CV per sample",
            "aggregation_levels": "1, 10, 100, 1000 households; 3 samples each",
            "model_set": "naive, lr, xgb",
            "hyperparameter_set": "m1_hp1, m6_hp1, m17_hp1",
            "evaluation_metrics": "nRMSE, RMSE, RMSE per household, CV variability, runtime",
        },
        {
            "experiment_part": "SA BESS load composition comparison",
            "dataset": "ds22, ds23, ds24",
            "target_variable": "netload_kW",
            "predictor_groups": "calendar, lag net load, weather",
            "lag_construction": "up to 7 days lag, horizon-aware",
            "forecast_horizon": "1 day (fh8=1440 min)",
            "cv_configuration": "10-fold CV",
            "aggregation_levels": "N/A",
            "model_set": "12 models",
            "hyperparameter_set": "hp1/hp2/hp3 as configured",
            "evaluation_metrics": "train/test RMSE, train/test nRMSE, stddev, runtime",
        },
    ]
    out_path = RESULTS_DIR / "paper_table_experiment_parameters.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def _build_pynnlf_outputs_table() -> Path:
    rows = [
        {
            "output_artifact": "Experiment summary",
            "file_pattern_or_name": "*_a1_experiment_result.csv",
            "content_summary": "One-row experiment recap with metadata and evaluation metrics",
            "purpose": "Primary result record for model comparison",
            "used_in_manuscript_section": "Methods, Results tables",
        },
        {
            "output_artifact": "Hyperparameter record",
            "file_pattern_or_name": "*_a2_hyperparameter.csv",
            "content_summary": "Resolved hyperparameters used in each run",
            "purpose": "Traceability and reproducibility",
            "used_in_manuscript_section": "Appendix / reproducibility notes",
        },
        {
            "output_artifact": "Cross-validation recap",
            "file_pattern_or_name": "*_a3_cross_validation_result.csv",
            "content_summary": "Fold-level aggregate metrics",
            "purpose": "Stability and fold-consistency analysis",
            "used_in_manuscript_section": "Results (stability discussion)",
        },
        {
            "output_artifact": "Fold test series",
            "file_pattern_or_name": "*_cv_test/*_test_result.csv",
            "content_summary": "Datetime, observation, forecast, residual for each fold",
            "purpose": "Actual-vs-forecast and error diagnostics",
            "used_in_manuscript_section": "Results figures",
        },
        {
            "output_artifact": "Fold train series",
            "file_pattern_or_name": "*_cv_train/*_train_result.csv",
            "content_summary": "Datetime, observation, forecast, residual for train folds",
            "purpose": "Train diagnostics and overfitting checks",
            "used_in_manuscript_section": "Supplementary diagnostics",
        },
        {
            "output_artifact": "Model binary outputs",
            "file_pattern_or_name": "*_models/*",
            "content_summary": "Serialized trained models",
            "purpose": "Reproducible model reuse",
            "used_in_manuscript_section": "Reproducibility",
        },
    ]
    out_path = RESULTS_DIR / "paper_table_pynnlf_output_artifacts.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def _load_publication_recap() -> pd.DataFrame:
    recap = _load_csv(EXP_ROOT / "a1_experiment_result.csv")
    recap["forecast_horizon_min"] = pd.to_numeric(recap["forecast_horizon_min"], errors="coerce")
    return recap


def _plot_aggregation_summary_and_cv(recap_agg: pd.DataFrame) -> list[Path]:
    fig_dir = RESULTS_DIR / "03_aedp_aggregation_level" / "figures"
    levels = [1, 10, 100, 1000]

    # Sample-level summary
    s = recap_agg.copy()
    s["aggregation_level_hh"] = pd.to_numeric(s["aggregation_level_hh"], errors="coerce")
    s["test_nRMSE"] = pd.to_numeric(s["test_nRMSE"], errors="coerce")
    s["test_RMSE"] = pd.to_numeric(s["test_RMSE"], errors="coerce")
    s["total_household_weight"] = pd.to_numeric(s["total_household_weight"], errors="coerce")
    s["rmse_per_hh"] = s["test_RMSE"] / s["total_household_weight"]


    summary = (
        s.groupby(["model_name", "aggregation_level_hh"], as_index=False)
        .agg(
            sample_mean_nrmse=("test_nRMSE", "mean"),
            sample_std_nrmse=("test_nRMSE", "std"),
            sample_mean_rmse_per_hh=("rmse_per_hh", "mean"),
            sample_std_rmse_per_hh=("rmse_per_hh", "std"),
        )
        .fillna(0)
    )

    paths: list[Path] = []
    for metric, y_label, m_col, s_col, out_name in [
        (
            "nrmse",
            "Test nRMSE (%)",
            "sample_mean_nrmse",
            "sample_std_nrmse",
            "fig10_aedp_agg_nrmse_mean_std_cvbg.png",
        ),
        (
            "rmse_per_hh",
            "Test RMSE per household (kW)",
            "sample_mean_rmse_per_hh",
            "sample_std_rmse_per_hh",
            "fig11_aedp_agg_rmse_per_hh_mean_std_cvbg.png",
        ),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=page_figsize(16, 4.7), sharex=True, sharey=True)
        for ax, model in zip(axes, MODEL_ORDER):
            m = summary.loc[summary["model_name"].eq(model)].sort_values("aggregation_level_hh")
            x = np.arange(len(levels))
            means = [float(m.loc[m["aggregation_level_hh"].eq(lvl), m_col].iloc[0]) if (m["aggregation_level_hh"] == lvl).any() else np.nan for lvl in levels]
            stds = [float(m.loc[m["aggregation_level_hh"].eq(lvl), s_col].iloc[0]) if (m["aggregation_level_hh"] == lvl).any() else np.nan for lvl in levels]

            ax.set_axisbelow(True)

            # Always show the three sample-level results in the background.
            sm = s.loc[s["model_name"].eq(model)]
            sample_metric_col = "test_nRMSE" if metric == "nrmse" else "rmse_per_hh"
            for i, lvl in enumerate(levels):
                svals = sm.loc[sm["aggregation_level_hh"].eq(lvl), sample_metric_col].dropna().to_numpy()
                if len(svals) == 0:
                    continue
                rng = np.random.default_rng(4321 + i)
                jitter = rng.uniform(-0.06, 0.06, size=len(svals))
                ax.scatter(
                    np.full(len(svals), i) + jitter,
                    svals,
                    s=22,
                    color=PALETTE["neutral"],
                    alpha=0.55,
                    edgecolors="white",
                    linewidths=0.4,
                    zorder=2,
                )

            ax.errorbar(
                x,
                means,
                yerr=stds,
                fmt="o-",
                capsize=4,
                linewidth=1.8,
                markersize=5,
                color=MODEL_COLORS[model],
                zorder=3,
            )
            ax.set_xticks(x)
            ax.set_xticklabels([str(lvl) for lvl in levels])
            ax.set_xlabel("Aggregation level (households)")
            ax.set_title(MODEL_SHORT_LABELS[model])
            _format_numeric_axis(ax)
            if ax is axes[0]:
                ax.set_ylabel(y_label)
        fig.suptitle(
            "AEDP aggregation comparison: sample mean +/- std with sample points and fold-level background",
            y=1.03,
        )
        fig.tight_layout()
        paths.append(save_figure(fig, fig_dir / out_name))

    return paths


def _plot_aggregation_forecast_views(recap_agg: pd.DataFrame) -> list[Path]:
    fig_dir = RESULTS_DIR / "03_aedp_aggregation_level" / "figures"
    levels = [1, 10, 100, 1000]
    level_to_suffix = {1: "a", 10: "b", 100: "c", 1000: "d"}
    out_paths: list[Path] = []

    recap_agg = recap_agg.copy()
    recap_agg["sample_no"] = pd.to_numeric(recap_agg["sample_no"], errors="coerce")

    for lvl in levels:
        # deterministic selection: sample_no=1 if present, otherwise lowest sample_no
        panel_rows = []
        for model in MODEL_ORDER:
            subset = recap_agg.loc[
                recap_agg["aggregation_level_hh"].astype(int).eq(lvl)
                & recap_agg["model_name"].astype(str).eq(model)
            ].sort_values(["sample_no", "experiment_no"])
            if subset.empty:
                continue
            if (subset["sample_no"] == 1).any():
                row = subset.loc[subset["sample_no"].eq(1)].iloc[0]
            else:
                row = subset.iloc[0]
            panel_rows.append(row)

        # Time series figure
        fig, axes = plt.subplots(3, 1, figsize=page_figsize(14, 11), sharex=True)
        for ax, row in zip(axes, panel_rows):
            cv1 = _pick_cv1_file(str(row["experiment_folder"]))
            df = _read_forecast_frame(cv1)
            if "datetime" in df.columns and df["datetime"].notna().any():
                x = df["datetime"]
            else:
                x = np.arange(len(df))
            d = df.iloc[: min(336, len(df))].copy()  # up to 7 days
            if "datetime" in d.columns and d["datetime"].notna().any():
                x = d["datetime"]
            else:
                x = np.arange(len(d))
            ax.plot(x, d["observation"], color=PALETTE["series_a"], linewidth=1.6, label="Actual")
            ax.plot(x, d["forecast"], color=PALETTE["series_b"], linewidth=1.2, label="Forecast")
            ax.set_title(MODEL_SHORT_LABELS[str(row['model_name'])])
            ax.set_ylabel("kW")
            _format_datetime_axis(ax)
            _format_numeric_axis(ax)
        axes[0].legend(loc="upper right", ncol=2, frameon=True)
        axes[-1].set_xlabel("Date")
        fig.suptitle(f"AEDP aggregation {lvl} households: actual vs forecast (sample 1, CV1)", y=1.02)
        fig.tight_layout()
        out_paths.append(
            save_figure(
                fig,
                fig_dir / f"fig12{level_to_suffix[lvl]}_aedp_agg{lvl}_actual_vs_forecast_timeseries_naive_lr_xgb.png",
            )
        )

        # Scatter figure
        fig, axes = plt.subplots(1, 3, figsize=page_figsize(14.2, 5.5), sharex=False, sharey=False)
        for ax, row in zip(axes, panel_rows):
            cv1 = _pick_cv1_file(str(row["experiment_folder"]))
            df = _read_forecast_frame(cv1)
            d = df.iloc[: min(336, len(df))].copy()
            ax.scatter(
                d["observation"],
                d["forecast"],
                s=12,
                alpha=0.45,
                color=MODEL_COLORS[str(row["model_name"])],
                edgecolors="none",
            )
            lo = float(min(d["observation"].min(), d["forecast"].min()))
            hi = float(max(d["observation"].max(), d["forecast"].max()))
            ax.plot([lo, hi], [lo, hi], color=PALETTE["neutral"], linewidth=1.1, linestyle="--")
            ax.set_title(MODEL_SHORT_LABELS[str(row["model_name"])])
            ax.set_xlabel("Actual (kW)")
            ax.set_ylabel("Forecast (kW)")
            _format_numeric_axis(ax, x=True)
        fig.suptitle(f"AEDP aggregation {lvl} households: actual vs forecast scatter (sample 1, CV1)", y=1.03)
        fig.tight_layout()
        out_paths.append(
            save_figure(
                fig,
                fig_dir / f"fig13{level_to_suffix[lvl]}_aedp_agg{lvl}_actual_vs_forecast_scatter_naive_lr_xgb.png",
            )
        )

    return out_paths


def _plot_aggregation_xgb_timeseries(recap_agg: pd.DataFrame) -> Path:
    fig_dir = RESULTS_DIR / "03_aedp_aggregation_level" / "figures"
    levels = [1, 10, 100, 1000]
    frames: dict[int, pd.DataFrame] = {}

    recap = recap_agg.copy()
    recap["aggregation_level_hh"] = pd.to_numeric(recap["aggregation_level_hh"], errors="coerce")
    recap["sample_no"] = pd.to_numeric(recap["sample_no"], errors="coerce")

    for level in levels:
        subset = recap.loc[
            recap["aggregation_level_hh"].eq(level)
            & recap["model_name"].astype(str).eq("m17_xgb_hp1")
        ].sort_values(["sample_no", "experiment_no"])
        if subset.empty:
            raise ValueError(f"Missing XGBoost aggregation result for {level} households")
        row = subset.loc[subset["sample_no"].eq(1)].iloc[0] if subset["sample_no"].eq(1).any() else subset.iloc[0]
        frame = _read_forecast_frame(_pick_cv1_file(str(row["experiment_folder"]))).iloc[:144].copy()
        if "datetime" not in frame.columns or frame["datetime"].isna().any():
            raise ValueError(f"Datetime unavailable for XGBoost aggregation level {level}")
        frames[level] = frame

    reference_dates = frames[levels[0]]["datetime"].reset_index(drop=True)
    for level in levels[1:]:
        if not frames[level]["datetime"].reset_index(drop=True).equals(reference_dates):
            raise ValueError("XGBoost aggregation comparison requires identical timestamps across household levels")

    fig, axes = plt.subplots(2, 2, figsize=page_figsize(14, 10), sharex=True)
    for panel_index, (ax, level) in enumerate(zip(axes.flat, levels)):
        frame = frames[level]
        ax.plot(frame["datetime"], frame["observation"], color=PALETTE["series_a"], linewidth=1.6, label="Actual")
        ax.plot(frame["datetime"], frame["forecast"], color=PALETTE["series_b"], linewidth=1.3, label="xgb_hp1 forecast")
        household_label = "household" if level == 1 else "households"
        ax.set_title(f"({chr(97 + panel_index)}) {level:,} {household_label}")
        ax.set_ylabel("kW")
        _format_datetime_axis(ax, rotation=0, month_year_format="%b-%Y")
        _format_numeric_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2)
    axes[1, 0].set_xlabel("Date")
    axes[1, 1].set_xlabel("Date")
    fig.suptitle("AEDP XGBoost actual vs forecast by aggregation level (sample 1, CV1)", y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return save_figure(fig, fig_dir / "fig14_aedp_aggregation_xgb_timeseries_aligned.png")


def _select_representative_day(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.DataFrame]:
    d = df.copy()
    d["date"] = d["datetime"].dt.date
    day_stats = (
        d.groupby("date", as_index=False)
        .agg(volatility=("observation", lambda x: float(np.std(np.diff(x.to_numpy())) if len(x) > 1 else 0.0)), n=("observation", "size"))
        .sort_values(["volatility", "date"], ascending=[False, True])
    )
    day_stats = day_stats.loc[day_stats["n"].ge(40)]
    if day_stats.empty:
        day = pd.to_datetime(d["date"].iloc[0])
    else:
        day = pd.to_datetime(day_stats.iloc[0]["date"])
    return day, d.loc[d["date"].eq(day.date())].copy()


def _select_representative_week(df: pd.DataFrame, points_per_day: int = 48) -> tuple[pd.Timestamp, pd.DataFrame]:
    d = df.copy()
    if "datetime" not in d.columns or d["datetime"].isna().all():
        start_idx = 0
        end_idx = min(len(d), points_per_day * 7)
        return pd.Timestamp("1970-01-01"), d.iloc[start_idx:end_idx].copy()

    d = d.sort_values("datetime").reset_index(drop=True)
    window = points_per_day * 7
    if len(d) <= window:
        start = d["datetime"].iloc[0]
        return start, d.copy()

    rolling_vol = (
        d["observation"]
        .rolling(window=window, min_periods=max(2, points_per_day * 3))
        .apply(lambda x: float(np.std(np.diff(x.to_numpy()))) if len(x) > 1 else np.nan, raw=False)
    )
    if rolling_vol.dropna().empty:
        start_idx = 0
    else:
        start_idx = int(rolling_vol.dropna().idxmax() - window + 1)
        start_idx = max(0, min(start_idx, len(d) - window))
    out = d.iloc[start_idx : start_idx + window].copy()
    return out["datetime"].iloc[0], out


def _select_composition_week(frames: dict[str, pd.DataFrame], points_per_day: int = 48) -> tuple[pd.Timestamp, dict[str, pd.DataFrame], str]:
    keys = ["underlying_load", "net_load_with_pv", "net_load_with_pv_battery"]
    sorted_frames = {k: frames[k].sort_values("datetime").reset_index(drop=True) for k in keys}
    min_len = min(len(sorted_frames[k]) for k in keys)
    window = points_per_day * 7
    if min_len <= window:
        clipped = {k: sorted_frames[k].iloc[:min_len].copy() for k in keys}
        start = clipped[keys[0]]["datetime"].iloc[0] if "datetime" in clipped[keys[0]].columns else pd.Timestamp("1970-01-01")
        return start, clipped, "fallback_short_series"

    best_idx = None
    best_rank = None
    best_gap = None
    for i in range(0, min_len - window + 1):
        rough = {}
        for k in keys:
            seg = sorted_frames[k].iloc[i : i + window]
            vals = seg["observation"].to_numpy()
            rough[k] = float(np.std(np.diff(vals))) if len(vals) > 1 else np.inf
        ordered = sorted(rough.items(), key=lambda kv: kv[1])
        rank_map = {name: idx for idx, (name, _) in enumerate(ordered)}
        underlying_rank = rank_map["underlying_load"]
        gap = rough["underlying_load"] - min(rough["net_load_with_pv"], rough["net_load_with_pv_battery"])

        candidate = (underlying_rank, gap)
        if best_rank is None or candidate < (best_rank, best_gap):
            best_idx = i
            best_rank = underlying_rank
            best_gap = gap

    assert best_idx is not None
    clipped = {k: sorted_frames[k].iloc[best_idx : best_idx + window].copy() for k in keys}
    start = clipped[keys[0]]["datetime"].iloc[0]
    mode = "underlying_not_most_volatile" if best_rank == 0 else "best_available_rank"
    return start, clipped, mode


def _plot_ashd_vs_aedp_xgb(
    recap_exp: pd.DataFrame,
    recap_agg: pd.DataFrame,
    notes: list[str],
    strict_ashd_aedp: bool = False,
) -> list[Path]:
    fig_dir = RESULTS_DIR / "01_ashd_aedp_148hh_comparison" / "figures"

    rows = {}
    # ASHD source (required)
    for ds_no, label in [("ds20", "ASHD")]:
        subset = recap_exp.loc[
            recap_exp["dataset_no"].astype(str).eq(ds_no)
            & recap_exp["forecast_horizon_min"].astype(int).eq(1440)
            & recap_exp["model_name"].astype(str).eq("m17_xgb_hp1")
        ].sort_values(["exp_date", "experiment_no"])
        if subset.empty:
            raise ValueError(f"Missing {ds_no} fh8 xgb experiment row")
        rows[label] = subset.iloc[-1]

    # Preferred AEDP source: ds11, fallback to AEDP aggregation level 100 sample 1
    ds11_subset = recap_exp.loc[
        recap_exp["dataset_no"].astype(str).eq("ds11")
        & recap_exp["forecast_horizon_min"].astype(int).eq(1440)
        & recap_exp["model_name"].astype(str).eq("m17_xgb_hp1")
    ].sort_values(["exp_date", "experiment_no"])
    if not ds11_subset.empty:
        rows["AEDP"] = ds11_subset.iloc[-1]
        aedp_label = "AEDP"
    else:
        if strict_ashd_aedp:
            raise ValueError("Strict ASHD-vs-AEDP mode requires ds11 fh8 xgb folds, but they are unavailable.")
        fallback = recap_agg.copy()
        fallback["aggregation_level_hh"] = pd.to_numeric(fallback["aggregation_level_hh"], errors="coerce")
        fallback["sample_no"] = pd.to_numeric(fallback["sample_no"], errors="coerce")
        fallback = fallback.loc[
            fallback["model_name"].astype(str).eq("m17_xgb_hp1")
            & fallback["aggregation_level_hh"].eq(100)
        ].sort_values(["sample_no", "exp_date", "experiment_no"])
        if fallback.empty:
            raise ValueError("Missing AEDP fallback row (aggregation level 100, XGBoost)")
        if (fallback["sample_no"] == 1).any():
            rows["AEDP"] = fallback.loc[fallback["sample_no"].eq(1)].iloc[-1]
        else:
            rows["AEDP"] = fallback.iloc[0]
        aedp_label = "AEDP (100hh fallback)"
        notes.append(
            "ASHD vs AEDP XGBoost figure used AEDP aggregation fallback (100 households, sample 1) because ds11 fh8 xgb folds are unavailable locally."
        )

    fixed_windows = {
        "ASHD": (pd.Timestamp("2013-02-06"), pd.Timestamp("2013-02-13")),
        "AEDP": (pd.Timestamp("2024-02-06"), pd.Timestamp("2024-02-13")),
    }

    week_info = {}
    positive_peaks = {}
    for label, row in rows.items():
        cv1 = _pick_cv1_file(str(row["experiment_folder"]))
        df = _read_forecast_frame(cv1)
        if "datetime" not in df.columns or df["datetime"].isna().all():
            raise ValueError(f"Datetime missing in {cv1}")
        positive_actual = pd.to_numeric(df["observation"], errors="coerce")
        positive_actual = positive_actual.loc[positive_actual.gt(0)]
        if positive_actual.empty:
            raise ValueError(f"Positive actual peak unavailable in {cv1}")
        positive_peaks[label] = float(positive_actual.max())
        start, end = fixed_windows[label]
        week_df = df.loc[(df["datetime"] >= start) & (df["datetime"] < end)].copy().reset_index(drop=True)
        if len(week_df) < 336:
            raise ValueError(
                f"Insufficient rows for fixed {label} window {start.date()} to {end.date()} in {cv1}: {len(week_df)}"
            )
        week_info[label] = (start, week_df)

    normalized_limits = (-1.3, 1.3)
    normalized_yticks = [-1.0, -0.5, 0.0, 0.5, 1.0]

    fig, axes = plt.subplots(2, 1, figsize=page_figsize(14, 11), sharex=False, sharey=True)
    for ax, label in zip(axes, ["ASHD", "AEDP"]):
        week_start, d = week_info[label]
        peak = positive_peaks[label]
        ax.set_axisbelow(True)
        ax.plot(d["datetime"], d["observation"] / peak, color=PALETTE["series_a"], linewidth=1.8, label="Actual")
        ax.plot(d["datetime"], d["forecast"] / peak, color=PALETTE["series_b"], linewidth=1.4, label="Forecast")
        ax.axhline(1.0, color=PALETTE["neutral"], linewidth=1.1, linestyle="--", label=f"CV1 actual peak: {peak:.2f} kW")
        title_label = "ASHD" if label == "ASHD" else aedp_label
        ax.set_title(f"{title_label} - representative week from {week_start.date()}")
        ax.set_ylabel("Load / positive peak")
        ax.set_ylim(*normalized_limits)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=3)
        _format_datetime_axis(ax)
        _format_numeric_axis(ax, y=False)
        ax.set_yticks(normalized_yticks)
    axes[-1].set_xlabel("Date")
    fig.suptitle("XGBoost actual vs forecast: peak-normalized shared scale", y=1.02)
    fig.tight_layout(h_pad=3.0)
    p1 = save_figure(
        fig,
        fig_dir / "fig30_ashd_vs_aedp_xgb_actual_vs_forecast_daypair.png",
    )

    normalized_errors = {
        label: (
            week_info[label][1]["forecast"] - week_info[label][1]["observation"]
        ) / positive_peaks[label]
        for label in ["ASHD", "AEDP"]
    }
    max_abs_normalized_error = max(
        float(np.nanmax(np.abs(error.to_numpy(dtype=float))))
        for error in normalized_errors.values()
    )
    normalized_error_limit = max_abs_normalized_error * 1.05

    # error profile
    fig, axes = plt.subplots(2, 1, figsize=page_figsize(14, 8.5), sharex=False, sharey=True)
    for ax, label in zip(axes, ["ASHD", "AEDP"]):
        week_start, d = week_info[label]
        ax.set_axisbelow(True)
        err = normalized_errors[label]
        ax.plot(d["datetime"], err, color=PALETTE["series_c"], linewidth=1.4)
        ax.axhline(0, color=PALETTE["neutral"], linewidth=1.0, linestyle="--")
        title_label = "ASHD" if label == "ASHD" else aedp_label
        ax.set_title(f"{title_label} forecast error profile (week from {week_start.date()})")
        ax.set_ylabel("Error / positive peak")
        ax.set_ylim(-normalized_error_limit, normalized_error_limit)
        _format_datetime_axis(ax)
        _format_numeric_axis(ax)
    axes[-1].set_xlabel("Date")
    fig.suptitle("XGBoost representative-week error profile: peak-normalized shared scale", y=1.02)
    fig.tight_layout()
    p2 = save_figure(fig, fig_dir / "fig31_ashd_vs_aedp_xgb_error_profile_daypair.png")

    return [p1, p2]


def _plot_horizon_xgb(recap_exp: pd.DataFrame) -> list[Path]:
    fig_dir = RESULTS_DIR / "02_ashd_148hh_forecast_horizon" / "figures"
    horizons = [(30, "30-minute", "30m"), (1440, "1-day", "1d"), (10080, "1-week", "1w")]

    frames = {}
    for h, label, short in horizons:
        subset = recap_exp.loc[
            recap_exp["dataset_no"].astype(str).eq("ds20")
            & recap_exp["forecast_horizon_min"].astype(int).eq(h)
            & recap_exp["model_name"].astype(str).eq("m17_xgb_hp1")
        ].sort_values(["exp_date", "experiment_no"])
        if subset.empty:
            raise ValueError(f"Missing ds20 horizon {h} xgb experiment row")
        row = subset.iloc[-1]
        cv1 = _pick_cv1_file(str(row["experiment_folder"]))
        df = _read_forecast_frame(cv1).sort_values("datetime").reset_index(drop=True)
        if "datetime" not in df.columns or df["datetime"].isna().all():
            raise ValueError(f"Datetime missing in {cv1}")
        df = df[["datetime", "observation", "forecast"]].dropna(subset=["datetime", "observation", "forecast"])
        frames[short] = df.rename(
            columns={
                "observation": f"observation_{short}",
                "forecast": f"forecast_{short}",
            }
        )

    common = frames["30m"][["datetime", "observation_30m", "forecast_30m"]].copy()
    common = common.merge(frames["1d"][["datetime", "forecast_1d"]], on="datetime", how="inner")
    common = common.merge(frames["1w"][["datetime", "forecast_1w"]], on="datetime", how="inner")
    common = common.sort_values("datetime").reset_index(drop=True)
    if common.empty:
        raise ValueError("No common target timestamps found across 30-minute, 1-day, and 1-week horizons")

    window_len = 336
    diffs = common["datetime"].diff()
    run_break = diffs.ne(pd.Timedelta(minutes=30)).fillna(True)
    run_id = run_break.cumsum()
    start_idx = None
    for _, grp in common.groupby(run_id):
        if len(grp) >= window_len:
            start_idx = int(grp.index.min())
            break
    if start_idx is None:
        raise ValueError("No contiguous common-target span with at least 336 points")
    window = common.iloc[start_idx : start_idx + window_len].copy().reset_index(drop=True)

    selected = {
        "30-minute": pd.DataFrame(
            {
                "datetime": window["datetime"],
                "observation": window["observation_30m"],
                "forecast": window["forecast_30m"],
            }
        ),
        "1-day": pd.DataFrame(
            {
                "datetime": window["datetime"],
                "observation": window["observation_30m"],
                "forecast": window["forecast_1d"],
            }
        ),
        "1-week": pd.DataFrame(
            {
                "datetime": window["datetime"],
                "observation": window["observation_30m"],
                "forecast": window["forecast_1w"],
            }
        ),
    }

    fig, axes = plt.subplots(3, 1, figsize=page_figsize(14, 12), sharex=False)
    shared_horizon_ylim = (-100.0, 180.0)
    horizon_labels = ["30-minute", "1-day", "1-week"]
    for ax, label in zip(axes, horizon_labels):
        d = selected[label]
        x = d["datetime"] if "datetime" in d.columns else np.arange(len(d))
        ax.plot(x, d["observation"], color=PALETTE["series_a"], linewidth=1.6, label="Actual")
        ax.plot(x, d["forecast"], color=PALETTE["series_b"], linewidth=1.3, label="Forecast")
        panel_index = horizon_labels.index(label)
        ax.set_title(f"({chr(97 + panel_index)}) {label} horizon")
        ax.set_ylabel("kW")
        ax.set_ylim(*shared_horizon_ylim)
        if "datetime" in d.columns:
            _format_datetime_axis(ax)
        else:
            _format_numeric_axis(ax, x=True, y=False)
        _format_numeric_axis(ax)
    axes[0].legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("Date")
    fig.suptitle("ASHD XGBoost actual vs forecast by horizon", y=1.01)
    fig.tight_layout()
    p1 = save_figure(fig, fig_dir / "fig40_ashd_horizons_xgb_actual_vs_forecast.png")

    # optional summary figure
    rows = []
    for h, label, _ in horizons:
        subset = recap_exp.loc[
            recap_exp["dataset_no"].astype(str).eq("ds20")
            & recap_exp["forecast_horizon_min"].astype(int).eq(h)
            & recap_exp["model_name"].astype(str).eq("m17_xgb_hp1")
        ].sort_values(["exp_date", "experiment_no"])
        row = subset.iloc[-1]
        # The engine already recorded per-fold MAE, so read it rather than
        # recomputing it from every fold's raw series. The means agree to four
        # decimal places; the spread here is the sample standard deviation,
        # which is the right one for a sample of folds.
        folds = _read_fold_metrics(str(row["experiment_folder"]))
        if folds is None or "test_MAE" not in folds.columns:
            raise FileNotFoundError(
                f"Per-fold metrics unavailable for {row['experiment_folder']}"
            )
        maes = pd.to_numeric(folds["test_MAE"], errors="coerce").dropna()
        rows.append({
            "horizon": label,
            "mae_mean": float(maes.mean()),
            "mae_std": float(maes.std()),
        })
    em = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=page_figsize(8, 4.6))
    ax.set_axisbelow(True)
    x = np.arange(len(em))
    ax.bar(x, em["mae_mean"], yerr=em["mae_std"], capsize=5, color=PALETTE["series_a"], alpha=0.9, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(em["horizon"])
    ax.set_ylabel("CV-fold MAE (kW)")
    ax.set_title("ASHD XGBoost error by horizon (mean +/- std over folds)")
    _format_numeric_axis(ax)
    fig.tight_layout()
    p2 = save_figure(fig, fig_dir / "fig41_ashd_horizons_xgb_error_summary.png")

    return [p1, p2]


def _plot_load_composition_timeseries(recap_sa: pd.DataFrame) -> list[Path]:
    fig_dir = RESULTS_DIR / "04_sa_bess_clean_44hh" / "figures"

    label_order = [
        "underlying_load",
        "net_load_with_pv",
        "net_load_with_pv_battery",
    ]
    pretty = {
        "underlying_load": "Underlying load",
        "net_load_with_pv": "Net load with PV",
        "net_load_with_pv_battery": "Net load with PV and battery",
    }
    dataset_files = {
        "underlying_load": DATA_DIR / "ds22_sa_bess_44hh_pos_underlying_load_30min.csv",
        "net_load_with_pv": DATA_DIR / "ds23_sa_bess_44hh_pos_net_load_with_pv_30min.csv",
        "net_load_with_pv_battery": DATA_DIR / "ds24_sa_bess_44hh_pos_net_load_with_pv_battery_30min.csv",
    }

    actual_frames: dict[str, pd.DataFrame] = {}
    for lbl in label_order:
        path = dataset_files[lbl]
        d = _load_csv(path)
        if "datetime" not in d.columns or "netload_kW" not in d.columns:
            raise ValueError(f"Missing datetime/netload_kW in {path}")
        d = d[["datetime", "netload_kW"]].copy()
        d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce")
        d["observation"] = pd.to_numeric(d["netload_kW"], errors="coerce")
        d = d.dropna(subset=["datetime", "observation"]).sort_values("datetime").reset_index(drop=True)
        actual_frames[lbl] = d[["datetime", "observation"]]

    frames = {}
    for lbl in label_order:
        subset = recap_sa.loc[
            recap_sa["dataset_label"].astype(str).eq(lbl)
            & recap_sa["model_name"].astype(str).eq("m17_xgb_hp1")
            & recap_sa["forecast_horizon_min"].astype(int).eq(1440)
        ].sort_values(["exp_date", "experiment_no"])
        if subset.empty:
            raise ValueError(f"Missing SA BESS xgb recap row for {lbl}")
        row = subset.iloc[-1]
        cv1 = _pick_cv1_file(str(row["experiment_folder"]))
        df = _read_forecast_frame(cv1)
        if "datetime" not in df.columns or df["datetime"].isna().all():
            raise ValueError(f"Datetime missing in {cv1}")
        aligned = actual_frames[lbl].merge(df[["datetime", "forecast"]], on="datetime", how="inner")
        if aligned.empty:
            raise ValueError(f"No datetime overlap between actual readings and forecast in {cv1}")
        frames[lbl] = aligned[["datetime", "observation", "forecast"]].sort_values("datetime").reset_index(drop=True)

    week_start, week_frames, week_mode = _select_composition_week(frames, points_per_day=48)

    # Component overlays based on actual measured series in ds22/ds23/ds24.
    # Direct 30-minute PV/battery channels are not stored as standalone publication CSVs.
    pv_generation_obs = (
        week_frames["underlying_load"]["observation"].to_numpy()
        - week_frames["net_load_with_pv"]["observation"].to_numpy()
    )
    battery_operation_obs = (
        week_frames["net_load_with_pv_battery"]["observation"].to_numpy()
        - week_frames["net_load_with_pv"]["observation"].to_numpy()
    )

    roughness = {
        lbl: float(np.std(np.diff(week_frames[lbl]["observation"].to_numpy())))
        for lbl in label_order
    }
    overlay_roughness = {
        "net_load_with_pv": float(np.std(np.diff(pv_generation_obs))),
        "net_load_with_pv_battery": float(np.std(np.diff(battery_operation_obs))),
    }

    y_values = []
    for lbl in label_order:
        y_values.append(week_frames[lbl]["observation"].to_numpy())
        y_values.append(week_frames[lbl]["forecast"].to_numpy())
    y_values.extend([pv_generation_obs, battery_operation_obs])
    y_all = np.concatenate(y_values)
    y_min = float(np.nanmin(y_all))
    y_max = float(np.nanmax(y_all))
    y_pad = 0.04 * (y_max - y_min if y_max > y_min else 1.0)
    y_limits = (y_min - y_pad, y_max + y_pad)

    fig, axes = plt.subplots(3, 1, figsize=page_figsize(14, 12), sharex=False)
    for ax, lbl in zip(axes, label_order):
        d = week_frames[lbl]
        ax.set_axisbelow(True)
        x = d["datetime"] if "datetime" in d.columns else np.arange(len(d))
        ax.plot(x, d["observation"], color=PALETTE["series_a"], linewidth=1.6, label="Actual")
        ax.plot(x, d["forecast"], color=PALETTE["series_b"], linewidth=1.2, label="Forecast")
        if lbl == "net_load_with_pv":
            ax.plot(
                x,
                pv_generation_obs,
                color=PALETTE["series_c"],
                linewidth=1.1,
                linestyle="--",
                label="PV generation (reading-based)",
            )
        elif lbl == "net_load_with_pv_battery":
            ax.plot(
                x,
                battery_operation_obs,
                color=PALETTE["series_d"],
                linewidth=1.1,
                linestyle=":",
                label="Battery operation (reading-based)",
            )
        ax.set_title(pretty[lbl])
        ax.set_ylabel("kW")
        ax.set_ylim(*y_limits)
        _format_datetime_axis(ax)
        _format_numeric_axis(ax)
        if lbl == "net_load_with_pv":
            ann = (
                f"Roughness (net load with PV): {roughness[lbl]:.2f} | "
                f"Roughness (PV): {overlay_roughness[lbl]:.2f}"
            )
        elif lbl == "net_load_with_pv_battery":
            ann = (
                f"Roughness (net load with PV+battery): {roughness[lbl]:.2f} | "
                f"Roughness (battery): {overlay_roughness[lbl]:.2f}"
            )
        else:
            ann = f"Roughness (underlying load): {roughness[lbl]:.2f}"
        ax.text(
            0.01,
            -0.29,
            ann,
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=ANNOTATION_PT,
        )
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.31), ncol=3)
    axes[-1].set_xlabel("Date")
    suffix = "(week where underlying is not the roughest)" if week_mode == "underlying_not_most_volatile" else "(best available representative week)"
    fig.suptitle(f"SA BESS composition: XGBoost actual vs forecast {suffix}, from {week_start.date()}", y=1.01)
    fig.tight_layout(h_pad=4.2)
    p1 = save_figure(fig, fig_dir / "fig20_sa_bess_composition_actual_vs_forecast_timeseries.png")

    errors = {
        lbl: week_frames[lbl]["forecast"] - week_frames[lbl]["observation"]
        for lbl in label_order
    }
    max_abs_error = max(
        float(np.nanmax(np.abs(error.to_numpy(dtype=float))))
        for error in errors.values()
    )
    error_limit = max_abs_error * 1.05

    fig, axes = plt.subplots(3, 1, figsize=page_figsize(14, 12), sharex=False, sharey=True)
    for ax, lbl in zip(axes, label_order):
        d = week_frames[lbl]
        ax.set_axisbelow(True)
        x = d["datetime"] if "datetime" in d.columns else np.arange(len(d))
        err = errors[lbl]
        ax.plot(x, err, color=PALETTE["series_c"], linewidth=1.3)
        ax.axhline(0, color=PALETTE["neutral"], linestyle="--", linewidth=1.0)
        ax.set_title(pretty[lbl])
        ax.set_ylabel("Error (kW)")
        ax.set_ylim(-error_limit, error_limit)
        _format_datetime_axis(ax)
        _format_numeric_axis(ax)
    axes[-1].set_xlabel("Date")
    fig.suptitle("SA BESS composition: XGBoost error time series", y=1.01)
    fig.tight_layout()
    p2 = save_figure(fig, fig_dir / "fig21_sa_bess_composition_error_timeseries.png")

    return [p1, p2]


def _build_model_selection_recommendation() -> Path:
    horizon_recap = _load_csv(RESULTS_DIR / "02_ashd_148hh_forecast_horizon" / "ashd_148hh_horizon_combined_recap.csv")
    d = horizon_recap.loc[horizon_recap["horizon_label"].astype(str).eq("1_week")].copy()
    d["test_nRMSE"] = pd.to_numeric(d["test_nRMSE"], errors="coerce")
    d["test_nRMSE_stddev"] = pd.to_numeric(d["test_nRMSE_stddev"], errors="coerce")
    d["runtime_ms"] = pd.to_numeric(d["runtime_ms"], errors="coerce")
    d["runtime_s"] = d["runtime_ms"] / 1000.0

    # Pareto (nRMSE, stddev, runtime)
    pts = d[["model_name", "test_nRMSE", "test_nRMSE_stddev", "runtime_s"]].dropna().reset_index(drop=True)
    pareto_models = []
    for i, r in pts.iterrows():
        dominated = False
        for j, q in pts.iterrows():
            if i == j:
                continue
            no_worse = (
                q["test_nRMSE"] <= r["test_nRMSE"]
                and q["test_nRMSE_stddev"] <= r["test_nRMSE_stddev"]
                and q["runtime_s"] <= r["runtime_s"]
            )
            strictly = (
                q["test_nRMSE"] < r["test_nRMSE"]
                or q["test_nRMSE_stddev"] < r["test_nRMSE_stddev"]
                or q["runtime_s"] < r["runtime_s"]
            )
            if no_worse and strictly:
                dominated = True
                break
        if not dominated:
            pareto_models.append(str(r["model_name"]))

    pareto_pick = (
        pts.loc[pts["model_name"].isin(pareto_models)]
        .sort_values(["test_nRMSE", "test_nRMSE_stddev", "runtime_s"]) 
        .iloc[0]["model_name"]
    )

    # Utility method
    util = d[["model_name", "test_nRMSE", "test_nRMSE_stddev", "runtime_s"]].dropna().copy()
    for c in ["test_nRMSE", "test_nRMSE_stddev", "runtime_s"]:
        vmin, vmax = util[c].min(), util[c].max()
        util[f"{c}_norm"] = 0.0 if math.isclose(vmin, vmax) else (util[c] - vmin) / (vmax - vmin)
    util["utility_score"] = 0.6 * util["test_nRMSE_norm"] + 0.3 * util["test_nRMSE_stddev_norm"] + 0.1 * util["runtime_s_norm"]
    util = util.sort_values(["utility_score", "test_nRMSE"])
    util_pick = str(util.iloc[0]["model_name"])

    # Satisficing method
    std_cap = 1.0
    runtime_cap_s = 180.0
    sat = d.loc[(d["test_nRMSE_stddev"] <= std_cap) & (d["runtime_s"] <= runtime_cap_s)].copy()
    if sat.empty:
        # deterministic fallback to still provide a recommendation if strict caps over-filter
        std_cap = 1.5
        sat = d.loc[(d["test_nRMSE_stddev"] <= std_cap) & (d["runtime_s"] <= runtime_cap_s)].copy()
    if sat.empty:
        sat = d.copy()
    sat = sat.sort_values(["test_nRMSE", "test_nRMSE_stddev", "runtime_s"])
    sat_pick = str(sat.iloc[0]["model_name"])

    table = pd.DataFrame(
        [
            {
                "method": "Pareto",
                "optimisation_metric": "Minimise test_nRMSE with non-dominated filtering on (test_nRMSE, test_nRMSE_stddev, runtime_s)",
                "constraints": "Non-dominance constraints across all three criteria",
                "eligible_models": "; ".join(sorted(set(pareto_models))),
                "recommended_model": pareto_pick,
                "notes": "Final pick from Pareto set uses lowest nRMSE then stddev then runtime",
            },
            {
                "method": "Utility",
                "optimisation_metric": "Minimise weighted utility score",
                "constraints": "weights: nRMSE=0.6, stddev=0.3, runtime=0.1; min-max normalization",
                "eligible_models": "; ".join(util["model_name"].astype(str).tolist()),
                "recommended_model": util_pick,
                "notes": "Single-score ranking of all models",
            },
            {
                "method": "Satisficing",
                "optimisation_metric": "Minimise test_nRMSE",
                "constraints": f"test_nRMSE_stddev <= {std_cap:.1f}, runtime_s <= {runtime_cap_s:.0f}",
                "eligible_models": "; ".join(sat["model_name"].astype(str).tolist()),
                "recommended_model": sat_pick,
                "notes": "Primary threshold was stddev<=1.0; relaxed to 1.5 if empty",
            },
        ]
    )
    out_path = RESULTS_DIR / "paper_table_model_selection_recommendation.csv"
    table.to_csv(out_path, index=False)
    return out_path


def _check_outputs(paths: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for p in sorted(set(paths)):
        rows.append({
            "path": str(p.relative_to(WORKSPACE_DIR)),
            "exists": p.exists(),
            "size_bytes": p.stat().st_size if p.exists() else 0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Load composition comparison, reported in absolute kilowatts
# ---------------------------------------------------------------------------

COMPOSITION_ORDER = ["underlying_load", "net_load_with_pv", "net_load_with_pv_battery"]
COMPOSITION_LABELS = {
    "underlying_load": "Underlying load",
    "net_load_with_pv": "Net load with PV",
    "net_load_with_pv_battery": "Net load with PV and battery",
}
COMPOSITION_DATASETS = {
    "ds22": "underlying_load",
    "ds23": "net_load_with_pv",
    "ds24": "net_load_with_pv_battery",
}


def _composition_metric_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot the load composition recap into model-by-composition RMSE tables.

    Reads the existing cross-validated results only; no experiment is re-run.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: mean test RMSE in kilowatts and its
        cross-validation fold standard deviation, both indexed by model and
        ordered by ``MODEL_SEQUENCE`` / ``COMPOSITION_ORDER``.
    """
    recap = _load_csv(RESULTS_DIR / "04_sa_bess_clean_44hh" / "sa_bess_44hh_fh8_combined_recap.csv")
    recap = recap.loc[recap["dataset_no"].astype(str).isin(COMPOSITION_DATASETS)].copy()
    recap["composition"] = recap["dataset_no"].astype(str).map(COMPOSITION_DATASETS)

    def pivot(column: str) -> pd.DataFrame:
        table = recap.pivot_table(index="model_name", columns="composition", values=column)
        missing_models = [m for m in MODEL_SEQUENCE if m not in table.index]
        missing_cols = [c for c in COMPOSITION_ORDER if c not in table.columns]
        if missing_models or missing_cols:
            raise ValueError(
                f"Composition table for {column} is incomplete. "
                f"Missing models: {missing_models}; missing compositions: {missing_cols}"
            )
        return table.loc[MODEL_SEQUENCE, COMPOSITION_ORDER]

    return pivot("test_RMSE"), pivot("test_RMSE_stddev")


def _build_composition_rmse_kw_table() -> list[Path]:
    """Write the load composition results as absolute RMSE in kilowatts.

    The section previously reported normalised RMSE. Each composition is
    normalised by its own peak net load (137.5, 101.4 and 71.7 kW), so the
    percentages are not comparable across compositions and the ordering they
    imply differs from the ordering in kilowatts. Absolute RMSE avoids that.

    Returns:
        list[Path]: the CSV written for downstream use and the Markdown copy
        intended for pasting into the manuscript.
    """
    rmse, rmse_std = _composition_metric_tables()
    out_dir = RESULTS_DIR / "04_sa_bess_clean_44hh"

    table = rmse.copy()
    table.index = [MODEL_SHORT_LABELS[m] for m in table.index]
    table.columns = [COMPOSITION_LABELS[c] for c in table.columns]
    table = table.sort_values(COMPOSITION_LABELS["underlying_load"]).round(2)
    table.index.name = "model_hp"

    csv_path = out_dir / "paper_table_sa_bess_44hh_signal_test_rmse_kw.csv"
    table.to_csv(csv_path)

    std_table = rmse_std.copy()
    std_table.index = [MODEL_SHORT_LABELS[m] for m in std_table.index]
    std_table.columns = [COMPOSITION_LABELS[c] for c in std_table.columns]
    std_table = std_table.loc[table.index].round(2)
    std_table.index.name = "model_hp"
    std_csv_path = out_dir / "paper_table_sa_bess_44hh_signal_test_rmse_kw_stddev.csv"
    std_table.to_csv(std_csv_path)

    # Markdown copy so the table can be pasted straight into the manuscript,
    # with the fold standard deviation shown as a +/- term in each cell.
    header = "| Model | " + " | ".join(f"{c} (kW)" for c in table.columns) + " |"
    divider = "|---" * (len(table.columns) + 1) + "|"
    lines = [
        "# Load composition comparison, test RMSE in kilowatts",
        "",
        "Mean test RMSE across 10 cross-validation folds, plus or minus the fold",
        "standard deviation. Forecast horizon 1 day. Models ordered by RMSE on the",
        "underlying load.",
        "",
        header,
        divider,
    ]
    for model in table.index:
        cells = [
            f"{table.loc[model, c]:.2f} +/- {std_table.loc[model, c]:.2f}"
            for c in table.columns
        ]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    md_path = out_dir / "paper_table_sa_bess_44hh_signal_test_rmse_kw.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return [csv_path, std_csv_path, md_path]


def _plot_grouped_model_bars(
    metric_table: pd.DataFrame,
    yerr_table: pd.DataFrame,
    *,
    ylabel: str,
    title: str,
    out_path: Path,
    highlight_lowest: bool = True,
) -> Path:
    """Draw one grouped bar per model within each load composition.

    Args:
        metric_table (pd.DataFrame): models by compositions, the bar heights.
        yerr_table (pd.DataFrame): same shape, the whisker half-lengths.
        ylabel (str): y axis label including units.
        title (str): figure title.
        out_path (Path): destination PNG.
        highlight_lowest (bool): mark the best model in each composition.

    Returns:
        Path: the file written.
    """
    x = np.arange(len(COMPOSITION_ORDER))
    total_width = 0.84
    bar_width = total_width / len(MODEL_SEQUENCE)
    best_by_composition = metric_table.idxmin(axis=0).to_dict() if highlight_lowest else {}

    fig, ax = plt.subplots(figsize=page_figsize(18, 10.5))
    for model_idx, model in enumerate(MODEL_SEQUENCE):
        offsets = x - total_width / 2 + bar_width / 2 + model_idx * bar_width
        heights = metric_table.loc[model, COMPOSITION_ORDER].to_numpy(dtype=float)
        yerr = yerr_table.loc[model, COMPOSITION_ORDER].to_numpy(dtype=float)
        bars = ax.bar(
            offsets,
            heights,
            width=bar_width * 0.92,
            label=MODEL_SHORT_LABELS[model],
            color=MODEL_SEQUENCE_COLORS[model],
            yerr=yerr,
            capsize=3,
            error_kw={"elinewidth": 1.1, "ecolor": "0.25", "capthick": 1.1},
        )
        if not highlight_lowest:
            continue
        # Twelve categorical colours cannot all be told apart, so the best model
        # in each group is marked by hatching and a star rather than by hue.
        for composition_idx, composition in enumerate(COMPOSITION_ORDER):
            if best_by_composition.get(composition) != model:
                continue
            bar = bars[composition_idx]
            bar.set_edgecolor("black")
            bar.set_linewidth(1.8)
            bar.set_hatch("///")
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                heights[composition_idx] + yerr[composition_idx] + 0.02 * float(np.nanmax(metric_table.to_numpy())),
                "*",
                ha="center",
                va="bottom",
                fontsize=TITLE_PT * 1.4,
                fontweight="bold",
                color="black",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([COMPOSITION_LABELS[key] for key in COMPOSITION_ORDER], rotation=0, fontsize=AXIS_LABEL_PT)
    ax.tick_params(axis="y", labelsize=TICK_PT)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=TICK_BINS))
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_PT)
    ax.set_xlabel("")
    ax.set_title(title, fontsize=TITLE_PT, pad=10)
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    ax.set_axisbelow(True)
    if highlight_lowest:
        ax.scatter([], [], marker="*", color="black", label="Lowest RMSE")
    ax.legend(
        title="Model / hyperparameter",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=TICK_PT,
        title_fontsize=AXIS_LABEL_PT,
    )
    fig.tight_layout()
    return save_figure(fig, out_path)


def _plot_composition_rmse_kw() -> Path:
    """Load composition model comparison in absolute kilowatts."""
    rmse, rmse_std = _composition_metric_tables()
    return _plot_grouped_model_bars(
        rmse,
        rmse_std,
        ylabel="Test RMSE (kW)",
        title="SA BESS 44-household clean cohort: 1-day model performance",
        out_path=RESULTS_DIR
        / "04_sa_bess_clean_44hh"
        / "figures"
        / "fig23_sa_bess_composition_test_rmse_kw.png",
    )


def _build_reference_mapping() -> tuple[Path, Path]:
    """Write the artefact-to-manuscript mapping in CSV and Markdown form.

    The manifest itself lives in notebook_artifact_build.PAPER_ARTIFACTS,
    which is the single source of truth for what the article ships. It is
    imported here rather than at module scope because that module imports this
    one. Keeping one copy of the list means a figure added to the build cannot
    silently go missing from the mapping.

    Returns:
        tuple[Path, Path]: the CSV and the Markdown file written.
    """
    import notebook_artifact_build

    written = notebook_artifact_build.build_tables_and_mapping()
    csv_path = RESULTS_DIR / "paper_artifact_reference_mapping.csv"
    md_path = RESULTS_DIR / "paper_artifact_reference_mapping.md"
    missing = [p for p in (csv_path, md_path) if p not in written]
    if missing:
        raise RuntimeError(f"Reference mapping was not written: {missing}")
    return csv_path, md_path


# ---------------------------------------------------------------------------
# Aggregation level comparison
# ---------------------------------------------------------------------------

AGGREGATION_LEVELS = [1, 10, 100, 1000]


def _aggregation_frames(recap_agg: pd.DataFrame) -> pd.DataFrame:
    """Prepare the per-sample and per-fold aggregation results for plotting.

    The denominator behind ``test_nRMSE`` is the maximum net load over each
    dataset's own cross-validation window, so every aggregation level is already
    normalised by its own peak. RMSE per household divides by the actual
    household weight rather than the nominal level, because the two differ where
    a sample could not be filled to the full count.

    Args:
        recap_agg (pd.DataFrame): the aggregation recap table.

    Returns:
        pd.DataFrame: one row per sample, carrying nRMSE and RMSE per household.
    """
    s = recap_agg.copy()
    for column in ["aggregation_level_hh", "test_nRMSE", "test_RMSE", "total_household_weight"]:
        s[column] = pd.to_numeric(s[column], errors="coerce")
    s["rmse_per_hh"] = s["test_RMSE"] / s["total_household_weight"]

    return s


def _draw_aggregation_panel(ax, samples: pd.DataFrame, model: str, *, sample_col: str) -> None:
    """Draw one model's aggregation trend: sample points and mean +/- SD."""
    x = np.arange(len(AGGREGATION_LEVELS))
    ax.set_axisbelow(True)

    model_samples = samples.loc[samples["model_name"].eq(model)]
    for i, level in enumerate(AGGREGATION_LEVELS):
        values = model_samples.loc[
            model_samples["aggregation_level_hh"].eq(level), sample_col
        ].dropna().to_numpy()
        if len(values) == 0:
            continue
        rng = np.random.default_rng(4321 + i)
        ax.scatter(
            np.full(len(values), i) + rng.uniform(-0.06, 0.06, size=len(values)),
            values,
            s=22,
            color=PALETTE["neutral"],
            alpha=0.55,
            edgecolors="white",
            linewidths=0.4,
            zorder=2,
        )

    means, stds = [], []
    for level in AGGREGATION_LEVELS:
        values = model_samples.loc[
            model_samples["aggregation_level_hh"].eq(level), sample_col
        ].dropna()
        means.append(float(values.mean()) if len(values) else np.nan)
        stds.append(float(values.std()) if len(values) > 1 else 0.0)

    # Every panel holds one model, named in its title, so the line style carries
    # no information; solid reads the slope most clearly. Colour still ties each
    # model to the same hue across figures.
    ax.errorbar(
        x,
        means,
        yerr=stds,
        fmt="o-",
        capsize=4,
        linewidth=1.8,
        markersize=5,
        color=MODEL_COLORS[model],
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(level) for level in AGGREGATION_LEVELS])
    _format_numeric_axis(ax)


def _plot_aggregation_two_panel(recap_agg: pd.DataFrame) -> Path:
    """Aggregation results as one figure carrying both metrics.

    Row (a) reports nRMSE, where each aggregation level is normalised by its own
    maximum positive net load, the convention used elsewhere in the article. Row
    (b) reports RMSE per household. The two rows move in opposite directions:
    aggregate peak per household falls faster than error per household, so the
    normalised error rises even as the absolute error per household falls.

    Args:
        recap_agg (pd.DataFrame): the aggregation recap table.

    Returns:
        Path: the figure written.
    """
    samples = _aggregation_frames(recap_agg)

    fig, axes = plt.subplots(2, 3, figsize=page_figsize(16, 11.6), sharex=True)
    rows = [
        ("(a)", "Test nRMSE (%)", "test_nRMSE"),
        ("(b)", "Test RMSE per household (kW)", "rmse_per_hh"),
    ]

    # Each row shares one y scale so levels stay comparable across models; the
    # two rows carry different units and must not share one.
    for row_idx, (panel_id, ylabel, sample_col) in enumerate(rows):
        for col_idx, model in enumerate(MODEL_ORDER):
            ax = axes[row_idx, col_idx]
            _draw_aggregation_panel(ax, samples, model, sample_col=sample_col)
            if row_idx == 0:
                ax.set_title(MODEL_SHORT_LABELS[model])
            if row_idx == len(rows) - 1:
                ax.set_xlabel("Aggregation level (households)")
            if col_idx == 0:
                ax.set_ylabel(f"{panel_id} {ylabel}")

        # One y scale per row, spanning every point drawn in that row. Sharing
        # the axes after plotting would keep the first panel's limits and clip
        # the others, so the limits are set explicitly here.
        row_axes = axes[row_idx]
        lo = min(ax.get_ylim()[0] for ax in row_axes)
        hi = max(ax.get_ylim()[1] for ax in row_axes)
        for col_idx, ax in enumerate(row_axes):
            ax.set_ylim(lo, hi)
            if col_idx > 0:
                ax.tick_params(labelleft=False)

    fig.suptitle("AEDP aggregation: sample mean +/- standard deviation", y=1.01)
    fig.tight_layout()
    return save_figure(
        fig,
        RESULTS_DIR
        / "03_aedp_aggregation_level"
        / "figures"
        / "fig15_aedp_agg_nrmse_and_rmse_per_hh_two_panel.png",
    )


def _aggregation_denominators(recap_agg: pd.DataFrame) -> pd.DataFrame:
    """Recover the peak net load each aggregation level was normalised by.

    The engine never records its denominator, so it is back-derived from the
    pair of metrics it did record: nRMSE = 100 * RMSE / peak.

    Args:
        recap_agg (pd.DataFrame): the aggregation recap table.

    Returns:
        pd.DataFrame: one row per aggregation level with the mean peak net load
        in kilowatts and that peak divided by the household count.
    """
    d = recap_agg.copy()
    for column in ["aggregation_level_hh", "test_RMSE", "test_nRMSE", "total_household_weight"]:
        d[column] = pd.to_numeric(d[column], errors="coerce")
    d["peak_kW"] = d["test_RMSE"] / d["test_nRMSE"] * 100.0
    summary = (
        d.groupby("aggregation_level_hh", as_index=False)
        .agg(peak_kW=("peak_kW", "mean"), household_weight=("total_household_weight", "mean"))
        .sort_values("aggregation_level_hh")
    )
    summary["peak_kW_per_household"] = summary["peak_kW"] / summary["household_weight"]
    return summary


def _plot_aggregation_fixed_denominator_supplementary(recap_agg: pd.DataFrame) -> Path:
    """Supplementary variant: nRMSE on a fixed per-household denominator.

    Normalises every aggregation level by N times the single-household peak
    rather than by that level's own peak. This is monotone-equivalent to RMSE
    per household, so it preserves the declining trend, but it invents a
    denominator that no other result in the article uses. Produced to document
    the alternative that was considered; not a manuscript figure.

    Args:
        recap_agg (pd.DataFrame): the aggregation recap table.

    Returns:
        Path: the figure written, under a supplementary subdirectory.
    """
    samples = _aggregation_frames(recap_agg)
    denominators = _aggregation_denominators(recap_agg)
    single_household_peak = float(
        denominators.loc[denominators["aggregation_level_hh"].eq(1), "peak_kW_per_household"].iloc[0]
    )

    samples = samples.copy()
    samples["fixed_nrmse"] = (
        100.0 * samples["test_RMSE"] / (samples["total_household_weight"] * single_household_peak)
    )

    fig, axes = plt.subplots(1, 3, figsize=page_figsize(16, 4.7), sharex=True, sharey=True)
    for ax, model in zip(axes, MODEL_ORDER):
        _draw_aggregation_panel(ax, samples, model, sample_col="fixed_nrmse")
        ax.set_title(MODEL_SHORT_LABELS[model])
        ax.set_xlabel("Aggregation level (households)")
        if ax is axes[0]:
            ax.set_ylabel("Test nRMSE (%)")

    fig.suptitle(
        "SUPPLEMENTARY - NOT A MANUSCRIPT FIGURE\n"
        "AEDP aggregation nRMSE on a fixed per-household denominator: "
        f"N x {single_household_peak:.3f} kW",
        y=1.12,
    )
    fig.tight_layout()
    return save_figure(
        fig,
        RESULTS_DIR
        / "03_aedp_aggregation_level"
        / "figures"
        / "supplementary"
        / "SUPPLEMENTARY_NOT_FOR_PAPER_aedp_agg_nrmse_fixed_per_hh_denominator.png",
    )



# ---------------------------------------------------------------------------
# Accuracy against stability
# ---------------------------------------------------------------------------

# Candidate label offsets in axes-fraction units, tried in order until one lands
# clear of the points and of the labels already placed.
# Candidate label positions, as (angle in degrees, radius in points) around the
# point being labelled. Rings are tried from the inside out so a label sits as
# close to its point as it can without colliding.
_LABEL_ANGLES = [20, 340, 70, 290, 140, 220, 0, 180, 110, 250, 45, 315]
_LABEL_RADII = [r * ANNOTATION_PT / 12.0 for r in (34.0, 52.0, 74.0, 100.0, 130.0)]

# Clearances in points: how far a label box and a leader line must stay from any
# plotted marker before the placement counts as clean.
_POINT_CLEARANCE = 9.0 * ANNOTATION_PT / 12.0
_LEADER_CLEARANCE = 7.0 * ANNOTATION_PT / 12.0


def _segment_point_distance(ax_, ay, bx, by, px, py) -> float:
    """Shortest distance from point p to the segment ab, all in display units."""
    dx, dy = bx - ax_, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return float(np.hypot(px - ax_, py - ay))
    t = max(0.0, min(1.0, ((px - ax_) * dx + (py - ay) * dy) / length_sq))
    return float(np.hypot(px - (ax_ + t * dx), py - (ay + t * dy)))


def _segments_cross(p1, p2, p3, p4) -> bool:
    """Whether segment p1-p2 properly intersects segment p3-p4."""

    def side(a, b, c) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = side(p3, p4, p1), side(p3, p4, p2)
    d3, d4 = side(p1, p2, p3), side(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _place_scatter_labels(fig, ax, points: list[tuple[float, float, str]]) -> None:
    """Label every scatter point without covering a point or another label.

    Labels are placed one at a time, densest neighbourhood first, because the
    crowded points are the ones with the fewest workable positions. For each
    point every candidate in :data:`_LABEL_ANGLES` x :data:`_LABEL_RADII` is
    scored and the cheapest is kept. The score charges, in decreasing order of
    weight, for leaving the axes, covering a plotted marker, overlapping a label
    already placed, and running the leader line close to another marker; ties
    break towards the shortest leader.

    Args:
        fig (matplotlib.figure.Figure): parent figure, needed for a renderer.
        ax (matplotlib.axes.Axes): the axes holding the scatter.
        points (list[tuple[float, float, str]]): x, y and label per point.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dpi_scale = fig.dpi / 72.0
    axes_box = ax.get_window_extent(renderer=renderer)

    marker_xy = [ax.transData.transform((px, py)) for px, py, _ in points]

    # Label the most crowded points first: they have the fewest free positions,
    # so letting them choose before the isolated points avoids dead ends.
    def crowding(index: int) -> float:
        x0, y0 = marker_xy[index]
        return -sum(
            1
            for j, (x1, y1) in enumerate(marker_xy)
            if j != index and np.hypot(x1 - x0, y1 - y0) < 120.0
        )

    order = sorted(range(len(points)), key=crowding)

    placed_boxes: list = []
    placed_leaders: list = []
    text_items: list[tuple[float, float, object]] = []

    for index in order:
        px, py, label = points[index]
        anchor_x, anchor_y = marker_xy[index]
        best = None

        for radius in _LABEL_RADII:
            for angle in _LABEL_ANGLES:
                theta = np.radians(angle)
                offset_x = radius * np.cos(theta) * dpi_scale
                offset_y = radius * np.sin(theta) * dpi_scale
                target_x, target_y = anchor_x + offset_x, anchor_y + offset_y
                data_x, data_y = ax.transData.inverted().transform((target_x, target_y))

                candidate = ax.text(
                    data_x,
                    data_y,
                    label,
                    ha="left" if offset_x >= 0 else "right",
                    va="center",
                    fontsize=ANNOTATION_PT,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "#CCCCCC",
                        "alpha": 0.9,
                        "boxstyle": "round,pad=0.18",
                    },
                    zorder=5,
                )
                box = candidate.get_window_extent(renderer=renderer)

                penalty = 0.0
                if box.x0 < axes_box.x0 or box.x1 > axes_box.x1:
                    penalty += 1000.0
                if box.y0 < axes_box.y0 or box.y1 > axes_box.y1:
                    penalty += 1000.0

                padded = box.expanded(
                    1.0 + 2 * _POINT_CLEARANCE * dpi_scale / max(box.width, 1.0),
                    1.0 + 2 * _POINT_CLEARANCE * dpi_scale / max(box.height, 1.0),
                )
                for j, (mx, my) in enumerate(marker_xy):
                    if j == index:
                        continue
                    if padded.x0 <= mx <= padded.x1 and padded.y0 <= my <= padded.y1:
                        penalty += 100.0

                penalty += 40.0 * sum(1 for other in placed_boxes if box.overlaps(other))

                # The leader runs from the marker to the centre of the box; charge
                # for passing close to any other marker on the way.
                cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
                for j, (mx, my) in enumerate(marker_xy):
                    if j == index:
                        continue
                    if _segment_point_distance(anchor_x, anchor_y, cx, cy, mx, my) < _LEADER_CLEARANCE * dpi_scale:
                        penalty += 20.0

                # Crossing leaders are readable but untidy, so they cost less
                # than a covered marker and more than a longer leader.
                for other_leader in placed_leaders:
                    if _segments_cross((anchor_x, anchor_y), (cx, cy), *other_leader):
                        penalty += 8.0

                # Prefer the shortest leader among otherwise equal candidates.
                penalty += radius / 1000.0

                if best is None or penalty < best[0]:
                    if best is not None:
                        best[1].remove()
                    best = (penalty, candidate, box, (cx, cy))
                else:
                    candidate.remove()

            if best is not None and best[0] < 1.0:
                break

        placed_boxes.append(best[2])
        placed_leaders.append(((anchor_x, anchor_y), best[3]))
        text_items.append((px, py, best[1]))

    # Leader lines stop at the outside edge of each rendered box, so the line
    # never runs across the text it points at.
    inverse = ax.transData.inverted()
    for px, py, text in text_items:
        box = text.get_window_extent(renderer=renderer).expanded(1.08, 1.18)
        (x0, y0), (x1, y1) = inverse.transform([[box.x0, box.y0], [box.x1, box.y1]])
        center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
        vx, vy = px - center_x, py - center_y
        scales = []
        if abs(vx) > 1e-12:
            scales.append(((x1 - center_x) / vx) if vx > 0 else ((x0 - center_x) / vx))
        if abs(vy) > 1e-12:
            scales.append(((y1 - center_y) / vy) if vy > 0 else ((y0 - center_y) / vy))
        positive = [s for s in scales if s > 0]
        if not positive:
            continue
        scale = min(positive)
        ax.plot(
            [px, center_x + vx * scale],
            [py, center_y + vy * scale],
            color="#777777",
            linewidth=0.5,
            zorder=2,
            solid_capstyle="round",
        )


def _draw_stability_panel(fig, ax, frame: pd.DataFrame, title: str) -> None:
    """Draw one accuracy-against-stability panel for a full model library."""
    frame = frame.sort_values("test_nRMSE")
    x = frame["test_nRMSE"].to_numpy(dtype=float)
    y = frame["test_nRMSE_stddev"].to_numpy(dtype=float)

    # Pad the limits before labelling so the boxes have somewhere to sit.
    x_pad = 0.20 * (x.max() - x.min())
    y_pad = 0.26 * (y.max() - y.min())
    ax.set_xlim(x.min() - x_pad, x.max() + x_pad)
    ax.set_ylim(max(0.0, y.min() - y_pad), y.max() + y_pad)

    ax.scatter(
        x,
        y,
        s=22,
        color=PALETTE["series_a"],
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.set_xlabel("Test nRMSE (%)")
    ax.set_ylabel("Test nRMSE stddev (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=TICK_BINS))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=TICK_BINS))

    labels = [
        (float(row.test_nRMSE), float(row.test_nRMSE_stddev), MODEL_SHORT_LABELS.get(str(row.model_name), str(row.model_name)))
        for row in frame.itertuples(index=False)
    ]
    _place_scatter_labels(fig, ax, labels)


def _validate_stability_frame(frame: pd.DataFrame, description: str) -> pd.DataFrame:
    """Check a scatter input covers the full model library exactly once."""
    if frame.empty:
        raise ValueError(f"No rows found for {description}.")
    missing = [m for m in MODEL_SEQUENCE if m not in set(frame["model_name"].astype(str))]
    if missing:
        raise ValueError(f"{description} is missing models: {missing}")
    return frame


def _plot_ashd_stability_scatter() -> Path:
    """Accuracy against stability for ASHD at the 1-week horizon.

    Returns:
        Path: the figure written.
    """
    recap = _load_csv(
        RESULTS_DIR / "02_ashd_148hh_forecast_horizon" / "ashd_148hh_horizon_combined_recap.csv"
    )
    frame = recap.loc[pd.to_numeric(recap["forecast_horizon_min"], errors="coerce").eq(10080)].copy()
    frame = _validate_stability_frame(frame, "ASHD at the 1-week horizon")

    fig, ax = plt.subplots(figsize=page_figsize(8.6, 6.0))
    _draw_stability_panel(fig, ax, frame, "ASHD 148-household dataset, 1-week forecast horizon")
    fig.tight_layout()
    return save_figure(
        fig,
        RESULTS_DIR
        / "02_ashd_148hh_forecast_horizon"
        / "figures"
        / "fig42_ashd_nrmse_stability_scatter_1week.png",
    )


def _plot_dataset_stability_scatter_pair() -> Path:
    """Accuracy against stability for both datasets at the 1-day horizon.

    ASHD is the only dataset with a 1-week run, so the two-dataset comparison is
    drawn at 1 day, the longest horizon both datasets share. The model ordering
    on ASHD is close between the two horizons, so the 1-day panel carries the
    same message as the 1-week single-dataset figure.

    The two panels keep independent axis scales because their ranges do not
    overlap: forcing a shared scale would compress each panel into a corner and
    hide the within-dataset spread that the figure exists to show.

    Returns:
        Path: the figure written.
    """
    recap = _load_csv(
        RESULTS_DIR / "01_ashd_aedp_148hh_comparison" / "ashd_aedp_148hh_fh8_combined_recap.csv"
    )
    panels = [
        ("(a)", "ASHD_148hh_weather", "ASHD 148-household dataset"),
        ("(b)", "AEDP_148hh_weather", "AEDP 148-household dataset"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=page_figsize(17, 8.25))
    for ax, (panel_id, dataset_label, pretty) in zip(axes, panels):
        frame = recap.loc[recap["dataset_label"].astype(str).eq(dataset_label)].copy()
        frame = _validate_stability_frame(frame, f"{dataset_label} at the 1-day horizon")
        _draw_stability_panel(fig, ax, frame, f"{panel_id} {pretty}")

    fig.suptitle("Forecast accuracy against cross-validation stability, 1-day forecast horizon", y=1.02)
    fig.tight_layout()
    return save_figure(
        fig,
        RESULTS_DIR
        / "01_ashd_aedp_148hh_comparison"
        / "figures"
        / "fig32_ashd_aedp_nrmse_stability_scatter_1day.png",
    )


def _write_pending_report(notes: list[str], failures: list[str]) -> Path:
    report_path = RESULTS_DIR / "paper_figure_pending_items.md"
    lines = [
        "# Supervisor Revision Pending Items",
        "",
        "This report lists items that were skipped or generated with fallback data in fast mode.",
        "",
        "## Fallback Notes",
    ]
    if notes:
        lines.extend([f"- {n}" for n in notes])
    else:
        lines.append("- None")

    lines.append("")
    lines.append("## Generation Failures")
    if failures:
        lines.extend([f"- {f}" for f in failures])
    else:
        lines.append("- None")

    lines.append("")
    lines.append("## Deferred Long-Run Items")
    lines.append("- Optional rerun for ds11 fh8 xgb experiments (if strict ASHD-vs-AEDP matched dataset requirement must be enforced).")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the journal article figures and tables from existing experiment results.")
    parser.add_argument(
        "--allow-rerun",
        action="store_true",
        help="Allow rerunning missing experiments (disabled by default for fast progress mode).",
    )
    parser.add_argument(
        "--strict-ashd-aedp",
        action="store_true",
        help="Require strict ds11 AEDP source for ASHD-vs-AEDP figures (no fallback).",
    )
    args = parser.parse_args()

    apply_publication_style()
    _replace_legacy_section_figures()

    notes: list[str] = []
    failures: list[str] = []

    ds11_available = _ensure_ds11_xgb_exists(allow_rerun=args.allow_rerun)
    if not ds11_available:
        notes.append("ds11 fh8 xgb experiment not available in local recap; fast mode kept reruns disabled.")

    recap_exp = _load_publication_recap()

    recap_agg = _load_csv(RESULTS_DIR / "03_aedp_aggregation_level" / "aedp_aggregation_fh8_recap.csv")
    recap_sa = _load_csv(RESULTS_DIR / "04_sa_bess_clean_44hh" / "sa_bess_44hh_fh8_combined_recap.csv")

    created: list[Path] = []
    steps = [
        ("table_experiment_parameters", lambda: [_build_experiment_parameters_table()]),
        ("table_pynnlf_outputs", lambda: [_build_pynnlf_outputs_table()]),
        ("table_model_selection", lambda: [_build_model_selection_recommendation()]),
        ("table_reference_mapping", lambda: list(_build_reference_mapping())),
        ("fig_aggregation_summary", lambda: _plot_aggregation_summary_and_cv(recap_agg)),
        ("fig_aggregation_forecast_views", lambda: _plot_aggregation_forecast_views(recap_agg)),
        ("fig_aggregation_xgb_timeseries", lambda: [_plot_aggregation_xgb_timeseries(recap_agg)]),
        ("fig_load_composition", lambda: _plot_load_composition_timeseries(recap_sa)),
        (
            "fig_ashd_vs_aedp",
            lambda: _plot_ashd_vs_aedp_xgb(
                recap_exp,
                recap_agg,
                notes,
                strict_ashd_aedp=args.strict_ashd_aedp,
            ),
        ),
        ("fig_horizon_xgb", lambda: _plot_horizon_xgb(recap_exp)),
    ]

    for step_name, step_fn in steps:
        try:
            created.extend(step_fn())
            _print(f"Completed step: {step_name}")
        except Exception as exc:
            msg = f"{step_name}: {exc}"
            failures.append(msg)
            _print(f"Skipped step due to error: {msg}")

    pending_path = _write_pending_report(notes, failures)
    created.append(pending_path)

    checks = _check_outputs(created)
    check_path = RESULTS_DIR / "paper_figure_output_check.csv"
    checks.to_csv(check_path, index=False)
    created.append(check_path)

    _print("Generated outputs:")
    for p in created:
        _print(f" - {p.relative_to(WORKSPACE_DIR)}")


# Entry by notebooks/imported helpers only.
