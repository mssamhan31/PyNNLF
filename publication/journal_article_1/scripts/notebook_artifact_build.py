from __future__ import annotations

from pathlib import Path

import pandas as pd

import generate_sa_bess_combined_figures as sb
import generate_paper_figures as gsr
from publication_plot_style import apply_publication_style


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = WORKSPACE_DIR / "results"


PAPER_ARTIFACTS: list[dict[str, str]] = [
    {
        "artifact_path": "results/paper_table_experiment_parameters.csv",
        "artifact_type": "table",
        "manuscript_section": "Methods - Experiment setup",
        "manuscript_label_proposed": "Table M1",
        "caption_short": "Experiment parameters: predictors, lags, horizons, CV, models, metrics",
        "status": "ready",
    },
    {
        "artifact_path": "results/paper_table_pynnlf_output_artifacts.csv",
        "artifact_type": "table",
        "manuscript_section": "Methods - PyNNLF workflow outputs",
        "manuscript_label_proposed": "Table M2",
        "caption_short": "PyNNLF output files, contents, and purpose",
        "status": "ready",
    },
    {
        "artifact_path": "results/paper_table_model_selection_recommendation.csv",
        "artifact_type": "table",
        "manuscript_section": "Results - Model selection recommendation",
        "manuscript_label_proposed": "Table M3",
        "caption_short": "Pareto, utility, and satisficing recommendation outputs",
        "status": "ready",
    },
    {
        "artifact_path": "results/03_aedp_aggregation_level/figures/fig10_aedp_agg_nrmse_mean_std_cvbg.png",
        "artifact_type": "figure",
        "manuscript_section": "Results - Aggregation level comparison",
        "manuscript_label_proposed": "Figure A1",
        "caption_short": "Aggregation nRMSE: sample mean +- SD with fold background",
        "status": "ready",
    },
    {
        "artifact_path": "results/03_aedp_aggregation_level/figures/fig11_aedp_agg_rmse_per_hh_mean_std_cvbg.png",
        "artifact_type": "figure",
        "manuscript_section": "Results - Aggregation level comparison",
        "manuscript_label_proposed": "Figure A2",
        "caption_short": "Aggregation RMSE per household: sample mean +- SD with fold background",
        "status": "ready",
    },
    {
        "artifact_path": "results/03_aedp_aggregation_level/figures/fig15_aedp_agg_nrmse_and_rmse_per_hh_two_panel.png",
        "artifact_type": "figure",
        "manuscript_section": "Results - Aggregation level comparison",
        "manuscript_label_proposed": "Figure A0",
        "caption_short": "Aggregation nRMSE and RMSE per household in one two-row figure",
        "status": "ready",
    },
    {
        "artifact_path": "results/03_aedp_aggregation_level/figures/supplementary/SUPPLEMENTARY_NOT_FOR_PAPER_aedp_agg_nrmse_fixed_per_hh_denominator.png",
        "artifact_type": "figure",
        "manuscript_section": "Not for the manuscript",
        "manuscript_label_proposed": "-",
        "caption_short": "Supplementary: aggregation nRMSE on a fixed per-household denominator",
        "status": "supplementary",
    },
    {
        "artifact_path": "results/03_aedp_aggregation_level/figures/fig12a_aedp_agg1_actual_vs_forecast_timeseries_naive_lr_xgb.png",
        "artifact_type": "figure",
        "manuscript_section": "Results - Aggregation level comparison",
        "manuscript_label_proposed": "Figure A3(a)",
        "caption_short": "Actual vs forecast (1 household)",
        "status": "ready",
    },
    {
        "artifact_path": "results/03_aedp_aggregation_level/figures/fig12b_aedp_agg10_actual_vs_forecast_timeseries_naive_lr_xgb.png",
        "artifact_type": "figure",
        "manuscript_section": "Results - Aggregation level comparison",
        "manuscript_label_proposed": "Figure A3(b)",
        "caption_short": "Actual vs forecast (10 households)",
        "status": "ready",
    },
    {
        "artifact_path": "results/03_aedp_aggregation_level/figures/fig12c_aedp_agg100_actual_vs_forecast_timeseries_naive_lr_xgb.png",
        "artifact_type": "figure",
        "manuscript_section": "Results - Aggregation level comparison",
        "manuscript_label_proposed": "Figure A3(c)",
        "caption_short": "Actual vs forecast (100 households)",
        "status": "ready",
    },
    {
        "artifact_path": "results/03_aedp_aggregation_level/figures/fig12d_aedp_agg1000_actual_vs_forecast_timeseries_naive_lr_xgb.png",
        "artifact_type": "figure",
        "manuscript_section": "Results - Aggregation level comparison",
        "manuscript_label_proposed": "Figure A3(d)",
        "caption_short": "Actual vs forecast (1000 households)",
        "status": "ready",
    },
    {
        "artifact_path": "results/03_aedp_aggregation_level/figures/fig13a_aedp_agg1_actual_vs_forecast_scatter_naive_lr_xgb.png",
        "artifact_type": "figure",
        "manuscript_section": "Results - Aggregation level comparison",
        "manuscript_label_proposed": "Figure A4(a)",
        "caption_short": "Scatter actual vs forecast (1 household)",
        "status": "ready",
    },
    {
        "artifact_path": "results/03_aedp_aggregation_level/figures/fig13b_aedp_agg10_actual_vs_forecast_scatter_naive_lr_xgb.png",
        "artifact_type": "figure",
        "manuscript_section": "Results - Aggregation level comparison",
        "manuscript_label_proposed": "Figure A4(b)",
        "caption_short": "Scatter actual vs forecast (10 households)",
        "status": "ready",
    },
    {
        "artifact_path": "results/03_aedp_aggregation_level/figures/fig13c_aedp_agg100_actual_vs_forecast_scatter_naive_lr_xgb.png",
        "artifact_type": "figure",
        "manuscript_section": "Results - Aggregation level comparison",
        "manuscript_label_proposed": "Figure A4(c)",
        "caption_short": "Scatter actual vs forecast (100 households)",
        "status": "ready",
    },
    {
        "artifact_path": "results/03_aedp_aggregation_level/figures/fig13d_aedp_agg1000_actual_vs_forecast_scatter_naive_lr_xgb.png",
        "artifact_type": "figure",
        "manuscript_section": "Results - Aggregation level comparison",
        "manuscript_label_proposed": "Figure A4(d)",
        "caption_short": "Scatter actual vs forecast (1000 households)",
        "status": "ready",
    },
    {
        "artifact_path": "results/03_aedp_aggregation_level/figures/fig14_aedp_aggregation_xgb_timeseries_aligned.png",
        "artifact_type": "figure",
        "manuscript_section": "Results - Aggregation level comparison",
        "manuscript_label_proposed": "Figure A5",
        "caption_short": "Three-day time-aligned XGBoost actual vs forecast across 1/10/100/1000 households",
        "status": "ready",
    },
    {
        "artifact_path": "results/04_sa_bess_clean_44hh/figures/fig20_sa_bess_composition_timeseries_error_combined.png",
        "artifact_type": "figure",
        "manuscript_section": "Results - Load composition comparison",
        "manuscript_label_proposed": "Figure B1",
        "caption_short": "Composition time-series and error comparison in one figure",
        "status": "ready",
    },
    {
        "artifact_path": "results/04_sa_bess_clean_44hh/figures/fig21_sa_bess_hourly_load_ramp_combined.png",
        "artifact_type": "figure",
        "manuscript_section": "Results - Load composition comparison",
        "manuscript_label_proposed": "Figure B2",
        "caption_short": "Hourly load and ramp distributions by composition",
        "status": "ready",
    },
    {
        "artifact_path": "results/04_sa_bess_clean_44hh/figures/fig22_sa_bess_pv_battery_distribution_combined.png",
        "artifact_type": "figure",
        "manuscript_section": "Results - Load composition comparison",
        "manuscript_label_proposed": "Figure B3",
        "caption_short": "Hourly PV generation and battery net charge distributions",
        "status": "ready",
    },
    {
        "artifact_path": "results/04_sa_bess_clean_44hh/paper_table_sa_bess_44hh_signal_test_rmse_kw.csv",
        "artifact_type": "table",
        "manuscript_section": "Results - Load composition comparison",
        "manuscript_label_proposed": "Table B1",
        "caption_short": "Load composition test RMSE in kilowatts, 12 models",
        "status": "ready",
    },
    {
        "artifact_path": "results/04_sa_bess_clean_44hh/paper_table_sa_bess_44hh_signal_test_rmse_kw.md",
        "artifact_type": "table",
        "manuscript_section": "Results - Load composition comparison",
        "manuscript_label_proposed": "Table B1",
        "caption_short": "Load composition test RMSE in kilowatts, formatted for the manuscript",
        "status": "ready",
    },
    {
        "artifact_path": "results/04_sa_bess_clean_44hh/figures/fig23_sa_bess_composition_test_rmse_kw.png",
        "artifact_type": "figure",
        "manuscript_section": "Results - Load composition comparison",
        "manuscript_label_proposed": "Figure B4",
        "caption_short": "Load composition model comparison in absolute kilowatts",
        "status": "ready",
    },
    {
        "artifact_path": "results/01_ashd_aedp_148hh_comparison/figures/fig32_ashd_aedp_nrmse_stability_scatter_1day.png",
        "artifact_type": "figure",
        "manuscript_section": "Results - Accuracy against stability",
        "manuscript_label_proposed": "Figure C3",
        "caption_short": "Accuracy against stability for ASHD and AEDP at the 1-day horizon",
        "status": "ready",
    },
    {
        "artifact_path": "results/02_ashd_148hh_forecast_horizon/figures/fig42_ashd_nrmse_stability_scatter_1week.png",
        "artifact_type": "figure",
        "manuscript_section": "Results - Accuracy against stability",
        "manuscript_label_proposed": "Figure D3",
        "caption_short": "Accuracy against stability for ASHD at the 1-week horizon",
        "status": "ready",
    },
    {
        "artifact_path": "results/01_ashd_aedp_148hh_comparison/figures/fig30_ashd_vs_aedp_xgb_actual_vs_forecast_daypair.png",
        "artifact_type": "figure",
        "manuscript_section": "Results - Dataset comparison",
        "manuscript_label_proposed": "Figure C1",
        "caption_short": "XGBoost actual vs forecast for ASHD vs AEDP with shared peak-normalized scale",
        "status": "ready",
    },
    {
        "artifact_path": "results/01_ashd_aedp_148hh_comparison/figures/fig31_ashd_vs_aedp_xgb_error_profile_daypair.png",
        "artifact_type": "figure",
        "manuscript_section": "Results - Dataset comparison",
        "manuscript_label_proposed": "Figure C2",
        "caption_short": "XGBoost error profile for ASHD vs AEDP",
        "status": "ready",
    },
    {
        "artifact_path": "results/02_ashd_148hh_forecast_horizon/figures/fig40_ashd_horizons_xgb_actual_vs_forecast.png",
        "artifact_type": "figure",
        "manuscript_section": "Results - Forecast horizon comparison",
        "manuscript_label_proposed": "Figure D1",
        "caption_short": "XGBoost actual vs forecast for 30-min, 1-day, and 1-week horizons",
        "status": "ready",
    },
    {
        "artifact_path": "results/02_ashd_148hh_forecast_horizon/figures/fig41_ashd_horizons_xgb_error_summary.png",
        "artifact_type": "figure",
        "manuscript_section": "Results - Forecast horizon comparison",
        "manuscript_label_proposed": "Figure D2",
        "caption_short": "XGBoost horizon error summary",
        "status": "ready",
    },
]


SECTION_FILTERS = {
    "00": ["results/paper_table_", "results/paper_artifact_reference_mapping."],
    "01": ["results/01_ashd_aedp_148hh_comparison/"],
    "02": ["results/02_ashd_148hh_forecast_horizon/"],
    "03": ["results/03_aedp_aggregation_level/"],
    "04": [
        "results/04_sa_bess_clean_44hh/figures/fig20_",
        "results/04_sa_bess_clean_44hh/figures/fig21_",
        "results/04_sa_bess_clean_44hh/figures/fig22_",
        "results/04_sa_bess_clean_44hh/figures/fig23_",
        "results/04_sa_bess_clean_44hh/paper_table_sa_bess_44hh_signal_test_rmse_kw",
    ],
}


def _artifact_df() -> pd.DataFrame:
    return pd.DataFrame(PAPER_ARTIFACTS)


def ensure_required_inputs() -> None:
    required_files = [
        WORKSPACE_DIR / "experiment_result" / "a1_experiment_result.csv",
        RESULTS_DIR / "03_aedp_aggregation_level" / "aedp_aggregation_fh8_recap.csv",
        RESULTS_DIR / "04_sa_bess_clean_44hh" / "sa_bess_44hh_fh8_combined_recap.csv",
    ]
    missing = [p for p in required_files if not p.exists()]
    if missing:
        msg = "Missing required existing outputs (rerun prohibited):\n" + "\n".join(str(p) for p in missing)
        raise FileNotFoundError(msg)


def remove_stale_trial_artifacts() -> list[Path]:
    removed: list[Path] = []
    targets = [
        RESULTS_DIR / "01_ashd_aedp_148hh_comparison" / "figures" / "fig30_alternatives",
        RESULTS_DIR / "01_ashd_aedp_148hh_comparison" / "figures" / "fig30_same_season_alternatives",
        RESULTS_DIR / "02_ashd_148hh_forecast_horizon" / "figures" / "fig40_common_target_alternatives",
        RESULTS_DIR / "04_sa_bess_clean_44hh" / "figures" / "fig20_sa_bess_composition_actual_vs_forecast_timeseries.png",
        RESULTS_DIR / "04_sa_bess_clean_44hh" / "figures" / "fig21_sa_bess_composition_error_timeseries.png",
    ]

    for t in targets:
        if t.is_dir() and t.exists():
            for child in sorted(t.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            t.rmdir()
            removed.append(t)
        elif t.is_file() and t.exists():
            t.unlink()
            removed.append(t)
    return removed


def build_tables_and_mapping() -> list[Path]:
    apply_publication_style()
    out: list[Path] = []
    out.append(gsr._build_experiment_parameters_table())
    out.append(gsr._build_pynnlf_outputs_table())
    out.append(gsr._build_model_selection_recommendation())

    df = _artifact_df()
    csv_path = RESULTS_DIR / "paper_artifact_reference_mapping.csv"
    md_path = RESULTS_DIR / "paper_artifact_reference_mapping.md"
    df.to_csv(csv_path, index=False)
    out.append(csv_path)

    lines = ["# Paper Artifact Reference Mapping", "", "| Artifact | Type | Section | Label | Status |", "|---|---|---|---|---|"]
    for r in df.itertuples(index=False):
        lines.append(
            f"| {r.artifact_path} | {r.artifact_type} | {r.manuscript_section} | {r.manuscript_label_proposed} | {r.status} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.append(md_path)
    return out


def build_section_03() -> list[Path]:
    apply_publication_style()
    recap_agg = gsr._load_csv(RESULTS_DIR / "03_aedp_aggregation_level" / "aedp_aggregation_fh8_recap.csv")
    out: list[Path] = []
    out.extend(gsr._plot_aggregation_summary_and_cv(recap_agg))
    out.append(gsr._plot_aggregation_two_panel(recap_agg))
    out.append(gsr._plot_aggregation_fixed_denominator_supplementary(recap_agg))
    out.extend(gsr._plot_aggregation_forecast_views(recap_agg))
    out.append(gsr._plot_aggregation_xgb_timeseries(recap_agg))
    return out


def build_section_02() -> list[Path]:
    apply_publication_style()
    recap_exp = gsr._load_publication_recap()
    out = list(gsr._plot_horizon_xgb(recap_exp))
    out.append(gsr._plot_ashd_stability_scatter())
    return out


def build_section_01() -> list[Path]:
    apply_publication_style()
    recap_exp = gsr._load_publication_recap()
    recap_agg = gsr._load_csv(RESULTS_DIR / "03_aedp_aggregation_level" / "aedp_aggregation_fh8_recap.csv")
    notes: list[str] = []
    out = list(gsr._plot_ashd_vs_aedp_xgb(recap_exp, recap_agg, notes, strict_ashd_aedp=True))
    out.append(gsr._plot_dataset_stability_scatter_pair())
    return out


def build_section_04() -> list[Path]:
    apply_publication_style()
    out: list[Path] = []
    out.append(sb._generate_combined_fig20_21())
    out.append(sb._generate_combined_fig22_23())
    out.append(sb._generate_combined_fig24_25())
    out.append(gsr._plot_composition_rmse_kw())
    out.extend(gsr._build_composition_rmse_kw_table())
    return out


def check_artifacts(section: str | None = None) -> pd.DataFrame:
    df = _artifact_df().copy()
    if section is not None:
        prefixes = SECTION_FILTERS[section]
        mask = df["artifact_path"].apply(lambda p: any(str(p).startswith(px) for px in prefixes))
        df = df.loc[mask].copy()

    rows = []
    for p in df["artifact_path"].tolist():
        path = WORKSPACE_DIR / p
        rows.append({
            "path": p,
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
        })

    out_df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "paper_artifact_output_check.csv"
    out_df.to_csv(out_path, index=False)
    return out_df


def assert_artifacts(section: str | None = None) -> None:
    out_df = check_artifacts(section)
    missing = out_df.loc[~out_df["exists"].astype(bool)]
    if not missing.empty:
        raise AssertionError(f"Missing artifacts:\n{missing[['path']].to_string(index=False)}")


def list_expected_artifacts(section: str | None = None) -> list[str]:
    df = _artifact_df()
    if section is None:
        return df["artifact_path"].tolist()
    prefixes = SECTION_FILTERS[section]
    return [p for p in df["artifact_path"].tolist() if any(p.startswith(px) for px in prefixes)]
