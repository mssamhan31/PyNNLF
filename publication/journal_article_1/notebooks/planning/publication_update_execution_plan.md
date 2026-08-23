# Publication Update Execution Plan (Plan Only, No Implementation Yet)

## 1. Purpose
This plan defines how to update the publication code, rerun experiments, and regenerate publication-ready tables (CSV) and figures (PNG) for journal article 1, based on supervisor feedback and your requested updates.

This document is intentionally implementation-ready, but no code/notebook changes are performed in this step.

## 2. Inputs Reviewed
- Supervisor tracker:
  - `PyNNLF_Supervisor_Feedback_Tracker_v3.xlsx`
- Current draft manuscript:
  - `5. PyNNLF Extension_BY_SS.docx`
- Current publication workspace:
  - `publication/journal_article_1`

## 3. Scope and Non-Scope
### In Scope
- Replace/extend figure generation workflows for:
  - aggregation-level comparison
  - load composition comparison
  - ASHD vs AEDP comparison
  - forecast horizon comparison
  - model-selection recommendation output
- Add CSV tables requested by supervisor.
- Keep folder structure intact and output organization clear by experiment section.
- Ensure publication-ready and consistent visual style across all figures.

### Out of Scope (for this implementation cycle)
- Full manuscript rewriting (background/literature sections, major text edits), except where figure/table captions or references need updates.
- New model development beyond what is needed to generate the requested outputs.

## 4. Supervisor Feedback -> Planned Action Mapping (Figure/Table Focus)

| Tracker Ref | Supervisor Need | Planned Action |
|---|---|---|
| #41, #37, #42 | Aggregation-level interpretation issue with only distribution style and n=3 samples | Replace distribution emphasis with mean, stddev, and fold-level background points for both nRMSE and RMSE/household. Add clear notes for n=3 samples + CV fold context. |
| #42 | Show forecast vs real across aggregation levels | Add time-series actual vs forecast plots for selected model(s), with subplots by aggregation level. |
| #31 | Explain why AEDP is harder than ASHD | Add actual vs forecast comparison for same model (XGBoost) on ASHD and AEDP, selecting representative days with higher AEDP volatility/irregularity. |
| #35 | Improve forecast-horizon insight visuals | Add actual vs forecast for 30-min, 1-day, and 1-week horizons. |
| #43, #40 | Highlight load composition dynamics (especially battery effects) | Add time-series performance visuals across underlying load, net load with PV, and net load with PV+battery. |
| #30 | Replace prose list of outputs with table | Add PyNNLF output table CSV: output artifact, content, purpose, usage. |
| #16 | Better parameter documentation | Add experiment parameters table CSV for predictors, lags, horizons, CV, aggregation, models, hyperparameters, metrics. |
| #45 | Tie model selection methods to practical PyNNLF recommendation | Add recommendation output table (or figure if needed), with metric, constraints, eligible models, recommended model. |

## 5. Target File/Folder Strategy (Keep Existing Structure)

## Existing experiment result roots (retain)
- `publication/journal_article_1/results/01_ashd_aedp_148hh_comparison`
- `publication/journal_article_1/results/02_ashd_148hh_forecast_horizon`
- `publication/journal_article_1/results/03_aedp_aggregation_level`
- `publication/journal_article_1/results/04_sa_bess_clean_44hh`

## Planned additions (within same structure)
- New figures saved into each section's `figures/` subfolder.
- New CSV tables saved in section root, with explicit `paper_table_...` style naming.
- Model-selection recommendation table saved in a dedicated section root (recommended under horizon or summary section), with stable filename.

## 6. Visual Style Specification (Publication-Ready Standard)
Apply this globally to all figure notebooks/scripts.

### Font and text
- Font family: Arial (all labels, titles, legends, annotations).
- Consistent font sizes by element type (title, axis, ticks, legend).
- Avoid overlaps: use constrained/tight layouts, legend placement rules, max label lengths.

### Color palette (fixed)
- Orange: `#eb932c`
- Dark blue: `#22303d`
- Grey: `#2F4D67`
- Light grey: `#5C7D99`
- Light white: `#ebe3e3`

### Figure formatting rules
- Consistent DPI (300 export).
- Consistent line widths, marker sizes, grid alpha, and axis style.
- Legend placement outside plotting area when needed.
- Explicit subplot labels `(a)`, `(b)`, `(c)` where comparisons are multi-panel.

## 7. Detailed Execution Plan by Experiment

## Phase A: Shared plotting utilities and style guardrail
### Goal
Prevent style drift and avoid repeating figure-style code in every notebook.

### Planned work
1. Create a reusable plotting style helper (script/module) under publication workspace scripts.
2. Include:
   - global matplotlib rcParams setup (Arial + publication defaults)
   - named palette constants (exact hex values)
   - helper for consistent legend placement and subplot labeling
   - helper for file save with standard PNG settings
3. Update figure notebooks to import and use this helper.

### QA checks
- Every newly generated PNG visually uses the same font/palette.
- No clipping/overlap in titles, labels, legends.

## Phase B: Aggregation-level experiment updates
### Existing touchpoints
- `notebooks/03_aedp_aggregation_level/4_process_pynnlf_output_aedp_aggregation.ipynb`
- `notebooks/03_aedp_aggregation_level/5_visualise_pynnlf_aedp_aggregation.ipynb`
- `results/03_aedp_aggregation_level/*.csv`

### Required updates
1. Replace distribution-centric presentation with summary-centric plots:
   - mean of 3 samples
   - sample stddev
   - CV fold-level values shown lightly in background if available
2. Do this for both:
   - nRMSE plot
   - RMSE per household plot
3. Add time-series actual vs forecast (naive, lr, xgboost).
4. Add scatter actual vs forecast (naive, lr, xgboost).
5. Use separate figure files per aggregation level for both actual-vs-forecast time-series and actual-vs-forecast scatter.
6. Fully replace legacy draft-style figures for this section with the new figure set.

### Planned output naming (proposed)
- `fig10_aedp_agg_nrmse_mean_std_cvbg.png`
- `fig11_aedp_agg_rmse_per_hh_mean_std_cvbg.png`
- `fig12a_aedp_agg1_actual_vs_forecast_timeseries_naive_lr_xgb.png`
- `fig12b_aedp_agg10_actual_vs_forecast_timeseries_naive_lr_xgb.png`
- `fig12c_aedp_agg100_actual_vs_forecast_timeseries_naive_lr_xgb.png`
- `fig12d_aedp_agg1000_actual_vs_forecast_timeseries_naive_lr_xgb.png`
- `fig13a_aedp_agg1_actual_vs_forecast_scatter_naive_lr_xgb.png`
- `fig13b_aedp_agg10_actual_vs_forecast_scatter_naive_lr_xgb.png`
- `fig13c_aedp_agg100_actual_vs_forecast_scatter_naive_lr_xgb.png`
- `fig13d_aedp_agg1000_actual_vs_forecast_scatter_naive_lr_xgb.png`

### Data logic notes
- Clarify that aggregation level has 3 samples; fold-level points are not independent samples but useful context.
- Ensure per-household normalization is explicit in axis label/caption.

## Phase C: Load composition experiment updates
### Existing touchpoints
- `notebooks/04_sa_bess_clean_44hh/7_process_pynnlf_output_sa_bess_44hh.ipynb`
- `notebooks/04_sa_bess_clean_44hh/8_visualise_pynnlf_sa_bess_44hh.ipynb`

### Required updates
1. Add time-series performance plots across compositions:
   - underlying load
   - net load with PV
   - net load with PV+battery
2. Use comparable windows and scale strategy so dynamics are interpretable.
3. Add annotations or panel notes to highlight composition-specific forecast behavior.

### Planned output naming (proposed)
- `fig20_sa_bess_composition_actual_vs_forecast_timeseries.png`
- `fig21_sa_bess_composition_error_timeseries.png`

## Phase D: ASHD vs AEDP comparison updates
### Existing touchpoints
- `notebooks/01_ashd_aedp_148hh_comparison/2_visualise_pynnlf_ashd_aedp_148hh.ipynb`
- `notebooks/01_ashd_aedp_148hh_comparison/3_create_paper_figure_05_ashd_aedp_key_models.ipynb`

### Required updates
1. Add actual vs forecast using same model (XGBoost) for:
   - (a) ASHD
   - (b) AEDP
2. Select representative days that demonstrate AEDP is harder (volatility/irregularity) using a deterministic volatility-score rule, then choose from top candidates.
3. Include transparent day-selection rule in notebook text (not cherry-picking).

### Planned output naming (proposed)
- `fig30_ashd_vs_aedp_xgb_actual_vs_forecast_daypair.png`
- `fig31_ashd_vs_aedp_xgb_error_profile_daypair.png`

## Phase E: Forecast horizon comparison updates
### Existing touchpoints
- `notebooks/02_ashd_148hh_forecast_horizon/3_visualise_pynnlf_ashd_148hh_horizons.ipynb`
- `notebooks/02_ashd_148hh_forecast_horizon/4_create_paper_figures_06_07_ashd_horizons.ipynb`

### Required updates
1. Add actual vs forecast figure with three panels:
   - (a) 30-minute horizon
   - (b) 1-day horizon
   - (c) 1-week horizon
2. Use XGBoost only.
3. Align times and include clear caption about horizon interpretation.

### Planned output naming (proposed)
- `fig40_ashd_horizons_xgb_actual_vs_forecast.png`
- `fig41_ashd_horizons_xgb_error_summary.png` (optional)

## Phase F: PyNNLF recommendation output demonstration
### Existing touchpoints
- Model-selection analysis currently described conceptually in manuscript.
- Candidate processed results available in horizon/summary outputs.

### Required updates
1. Implement recommendation output artifact showing all three methods:
   - Pareto optimization
   - utility-based ranking
   - satisficing threshold filtering
2. Use a clear table as the primary output (supervisor-aligned), with an optional supporting figure only if it adds new insight.

### Planned output naming (proposed)
- `paper_table_model_selection_recommendation.csv`
- Optional figure: `fig50_model_selection_frontier_or_screening.png`

### Minimum columns for recommendation table
- experiment_scope
- optimisation_metric
- constraints
- eligible_models
- recommended_model
- recommendation_reason

## 8. New CSV Tables Requested (Explicit)

## Table 1: Experiment parameters
### Filename (proposed)
- `paper_table_experiment_parameters.csv`

### Columns
- experiment_part
- dataset
- target_variable
- predictor_groups
- lag_construction
- forecast_horizon
- cv_configuration
- aggregation_levels
- model_set
- hyperparameter_set
- evaluation_metrics

## Table 2: PyNNLF output artifacts
### Filename (proposed)
- `paper_table_pynnlf_output_artifacts.csv`

### Columns
- output_artifact
- file_pattern_or_name
- content_summary
- purpose
- used_in_manuscript_section

## Table 3: Model-selection recommendation
### Filename (proposed)
- `paper_table_model_selection_recommendation.csv`

### Columns
- optimisation_metric
- constraints
- eligible_models
- recommended_model
- notes

## 9. Rerun and Regeneration Workflow (Planned Order)
1. Regenerate/verify processed recap CSVs for each experiment section.
2. Run updated process notebooks (if schema extensions are needed).
3. Run updated figure notebooks section-by-section.
4. Verify all planned PNG/CSV outputs exist at expected paths.
5. Run a consistency pass:
   - font check (Arial)
   - palette check
   - legend overlap check
   - axis/caption consistency check
6. Build/update a simple artifact index CSV for traceability (optional but recommended).

## 10. Acceptance Criteria
A change set is accepted only if all are true:
1. All requested new figure types are generated in PNG and saved in experiment-specific folders.
2. All three requested new tables are generated in CSV.
3. Aggregation-level visuals clearly show mean, stddev, and fold-level context (where available).
4. ASHD vs AEDP and horizon comparisons include actual-vs-forecast visual evidence.
5. Load composition section includes time-series visuals demonstrating composition effects.
6. Recommendation output is demonstrated via table (and optional figure).
7. All figures follow the exact style constraints (Arial + fixed palette + no overlap).

## 11. Risks and Mitigations
- Risk: Fold-level outputs may not be uniformly available in recap files.
  - Mitigation: pull from per-fold CV exports in experiment folders and document fallback behavior.
- Risk: Day selection for "AEDP is harder" could be challenged.
  - Mitigation: define deterministic day-selection criteria (e.g., top volatility quantile with matched season).
- Risk: Style inconsistency across old notebooks.
  - Mitigation: central style helper + pre-export checklist.

## 12. Confirmed Decisions (Locked)
The following choices have been confirmed and are now fixed for implementation:
1. Aggregation-level actual-vs-forecast and scatter outputs will be separate figures per aggregation level.
2. ASHD vs AEDP day selection will use a deterministic volatility-score rule, with final picks from top candidates.
3. Forecast-horizon actual-vs-forecast comparisons will use XGBoost only.
4. Model-selection recommendation output will implement all three methods: Pareto, utility, and satisficing.
5. Legacy draft-style figures will be fully replaced by the new publication figure set.

## 13. Implementation Readiness
Status: Ready for implementation.

When you approve, the next step will be to apply this plan notebook-by-notebook and produce the new CSV/PNG artifacts while preserving folder structure.
