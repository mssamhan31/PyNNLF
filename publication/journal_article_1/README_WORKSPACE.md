# Journal article 1 workspace

This is a PyNNLF workspace holding everything behind the journal article: the
datasets the experiments read, the cross-validated experiment output, the
notebooks and scripts that turn that output into figures and tables, and the
generated artefacts themselves.

Nothing here re-runs an experiment as part of building a figure. The figure
build reads existing results only, so a rebuild is reproducible and cheap.

## Layout

| Directory | Contents |
|---|---|
| `data/` | The processed datasets, one CSV per dataset id, with a `netload_kW` column and a datetime index. See `data/README_data.md`. |
| `specs/` | Experiment and batch specifications, plus the workspace config. |
| `models/` | Model definitions and the hyperparameter file used by the runs. |
| `experiment_result/` | Cross-validated output of the runs executed locally, one folder per experiment, plus the `a1_experiment_result.csv` index. |
| `experiment_result_databricks/` | The same, for the load-composition runs executed on Databricks. |
| `notebooks/` | The workflow, numbered per section. `_archive/` holds superseded versions. |
| `results/` | Everything generated: figures under `<section>/figures/`, tables as `paper_table_*.csv`. |
| `scripts/` | The plotting and table code the notebooks call. |

## How a figure gets built

```
notebooks/0N_*/N_build_final_figures.ipynb
  -> scripts/notebook_artifact_build.py        PAPER_ARTIFACTS manifest, build_section_0N()
       -> scripts/generate_paper_figures.py    most figures and tables
       -> scripts/generate_sa_bess_combined_figures.py
            -> scripts/publication_plot_style.py   palette, rcParams, save_figure
```

`PAPER_ARTIFACTS` in `scripts/notebook_artifact_build.py` is the single source of
truth for what the article ships. Continuous integration runs
`assert_artifacts()` against it, so a figure that is not in that list is not
checked, and a figure in the list that fails to build breaks the build. Add new
artefacts there and to the matching `build_section_0N()`.

Every figure is written by `save_figure()` as a **PNG at 300 dpi** with
`bbox_inches="tight"`. Colours come from `PALETTE` in
`scripts/publication_plot_style.py`, which is the Okabe-Ito qualitative set.
Where traces share an axis, pair a colour with a dash pattern so the figure
survives greyscale printing and colour-blind vision.

## Figure numbering

Filenames use a stable internal `figNN_` sequence that does **not** follow the
manuscript's figure numbers, because the manuscript renumbers between drafts.
`results/paper_artifact_reference_mapping.csv` records the proposed manuscript
label for each artefact. The correspondence to the current manuscript is:

| Manuscript | Artefact |
|---|---|
| Aggregation, RMSE per household | `results/03_aedp_aggregation_level/figures/fig11_aedp_agg_rmse_per_hh_mean_std_cvbg.png` |
| Aggregation, nRMSE | `results/03_aedp_aggregation_level/figures/fig10_aedp_agg_nrmse_mean_std_cvbg.png` |
| Aggregation, both metrics in one figure | `results/03_aedp_aggregation_level/figures/fig15_aedp_agg_nrmse_and_rmse_per_hh_two_panel.png` |
| Load composition, model comparison | `results/04_sa_bess_clean_44hh/figures/fig23_sa_bess_composition_test_rmse_kw.png` |
| Load composition, time series and error | `results/04_sa_bess_clean_44hh/figures/fig20_sa_bess_composition_timeseries_error_combined.png` |
| Load composition, results table | `results/04_sa_bess_clean_44hh/paper_table_sa_bess_44hh_signal_test_rmse_kw.csv` and `.md` |
| Accuracy against stability, ASHD at 1 week | `results/02_ashd_148hh_forecast_horizon/figures/fig42_ashd_nrmse_stability_scatter_1week.png` |
| Accuracy against stability, ASHD and AEDP at 1 day | `results/01_ashd_aedp_148hh_comparison/figures/fig32_ashd_aedp_nrmse_stability_scatter_1day.png` |

Figures under `results/*/figures/supplementary/` are **not** manuscript figures.
They record alternatives that were considered and not adopted, and their
filenames say so.

## Metrics

`test_RMSE` is in kilowatts. `test_nRMSE` is that RMSE as a percentage of the
maximum net load over the experiment's own cross-validation window:

```python
max_y = df['y'].max()                  # engine.py, once before the folds
test_nRMSE = 100 * test_RMSE / max_y
```

Two consequences worth knowing before comparing percentages:

- The denominator is **per dataset**. Two datasets with different peaks are not
  on the same scale, so an nRMSE difference between them mixes a difference in
  error with a difference in peak. Where a section compares datasets of
  different magnitude, it reports kilowatts.
- The denominator is never written to any output file. It can be recovered as
  `100 * test_RMSE / test_nRMSE`; `generate_paper_figures._aggregation_denominators()`
  does exactly that.

`*_stddev` columns are the standard deviation across the 10 cross-validation
folds of a single experiment, not across repeated samples. Where a result
averages several samples of the same design, the figure shows the spread across
samples and keeps the fold spread as a faint background.

## Rebuilding everything

From `scripts/`:

```powershell
python -c "import notebook_artifact_build as nab; nab.build_tables_and_mapping(); nab.build_section_01(); nab.build_section_02(); nab.build_section_03(); nab.build_section_04(); nab.assert_artifacts()"
```

Or run the four `*_build_final_figures.ipynb` notebooks in order, which is the
documented path and the one continuous integration exercises.
