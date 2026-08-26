# PyNNLF Workspace Guide

This workspace contains the PyNNLF forecasting project and the publication artifacts for Journal Article 1.

## Repository Map

- `src/`: PyNNLF package source code.
- `data/`: project datasets used by examples and experiments.
- `models/`: model definitions and model-related project files.
- `specs/`: experiment configuration files.
- `experiment_result/`: experiment outputs and recap data.
- `publication/journal_article_1/`: self-contained publication workspace.
- `docs/`: project and publication documentation.
- `example_project/`: example project workspace and this guide.

## Journal Article 1

The publication workspace is organized as follows:

- `publication/journal_article_1/data/`: publication dataset copies, including the ASHD, AEDP, and SA-BESS inputs used by the article.
- `publication/journal_article_1/results/`: processed recap CSVs, paper tables, canonical PNG figures, artifact mapping, and output validation reports.
- `publication/journal_article_1/specs/`: experiment specifications used by the retained experiment workflows.
- `publication/journal_article_1/scripts/`: shared plotting style, artifact-building helpers, and retained data-processing utilities.
- `publication/journal_article_1/notebooks/`: runnable analysis and publication workflows, organized by article section.

The active paper-facing build notebooks are:

| Notebook | Purpose | Main outputs |
|---|---|---|
| `notebooks/00_data_exploration_and_processing/1_build_paper_tables_and_manifest.ipynb` | Rebuild paper tables and the canonical artifact mapping from existing outputs. | Methods/results CSV tables and artifact manifest |
| `notebooks/01_ashd_aedp_148hh_comparison/2_build_final_figures.ipynb` | Rebuild the ASHD versus AEDP comparison. | Figures 30 and 31 |
| `notebooks/02_ashd_148hh_forecast_horizon/3_build_final_figures.ipynb` | Rebuild the ASHD forecast-horizon comparison. | Figures 40 and 41 |
| `notebooks/03_aedp_aggregation_level/5_build_final_figures.ipynb` | Rebuild the AEDP aggregation-level comparison. | Figures 10-14 |
| `notebooks/04_sa_bess_clean_44hh/8_build_final_figures.ipynb` | Rebuild the clean 44-household SA-BESS composition figures. | Figures 20-22 |

These five notebooks are the canonical paper-artifact entrypoints. They rebuild figures and tables from existing recap files and datasets; they do not rerun model training.

## Supporting Notebooks

The numbered notebooks before each build notebook document data lineage and, where applicable, experiment execution:

- Section 00 contains ASHD preparation, source exploration, and the workspace-structure figure workflow.
- Section 01 contains the ASHD/AEDP recap-processing workflow.
- Section 02 contains the ASHD experiment-run and recap-processing workflow.
- Section 03 contains AEDP checkpoint creation, aggregation-dataset generation, experiment execution, and recap processing.
- Section 04 contains SA-BESS signal diagnostics, processing, household cleaning, experiment execution, and recap processing.

The `notebooks/_archive/` directory contains historical investigations and superseded figure notebooks. It is not part of the active CI workflow.

## Manual Publication Test

From the repository root, activate the project environment:

```powershell
& .\.venv\Scripts\Activate.ps1
```

Run one canonical notebook in VS Code by opening it, selecting the `.venv` Python kernel, and choosing **Run All**. The final validation cell prints the expected artifact paths, generated outputs, and missing-file count.

To execute all five canonical notebooks from PowerShell:

```powershell
$notebooks = @(
  "publication/journal_article_1/notebooks/00_data_exploration_and_processing/1_build_paper_tables_and_manifest.ipynb",
  "publication/journal_article_1/notebooks/01_ashd_aedp_148hh_comparison/2_build_final_figures.ipynb",
  "publication/journal_article_1/notebooks/02_ashd_148hh_forecast_horizon/3_build_final_figures.ipynb",
  "publication/journal_article_1/notebooks/03_aedp_aggregation_level/5_build_final_figures.ipynb",
  "publication/journal_article_1/notebooks/04_sa_bess_clean_44hh/8_build_final_figures.ipynb"
)

foreach ($notebook in $notebooks) {
  python -m jupyter nbconvert --to notebook --execute --inplace $notebook
  if ($LASTEXITCODE -ne 0) { throw "Notebook failed: $notebook" }
}
```

Verify the canonical artifact set:

```powershell
python -c "import sys; sys.path.insert(0, 'publication/journal_article_1/scripts'); import notebook_artifact_build as build; build.assert_artifacts(); print('All canonical artifacts verified')"
```

A successful verification reports 21 expected artifacts with zero missing. The generated report is `publication/journal_article_1/results/paper_artifact_output_check.csv`.

## Important Distinction

The SA-BESS paper figures use the clean 44-household cohort. The source workflow initially selected 100 households from the raw eligible population, then filtered to 44 households with no meaningful negative reconstructed underlying-load intervals.
