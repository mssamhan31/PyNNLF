# PyNNLF Project Structure

PyNNLF (Python for Network Net Load Forecasting) is a Python package with YAML-based configuration. The notebooks are no longer required to run experiments.

## Top-level layout

```text
PyNNLF/
|-- src/
|   `-- pynnlf/
|       |-- __about__.py            single source of the package version
|       |-- api.py                  public entry points re-exported by __init__
|       |-- discovery.py            resolves dataset and model identifiers to files
|       |-- engine.py               features, cross-validation, metrics, plots
|       |-- hyperparams.py          loads and indexes hyperparameters.yaml
|       |-- model_utils.py          shared model helpers, plus the legacy JSON workflow
|       |-- recap_experiments.py    aggregates many experiment results into one CSV
|       |-- reproducibility.py      seeds Python, NumPy and PyTorch
|       |-- runner.py               resolves a YAML spec and runs an experiment
|       |-- tests_runner.py         benchmark comparison harness
|       |-- workspace.py            pynnlf.init() creates a user workspace
|       |-- yamlio.py               YAML loading with existence and type checks
|       |-- tools/                  developer utilities, not part of the public API
|       `-- scaffold/
|           |-- data/               the ds0 sample dataset
|           |-- models/             the 18 bundled models and a template
|           `-- specs/              experiment, batch and test specifications
|-- tests/                          pytest suite
|-- docs/                           documentation source for the GitHub Pages site
|-- example_project/                a materialised copy of the workspace scaffold
|-- data/                           the public dataset library
|-- paper/                          Journal of Open Source Software submission
|-- publication/                    journal article notebooks, results and figures
|-- .github/workflows/              continuous integration and docs deployment
|-- CITATION.cff                    how to cite this work
|-- LICENSE                         MIT licence
|-- requirements.txt                pinned environment for reproducing a run
`-- pyproject.toml                  package metadata and dependencies
```

## Key folders

### `src/pynnlf/`
Core package source code: the experiment engine, the runner, the benchmark harness, workspace initialisation, and the supporting loaders. Every module opens with a docstring stating its purpose, inputs, outputs and key steps.

### `src/pynnlf/scaffold/`
Template workspace copied when you run initialisation. Includes the sample dataset, the model library, and the YAML specifications.

### `tests/`
The pytest suite, covering the package's core guarantees: that it installs and creates a workspace, that an experiment runs end to end and writes the expected columns, and that the error metrics return known values. Run them with `python -m pytest`.

### `example_project/`
Example workspace used throughout the documentation. Note that `pynnlf.init()` writes into whatever directory you name, so use a fresh name such as `my_project` rather than overwriting this tracked copy.

### `docs/`
Documentation source files for the GitHub Pages site.

### `data/`
The repository-level public dataset library and its metadata. Workspace datasets live in each workspace under `data/`.

### `paper/` and `publication/`
`paper/` holds the Journal of Open Source Software submission. `publication/journal_article_1/` holds the journal article artefacts: the analysis notebooks, the processed results, and the figures. The data preparation notebooks there read source inputs from a local directory named by the `PYNNLF_RAW_DATA_DIR` environment variable; see [Datasets and models](datasets_models.md) for the layout it expects.

## Workspace structure (created by `init`)

```text
my_project/
|-- data/
|-- models/
|-- specs/
|   |-- experiment.yaml
|   |-- batch.yaml
|   |-- tests_ci.yaml
|   `-- tests_full.yaml
`-- experiment_result/
    `-- Archive/Testing Result/
```

## Where to edit

1. Define experiments in `my_project/specs/experiment.yaml`.
2. Add or edit models in `my_project/models/`.
3. Add datasets in `my_project/data/`.
4. Run experiments to populate `my_project/experiment_result/`.
