# PyNNLF Project Structure

PyNNLF is a Python package with YAML-based configuration. The notebooks are no longer required to run experiments.

## Top-level layout

```text
PyNNLF/
|-- src/
|   `-- pynnlf/
|       |-- api.py
|       |-- engine.py
|       |-- runner.py
|       |-- tests_runner.py
|       |-- workspace.py
|       `-- scaffold/
|           |-- data/
|           |-- models/
|           `-- specs/
|-- docs/
|-- example_project/
|   |-- data/
|   |-- models/
|   |-- specs/
|   `-- experiment_result/
|-- data/
|-- requirements.txt
`-- pyproject.toml
```

## Key folders

### `src/pynnlf/`
Core package source code, including the experiment engine, runner, test runner, and workspace initialization.

### `src/pynnlf/scaffold/`
Template workspace copied when you run initialization. Includes sample data, model templates, and YAML specs.

### `example_project/`
Example workspace used throughout the documentation. This is where you edit YAML specs and run experiments in the examples shown on the docs site.

### `docs/`
Documentation source files for the GitHub Pages site.

### `data/`
Repository-level datasets and metadata. Workspace datasets live in each workspace under `data/`.

## Workspace structure (created by `init`)

```text
example_project/
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

1. Define experiments in `example_project/specs/experiment.yaml`.
2. Add or edit models in `example_project/models/`.
3. Add datasets in `example_project/data/`.
4. Run experiments to populate `example_project/experiment_result/`.
