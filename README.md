# PyNNLF — Python for Network Net Load Forecasting

PyNNLF (Python for Network Net Load Forecasting) evaluates net load forecasting model performance in a reliable and reproducible way.

Net load is the demand remaining after behind-the-meter generation, mostly rooftop solar, has been subtracted. Research on net load forecasting is difficult to compare across studies, because each one uses its own data and its own evaluation protocol. PyNNLF supplies both: a library of existing public net load datasets and a library of existing forecasting models, evaluated on one consistent protocol. You can add your own datasets and models, so a new model can be compared against established ones on public data. It is built for researchers in academia and industry who evaluate and optimise net load forecasting models.

![Home Illustration](./docs/img/home_illustration.png)

Full documentation: **[mssamhan31.github.io/PyNNLF](https://mssamhan31.github.io/PyNNLF/)**

## Installation

Requires **Python 3.11 or later**; tested on 3.12. On macOS use `python3`/`pip3`.

```bash
python -m pip install pynnlf
```

## Quick Start

```python
import pynnlf

pynnlf.init("my_project")                                        # create a workspace
pynnlf.run_experiment("my_project/specs/experiment.yaml")        # run it
```

Edit `my_project/specs/experiment.yaml` between those two steps to choose the dataset, forecast horizon, model and hyperparameters. Pass `plot_enabled=False` to skip plots.

Results land in a folder named for the run, for example `my_project/experiment_result/E00001_260903_ds0_fh30_m6_lr_hp1/`:

```text
E00001_a1_experiment_result.csv         accuracy, stability, runtime, seed
E00001_a2_hyperparameter.csv            hyperparameters used
E00001_a3_cross_validation_result.csv   per-fold results
E00001_cv_train/  E00001_cv_test/       per-fold observations and forecasts
E00001_models/                          the fitted model for each fold
E00001_cv1_plots/                       plots for fold 1, if enabled
```

More detail: [Getting started](https://mssamhan31.github.io/PyNNLF/getting_started/) and [Run an experiment](https://mssamhan31.github.io/PyNNLF/run_experiment/).

## Repository Structure

```text
src/pynnlf/       the package: engine, runner, metrics, and the workspace scaffold
tests/            pytest suite
data/             the public dataset library
example_project/  a materialised copy of the workspace scaffold
docs/             documentation source
paper/            Journal of Open Source Software submission
publication/      journal article notebooks, results and figures
```

Annotated version: [Repository structure](https://mssamhan31.github.io/PyNNLF/repo_structure/).

## Data

`data/` holds the public net load dataset library. Sources include the Ausgrid Solar Home Dataset (ASHD), the Australia Energy Data Platform (AEDP), an Ausgrid zone substation, and a South Australian battery energy storage system (BESS) cohort, with weather from Solcast and the Australian Bureau of Meteorology (BOM).

A new workspace ships with `ds0` only. To fetch more:

```python
pynnlf.init("my_project", download_data=True, all_data=True)
```

The unaggregated source files behind these datasets are not distributed here. See [Datasets and models](https://mssamhan31.github.io/PyNNLF/datasets_models/) for the full list and for the environment variable the publication notebooks expect.

## Tests

```bash
python -m pip install -e ".[test]"
python -m pytest
```

See [Testing](https://mssamhan31.github.io/PyNNLF/tool_testing/) for the benchmark comparison run by `pynnlf.run_tests()`.

## Continuous Integration

Every push and pull request runs linting, the test suite, and smoke experiments on three models checked against the standard benchmark.

## Licence

MIT. See [LICENSE](./LICENSE).

## Citation

Please cite PyNNLF if you use it. Metadata is in [CITATION.cff](./CITATION.cff); the archived release is [10.5281/zenodo.22104164](https://doi.org/10.5281/zenodo.22104164).

## Acknowledgements
This project is part of Samhan's PhD study, supported by the University International Postgraduate Award (UIPA) Scholarship from UNSW, the Industry Collaboration Project Scholarship from Ausgrid, the RACE for 2030 Scholarship, and the NSW Decarbonisation Innovation Hub (NSW Decarb Hub). We also acknowledge Solcast and the Australian Bureau of Meteorology (BOM) for providing access to historical weather datasets for this research. We further acknowledge the use of Python libraries including Pandas, NumPy, PyTorch, Scikit-learn, XGBoost, Prophet, Statsmodels, and Matplotlib. Finally, we thank the reviewers and editor of the Journal of Open Source Software for their valuable feedback and guidance.

The authors declare that they have no competing financial, personal, or professional interests related to this work.

## Contributors
- **M. Syahman Samhan** (m.samhan@unsw.edu.au): Lead developer and researcher. Responsible for conceptualization, implementation, documentation, and experimentation.
- **Anna Bruce**: Supervisor. Provided guidance on research direction and methodology.
- **Baran Yildiz**: Supervisor. Provided guidance on research direction and methodology.
