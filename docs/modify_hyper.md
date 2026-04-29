# Model Hyperparameter

All hyperparameter values are stored separately from the model code in `models/hyperparameters.yaml`.

This design avoids hard-coded values and provides a central place to manage model settings.

The workspace config also includes `reproducibility.seed` in `specs/pynnlf_config.yaml`. When this value is set, PyNNLF seeds common random number generators and overrides seed-like hyperparameter keys such as `seed`, `xgb_seed`, `random_seed`, and `random_state` at runtime. The experiment summary records the central seed in `a1_experiment_result.csv`, and `a2_hyperparameter.csv` records the effective hyperparameters used for the run.

## Format

PyNNLF expects a YAML mapping:

- The top-level key must match the full model file stem, such as `m7_ann`.
- Each hyperparameter set is stored under an `hp_no` key such as `hp1`, `hp2`, or `hp3`.

Example:

```yaml
m7_ann:
  hp1:
    seed: 99
    hidden_size: 10
    activation_function: relu
    learning_rate: 0.001
    solver: adam
    epochs: 500
  hp2:
    seed: 99
    hidden_size: 10
    activation_function: relu
    learning_rate: 0.01
    solver: adam
    epochs: 500
```

## How to Modify Model Hyperparameter

If you want to modify the learning rate of the ANN model, add a new hyperparameter set with a new `hp_no`:

```yaml
m7_ann:
  hp1:
    seed: 99
    hidden_size: 10
    activation_function: relu
    learning_rate: 0.001
    solver: adam
    epochs: 500
  hp2:
    seed: 99
    hidden_size: 10
    activation_function: relu
    learning_rate: 0.01
    solver: adam
    epochs: 500
  hp3:
    seed: 99
    hidden_size: 10
    activation_function: relu
    learning_rate: 0.1
    solver: adam
    epochs: 500
```

Then select the new hyperparameter set in `example_project/specs/experiment.yaml`:

```yaml
model: m7
hyperparameter: hp3
```

The experiment spec uses the short model ID such as `m7`, while `models/hyperparameters.yaml` uses the full model file stem such as `m7_ann`.
