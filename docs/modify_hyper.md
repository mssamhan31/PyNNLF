
# Model Hyperparameter

All hyperparameter values are stored separately from the model code in a YAML file: `models/hyperparameters.yaml`.  
**This design avoids hard-coded values and provides a centralized location for managing all hyperparameters.**

## Format

Hyperparameter values are stored as a list of dictionaries. For example, for `m7_ann`, the values are stored under the `m7` key. Each dictionary contains an `hp_no` (hyperparameter set ID) and key-value pairs for the parameters.

```yaml
m7:
    - hp_no: hp1
        seed: 99
        hidden_size: 10
        activation_function: relu
        learning_rate: 0.001
        solver: adam
        epochs: 500
    - hp_no: hp2
        seed: 99
        hidden_size: 10
        activation_function: relu
        learning_rate: 0.01
        solver: adam
        epochs: 500
```

# How to Modify Model Hyperparameter
If, for example, you want to modify the learning rate of the ANN model, create a new hyperparameter set with a new `hp_no`:

```yaml
m7:
    - hp_no: hp1
        seed: 99
        hidden_size: 10
        activation_function: relu
        learning_rate: 0.001
        solver: adam
        epochs: 500
    - hp_no: hp2
        seed: 99
        hidden_size: 10
        activation_function: relu
        learning_rate: 0.01
        solver: adam
        epochs: 500
    - hp_no: hp3
        seed: 99
        hidden_size: 10
        activation_function: relu
        learning_rate: 0.1  # modified learning rate
        solver: adam
        epochs: 500
```

When running experiments, select the desired hyperparameter set by referencing its `hp_no`, such as `hp3`, in `specs/experiment.yaml` under `hyperparameter`.