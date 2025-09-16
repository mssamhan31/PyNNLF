
# Model Hyperparameter

All hyperparameter values are stored separately from the model code in a designated notebook: `notebooks/config/model_hyperparameters.ipynb`.  
**This design avoids hard-coded values and provides a centralized location for managing all hyperparameters.**

## Format

Hyperparameter values are stored as a list of dictionaries. For example, for `m7_ann`, the values are stored in the `m7_hp_table` list.  
Each dictionary contains an `hp_no` (hyperparameter set ID) and key-value pairs for the parameters.

```python
m7_hp_table = [
    {
        "hp_no": "hp1",
        "seed": 99, # we will use the same seed for reproducibility
        "hidden_size": 10, #number of neurons in hidden layers
        "activation_function": "relu",
        "learning_rate" : 0.001,
        "solver" : "adam",
        "epochs" : 500
    },
    {
        "hp_no": "hp2",
        "seed": 99, # we will use the same seed for reproducibility
        "hidden_size": 10, #number of neurons in hidden layers
        "activation_function": "relu",
        "learning_rate" : 0.01,
        "solver" : "adam",
        "epochs" : 500
    },
]
```

# How to Modify Model Hyperparameter
If for example we want to modify the learning rate of the ANN model we can create a new hyperparameter set, and below is the update result:
```
m7_hp_table = [
    {
        "hp_no": "hp1",
        "seed": 99, # we will use the same seed for reproducibility
        "hidden_size": 10, #number of neurons in hidden layers
        "activation_function": "relu",
        "learning_rate" : 0.001,
        "solver" : "adam",
        "epochs" : 500
    },
    {
        "hp_no": "hp2",
        "seed": 99, # we will use the same seed for reproducibility
        "hidden_size": 10, #number of neurons in hidden layers
        "activation_function": "relu",
        "learning_rate" : 0.01,
        "solver" : "adam",
        "epochs" : 500
    },
    {
        "hp_no": "hp3",
        "seed": 99, # we will use the same seed for reproducibility
        "hidden_size": 10, #number of neurons in hidden layers
        "activation_function": "relu",
        "learning_rate" : 0.1, **Modified learning rate**
        "solver" : "adam",
        "epochs" : 500
    },
]
```

When running experiments, you can select the desired hyperparameter set by referencing its `hp_no`, such as `hp3`.