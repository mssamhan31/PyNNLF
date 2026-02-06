## Model Structure

All models are stored in the workspace `models` folder as `.py` files. Each model defines two functions: one for training and one for forecasting. Cross-validation, evaluation, and plotting are handled by the core engine, so model files only need to implement model-specific logic.

## Training Function

The training function trains the model using the training set predictors and target values. It returns a single object, `model`, which contains the trained model.

## Forecast Function
The forecast function uses the trained model along with the training and testing predictors. It returns two outputs: forecasts for the training and testing sets.

## How to Add a Model

To add a new model, follow these steps:

### 1. Create a new model file in your workspace

Create a new `.py` file under your workspace `models` folder. The file name must start with a model ID prefix, for example `m19_my_model.py`. The functions inside must match the file stem:

- `train_model_m19_my_model(...)`
- `produce_forecast_m19_my_model(...)`

You can start from the template file `models/mXX_template.py` in your workspace.

### 2. Add hyperparameters for the new model

Add a new section in `models/hyperparameters.yaml` under the model ID (e.g., `m19`), and include one or more `hp_no` entries.

### 3. Select the model in your experiment spec

Set the model in `specs/experiment.yaml`:

```
model: m19
hyperparameter: hp1
```

The model will be discovered automatically based on the `m19_*.py` file in your workspace `models` folder. No dispatch code changes are required.
