# Installation Instructions

1. Clone the repository to your local machine.  
2. Create a Python virtual environment.  
3. Install the required packages:

```bash
pip install -r requirements.txt
```

⚠️ This may take ~10 minutes.
Although newer Python versions may work, the tool was tested on Python 3.12.3.


# How to Use The Tool
1. Open the notebook: `notebooks/model/run_experiments.ipynb.`
2. Fill in the input values for your forecast problem and model specification. Example:
```
dataset = "ds0"            # Dataset for testing
forecast_horizon = "fh1"   # fh1 = 30 minutes ahead
model_name = "m6"          # Linear regression
hyperparameter_no = "hp1"  # Hyperparameter ID
```
3. Run the notebook.

4. The tool outputs evaluation results to the `experiment_result/folder`.

5. To evaluate multiple forecast problems and model specifications at once, use `notebooks/model/run_experiments_batch.ipynb`

For the list of available datasets & models, how to modify model hyperparameter, how to add a model, how to add a dataset, and exhaustive list of API Reference see the Detailed Guide page.