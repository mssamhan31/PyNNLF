# Simple Testing

To quickly test the tool, follow the steps in the **Getting Started** section before. You can try it out with the following example inputs:

```
dataset = ds0              # Sample dataset for testing
forecast_horizon = fh1     # fh1 = 30 minutes ahead
model_name = m6            # m6 = Linear Regression
hyperparameter_no = 'hp1'  # Example hyperparameter set
```
This is a great way to get familiar with how the tool works before running full-scale tests.


# Full Model Testing (All 18 Models)

For a complete evaluation and checking the result against benchmark values, which is available on `experiment_result/Archive/Testing Result/testing_benchmark`. 
To test it, follow these steps:

1. Open the file `run_tests.ipynb`.
2. Run **all cells** without making any changes.
3. The notebook will automatically: Run all 18 models, compare their outputs against three benchmark results. The results are available in:  
     `experiment_result/Archive/Testing Result/`

> ⏱ **Estimated time**: ~1 hour on a personal computer with an Intel i5 processor and 32GB RAM. This tool has undergone full model testing on three different computers, and the result can be seen on 
```
experiment_result/Archive/Testing Result/20250821_test_result_CEEM Computer.csv
experiment_result/Archive/Testing Result/20250821_test_result_UNSW Laptop.csv
experiment_result/Archive/Testing Result/20250822_test_result_SS Personal Laptop
```
