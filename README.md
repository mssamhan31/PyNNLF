# Welcome to PyNNLF
PyNNLF (Python for Network Net Load Forecast) is a tool to evaluate net load forecasting model performance in a reliable and reproducible way.

You can access the [GitHub repository here](https://github.com/mssamhan31/PyNNLF).

# Objective
This tool evaluates net load forecasting models reliably and reproducibly. It includes a library of public net load datasets and common forecasting models, including simple benchmark models. Users input the forecast problem and model specification, and the tool outputs evaluation results. 

It also allows users to add datasets, models, and modify hyperparameters. Researchers claiming a new or superior model can compare their model with existing ones on public datasets. The target audience includes researchers in academia or industry focused on evaluating and optimizing net load forecasting models. 

A visual illustration of the tool workflow is shown below.
![Home Illustration](./docs/img/home_illustration.png)

# Input
1. **Forecast Target**: dataset & forecast horizon. List of possible forecast problem values is in `notebooks/config/config.ipynb`.
2. **Model Specification**: model & hyperparameters. List of possible model specifications is in `notebooks/config/model_hyperparameters.ipynb`.

# Output
1. `a1_experiment_result.csv` – contains accuracy (cross-validated test n-RMSE), stability (accuracy stddev), and training time.
2. `a2_hyperparameter.csv` – lists hyperparameters used for each model.
3. `a3_cross_validation_result.csv` – detailed results for each cross-validation split.
4. `cv_plots/` – folder with plots including:
   - Observation vs forecast (time plot)
   - Observation vs forecast (scatter plot)
   - Residual time plot
   - Residual histogram
5. `cv_test/` and `cv_train/` – folders containing time series of observation, forecast, and residuals for each cross-validation split.

# Tool Output Naming Convention
Format:
`[experiment_no]_[experiment_date]_[dataset]_[forecast_horizon]_[model]_[hyperparameter]`

Example:
`E00001_250915_ds0_fh30_m6_lr_hp1`

# Installation Instruction
1. Clone the whole repository to your personal computer.
2. Create a python virtual environment and install the requirements using pip install -r requirements.txt. This will take ~10 minutes. Although newer Python version should work, this tool was tested using Python 3.12.3. 

# How to Use The Tool
1. Open notebooks/model/run_experiments.ipynb.
2. Fill the input values (forecast problem & model specification) that you want. 
3. Run notebooks/model/run_experiments.ipynb.
4. Tool will output the evaluation result on experiment_result/ as one folder. 
5. If you want to evaluate several forecast problems and model specifications at once, you can use notebooks/model/run_experiments_batch.ipynb

# Tool Testing & Example of How to Use The Tool
- For simple testing, follow the steps on "How to Use The Tool" above and try certain inputs. For example, use these inputs
dataset = ds0 # This is dataset for testing
forecast_horizon = fh1 # fh1 = 30 minutes ahead
model_name = m6 # this is linear regression
hyperparameter_no = 'hp1'
- For a complete test of all 18 models, use the file run_tests.ipynb and run all cells without modifying anything. It will output a file in experiment_result/Archive/Testing Result which compares all experiment outputs are within the acceptable range according to three benchmark outputs. This test takes around 1 hour using a personal computer with Intel i5 processor & 32GB RAM. 

# Result Example
- Suppose we do the simple test using the inputs above, this should take less than 1 minute.  
- <img src="image-1.png" alt="test output" style="height:300; width:auto;"/>  
- This is the output folder  
- <img src="image.png" alt="output folder" style="height:200; width:auto;"/>  
- The file a1_experiment_result.csv will summarise the result, including the cross validated nRMSE & its standard deviation  
- <img src="image-2.png" alt="result summary" style="height:30; width:auto;"/>  
- The file a3_cross_validation_result.csv will provide the cross validation (CV) detailed result, from CV1 to CV10  
- <img src="image-5.png" alt="cv results" style="height:200; width:auto;"/>  
- Below are some plots on the test set <br>
  <img src="E00001_c1_test_timeplot.png" alt="timeplot" height="300"/> <img src="E00001_c2_test_scatterplot.png" alt="scatterplot" height="300"/> <br>
  <img src="E00001_c3_test_residual_timeplot.png" alt="residual timeplot" height="300"/> <img src="E00001_c4_test_residual_histogram.png" alt="histogram" height="300"/>

# Tool Features
1. Adjustable. User can specify the dataset, forecast horizon, model, and model hyperparameter
2. Systematic. Every experiment performed also outputs metadata like experiment date, experiment number, forecast problem, model specification, etc.
3. Flexible. User can add or modify various aspects: model, dataset, hyperparameter, general functions, etc.

# Tool Limitation
1. Basic user interface. User needs to modify the code manually to change the model
2. Limited dataset and model. There are still many public net load datasets & models that can be added to the library. We welcome contribution to add these dataset & models into the library!

# Folder Structure
- data: contains all dataset being used on this project
- experiment_result : storing experiment results and plots from running the model
- notebooks : all the codes being used

# Current Dataset and Model Library
The list of dataset available can be seen on the folder data, and the metadata can be seen on data/metadata.xlsx
The list of available model and its hyperparameter can be seen on config/model_hyperparameters.ipynb

# License
MIT License

# Acknowledgements
This project is part of Samhan's PhD study, supported by the University International Postgraduate Award (UIPA) Scholarship from UNSW, the Industry Collaboration Project Scholarship from Ausgrid, and the RACE for 2030 Industry PhD Scholarship. We also acknowledge Solcast and the Australian Bureau of Meteorology (BOM) for providing access to historical weather datasets for this research. We further acknowledge the use of Python libraries including Pandas, NumPy, PyTorch, Scikit-learn, XGBoost, Prophet, Statsmodels, and Matplotlib. Finally, we thank the reviewers and editor of the Journal of Open Source Software for their valuable feedback and guidance.

# Potential Conflict Disclosure
The authors declare that they have no competing financial, personal, or professional interests related to this work.

# Raising Issue, Contributing, and Support Request
To report bugs, request features, or suggest improvements, please use the GitHub Issues feature. For contributing or seeking support, contact m.samhan@unsw.edu.au.

# Full Documentation
Detailed documentation including API documentation, how to add model and dataset, can be seen here: 