# Tool Name

# Introduction
Net load is the difference between electricity load and solar generation. Forecasting net load is important to network operators both for operational and planning purpose. Since 2016, there has been more than 60 papers published on the topic. Most of them introduced new net load forecasting model and claimed superior performance. However, most of these results are obtained by a private evaluation. Many researchers used a publicly unavailable dataset, compared with another complex model as benchmark model, and did not share their implementaiton code publicly

# Objective
This tool aims to evaluate net load forecasting models in a reliable and reproducible manner. It has library of public net load datasets and common net load forecasting models including simple benchmark models. The user inputs the forecast problem and model specification, and the tool outputs the evaluation results. The tool enables the user to add their own dataset, model, and to modify the hyperparameters. Thus, a researcher claiming new and superior model can use this tool to compare their model with other existing models on a public dataset.

# Tool Input
1. Forecast problem: dataset & forecast horizon. List of possible forecast problem values is on notebooks/config/config.ipynb. 
2. Forecast model specification: model & hyperparameter. List of possible model specification is on notebooks/config/model_hyperparameters.ipynb.

# Tool output
1. a1_experiment result.csv : accuracy (cross-validated test n-RMSE Error), stability (accuracy stddev), training time,
2. a2_hyperparameter.csv : hyperparameter being used on the model
3. a3_cross_validation_result.csv : detailed result for each of the cross validation split
4. cv_plots folder: some plots of train and test set, consisting of
   1. observation vs forecast time plot
   2. observation vs forecast scatter plot
   3. residual time plot
   4. residual histogram
5. cv_test and cv_train folder: consisting of time series of the observation, forecast, and residual of each of the cross validation

# Tool output naming convention
[experiment no]_[experiment date]_[dataset]_[forecast horizon]_[model]_[hyperparameter]

# Quick Start
1. Clone the whole repository to your personal computer.
2. Open notebooks/model/run_experiments.ipynb.
3. Fill the input values (forecast problem & model specification). 
4. Run notebooks/model/run_experiments.ipynb.
5. Tool will output the evaluation result on experiment_result/ as one folder. 
6. If you want to evaluate several forecast problems and model specifications at once, you can use notebooks/model/run_experiments_batch.ipynb

# Tool Features
1. Adjustable. User can specify the dataset, forecast horizon, model, and model hyperparameter
2. Systematic. Every experiment performed also outputs metadata like experiment date, experiment number, forecast problem, model specification, etc.
3. Flexible. User can add or modify various aspects: model, dataset, hyperparameter, general functions, etc.

# Tool Limitation
1. Basic user interface. User needs to modify the code manually to change the model
2. Limited dataset and model. There are still many public net load datasets & models that can be added to the library.

# Folder Structure (WIP HERE)
- data: contains all dataset being used on this project
- experiment_result : storing experiment results and plots from running the model
- notebooks : all the codes being used

# Current Dataset and Model Library
The list of dataset available can be seen on the folder data, and the metadata can be seen on data/metadata.xlsx
The list of available model and its hyperparameter can be seen on config/model_hyperparameters.ipynb

# Configuration and Customization
## How to edit hyperparameter
Edit it on notebooks/0. config/config.ipynb

## How to add a dataset

## How to add a model

# License
MIT

# Acknowledgements
This project is part of Samhan's PhD study, supported by the University International Postgraduate Award (UIPA) Scholarship from UNSW, the Industry Collaboration Project Scholarship from Ausgrid, and the RACE for 2030 Industry PhD Scholarship. We also acknowledge Solcast and the Australian Bureau of Meteorology (BOM) for providing access to historical weather datasets for this research. Finally, we thank the reviewers and editor of the Journal of Open Source Software for their valuable feedback and guidance.
