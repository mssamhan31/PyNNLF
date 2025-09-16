## Dataset
The available datasets are listed in the `data` folder, with metadata provided in `data/metadata.xlsx`. Below are some of the datasets currently included in the library:

### Ausgrid Solar Home Datasets 
This dataset has been widely used in net load forecasting research since 2016. It includes data from 300 solar-equipped households within the Ausgrid network in Sydney. The original source is [here](https://data.nsw.gov.au/data/dataset/solar-home-electricty-data/resource/d2dc76f0-22e3-4efc-bed9).


| Dataset Name                          | Description                                                                 |
|--------------------------------------|-----------------------------------------------------------------------------|
| `ds1_ahsd.csv`                       | Ausgrid Solar Home Dataset, aggregate of 300 household data in Ausgrid network |
| `ds4_ashd_with_weather.csv`          | `ds1` enhanced with temperature, relative humidity, and wind speed data     |
| `ds13_ashd_with_cloud_solcast.csv`  | `ds4` further enriched with cloud data from Solcast                         |

### Australia Energy Data Platform (AEDP) Datasets
These datasets were compiled by UNSW Sydney using data from Solar Analytics and Wattwatchers. Sensitive information such as customer addresses, names, and NMIs has been removed. The original source is
 [here](https://darth.engineering.unsw.edu.au/).

| Dataset Name                                | Description                                                              |
|--------------------------------------------|--------------------------------------------------------------------------|
| `ds10_aedp_cluster2_30min.csv`             | AEDP dataset for Cluster 2 with 30-minute resolution                     |
| `ds11_aedp_cluster2_30min_with_weather.csv`| `ds10` enhanced with weather data including temperature, humidity, etc. |

### Ausgrid Zone Substation (ZS) Datasets – Mascot
Unlike the previous household-focused datasets, this dataset covers a zone substation, which includes residential, commercial & industrial (C&I), and major customers. The original source is [here](https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Distribution-zone-substation).

| Dataset Name                                 | Description                                                              |
|---------------------------------------------|--------------------------------------------------------------------------|
| `ds14_ausgrid_zs_mascot.csv`                | Zone Substation data for Mascot                                          |
| `ds15_ausgrid_zs_mascot_30min_with_weather.csv` | `ds14` enhanced with weather data at 30-minute resolution                |


## Model

### Forecasting Models Overview

| Model ID   | Model Name       | Short Description                                                                 |
|------------|------------------|------------------------------------------------------------------------------------|
| `m1_naive` | Naive            | Forecast equals the last observed value                                           |
| `m2_snaive`| Seasonal Naive   | Forecast equals the value from the same season in the previous cycle              |

#### Statistical Models

| Model ID   | Model Name       | Short Description                                                                 |
|------------|------------------|------------------------------------------------------------------------------------|
| `m3_ets`   | ETS              | Exponential smoothing model with error, trend, and seasonality components         |
| `m4_arima` | ARIMA            | Autoregressive Integrated Moving Average model for time series forecasting        |
| `m5_sarima`| SARIMA           | Seasonal ARIMA model with seasonal components                                     |
| `m6_lr`    | Linear Regression| Predicts future values using a linear combination of input features               |

#### Machine Learning Models

| Model ID   | Model Name       | Short Description                                                                 |
|------------|------------------|------------------------------------------------------------------------------------|
| `m7_ann`   | ANN              | Basic Artificial Neural Network with one hidden layers                    |
| `m8_dnn`   | Deep Neural Network  | ANN with more than one hidden layer                         |
| `m9_rt`    | Regression Tree  | Decision tree model for regression tasks                                          |
| `m10_rf`   | Random Forest    | Ensemble of regression trees for improved accuracy and robustness                 |
| `m11_svr`  | Support Vector Regression | Uses support vectors to perform regression with margin of tolerance       |
| `m12_rnn`  | Recurrent Neural Network | Neural network with feedback loops for sequential data                     |
| `m13_lstm` | Long Short-Term Memory | RNN variant designed to capture long-term dependencies                     |
| `m14_gru`  | Gated Recurrent Unit | Simplified LSTM with fewer parameters                                     |
| `m15_transformer` | Transformer     | Attention-based model for sequence modeling without recurrence                  |
| `m16_prophet` | Prophet         | Time series model developed by Facebook for business forecasting                 |
| `m17_xgb`  | XGBoost          | Gradient boosting framework optimized for speed and performance                  |
| `m18_nbeats`| N-BEATS         | Deep learning model for univariate time series forecasting                       |

### Hyperparameter
The list of available model and its hyperparameter can be seen on `config/model_hyperparameters.ipynb`. The values currently available are the hyperparameter values mostly used in academic literature, but not necessarily the optimum value. 