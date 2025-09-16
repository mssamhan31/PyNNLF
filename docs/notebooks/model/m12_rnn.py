#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# IMPORT IMPORTANT LIBRARY
import pandas as pd
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# In[ ]:


def train_model_m12_rnn(hyperparameter, train_df_X, train_df_y):
    ''' Train and test an RNN model for point forecasting. 
    Essentially we use the RNN block to learn the temporal patterns of the time series, 
    and then use a fully connected layer to learn the relationship between the lag features.
    We take the last hidden state of the RNN as the output, and concatenate it with the exogenous features 
    (like calendar) to make the final prediction using a fully connected layer.
    '''

    # UNPACK HYPERPARAMETER
    seed = int(hyperparameter['seed'])
    input_size = int(hyperparameter['input_size'])
    hidden_size = int(hyperparameter['hidden_size'])
    num_layers = int(hyperparameter['num_layers'])
    output_size = int(hyperparameter['output_size'])
    batch_size = int(hyperparameter['batch_size'])
    epochs = int(hyperparameter['epochs'])
    learning_rate = hyperparameter['learning_rate']

    # DEFINE MODEL AND TRAINING FUNCTION
    class RNNModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, exog_size, output_size=1):
            super(RNNModel, self).__init__()

            # Define the RNN layer
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

            # Define the Fully Connected (FC) layer
            self.fc = nn.Linear(hidden_size + exog_size, output_size)

        def forward(self, x, exogenous_data):
            # Pass the input through the RNN
            out, h_n = self.rnn(x)

            # Get the last timestep hidden state
            last_hidden_state = out[:, -1, :]  # Shape: (batch_size, hidden_size)

            # Concatenate hidden state with exogenous vars
            combined_input = torch.cat((last_hidden_state, exogenous_data), dim=1)

            # Pass through FC
            out = self.fc(combined_input)
            return out

    def train_rnn_with_minibatches(model, train_loader, epochs, learning_rate=0.001):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            print(f'Epoch [{epoch+1}/{epochs}]')
            start_time = time.time()

            model.train()
            batch_no = 1
            for X_lags_batch, X_exog_batch, y_batch in train_loader:
                print(f'Epoch [{epoch+1}/{epochs}] and batch [{batch_no}/{len(train_loader)}]')
                batch_no += 1

                # Forward pass
                predictions = model(X_lags_batch, X_exog_batch)
                loss = criterion(predictions, y_batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            end_time = time.time()
            epoch_time = end_time - start_time
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, time taken: {epoch_time:.2f} seconds')

    def set_seed(seed=seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    # PREPARE TRAIN DATA
    X_lags, X_exog = separate_lag_and_exogenous_features(train_df_X)
    X_lags_tensor = torch.tensor(X_lags.values, dtype=torch.float32)
    X_exog_tensor = torch.tensor(X_exog.values, dtype=torch.float32)
    y_tensor = torch.tensor(train_df_y.values, dtype=torch.float32).view(-1, 1)

    total_lag_features = X_lags_tensor.shape[1]
    sequence_length = total_lag_features // input_size
    exog_size = X_exog_tensor.shape[1]

    # Reshape to 3D
    X_lags_tensor = X_lags_tensor.view(-1, sequence_length, input_size)

    # INITIALIZE MODEL + DATALOADER
    set_seed(seed=seed)
    rnn = RNNModel(input_size, hidden_size, num_layers, exog_size, output_size)
    train_data = TensorDataset(X_lags_tensor, X_exog_tensor, y_tensor)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # TRAIN MODEL
    train_rnn_with_minibatches(rnn, train_loader, epochs=epochs, learning_rate=learning_rate)

    # PACK MODEL
    model = {"rnn": rnn, 'hyperparameter': hyperparameter, "train_df_X": train_df_X, "train_df_y": train_df_y}
    return model


# In[ ]:


def produce_forecast_m12_rnn(model, train_df_X, test_df_X):
    """Create forecast at the train and test set using the trained RNN model"""

    # UNPACK MODEL
    rnn = model['rnn']
    hyperparameter = model['hyperparameter']

    # UNPACK HYPERPARAMETER
    input_size = int(hyperparameter['input_size'])
    batch_size = int(hyperparameter['batch_size'])

    # PRODUCE FORECAST
    def produce_forecast(rnn, X):
        X_lags, X_exog = separate_lag_and_exogenous_features(X)
        X_lags_tensor = torch.tensor(X_lags.values, dtype=torch.float32)
        X_exog_tensor = torch.tensor(X_exog.values, dtype=torch.float32)

        total_lag_features = X_lags_tensor.shape[1]
        sequence_length = total_lag_features // input_size
        X_lags_tensor = X_lags_tensor.view(-1, sequence_length, input_size)

        predictions = []
        for i in range(0, len(X_lags_tensor), batch_size):
            batch_X_lags = X_lags_tensor[i:i+batch_size]
            batch_X_exog = X_exog_tensor[i:i+batch_size]

            with torch.no_grad():
                batch_pred = rnn(batch_X_lags, batch_X_exog)

            predictions.append(batch_pred)

        predictions = torch.cat(predictions, dim=0)
        return predictions.detach().numpy()

    train_df_y_hat = produce_forecast(rnn, train_df_X)
    test_df_y_hat = produce_forecast(rnn, test_df_X)

    return train_df_y_hat, test_df_y_hat


# In[ ]:


def separate_lag_and_exogenous_features(train_df_X, target_column='y', lag_prefix='y_lag'):
    '''
    This function separates the lag features and exogenous variables from the training dataframe.

    Args:
        train_df_X (pd.DataFrame): The dataframe containing both lag features and exogenous variables.
        target_column (str): The name of the target column (e.g., 'y').
        lag_prefix (str): The prefix used for lag columns (e.g., 'y_lag').

    Returns:
        X_lags (pd.DataFrame): DataFrame containing only the lag features.
        X_exog (pd.DataFrame): DataFrame containing only the exogenous variables.
    '''
    
    # Identify lag features (columns that start with 'y_lag')
    lag_features = [col for col in train_df_X.columns if col.startswith(lag_prefix)]
    
    # Identify exogenous variables (everything except the target and lag features)
    exog_features = [col for col in train_df_X.columns if col not in [target_column] + lag_features]
    
    # Create dataframes for lag features and exogenous features
    X_lags = train_df_X[lag_features]
    X_exog = train_df_X[exog_features]
    
    return X_lags, X_exog

