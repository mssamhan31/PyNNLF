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


def train_model_m13_lstm(hyperparameter, train_df_X, train_df_y):
    """Train an LSTM model for point forecasting using lag and exogenous features.

    The LSTM captures temporal patterns from lag features. The last hidden state 
    is concatenated with exogenous features and passed through a fully connected 
    layer to produce the forecast.

    Args:
        hyperparameter (dict): Model hyperparameters (seed, input_size, hidden_size, 
                               num_layers, output_size, batch_size, epochs, learning_rate).
        train_df_X (pd.DataFrame): Training predictors including lag and exogenous features.
        train_df_y (pd.DataFrame): Training target values.

    Returns:
        model (dict): Dictionary containing the trained LSTM model, hyperparameters, 
                      and original training data.
    """
    
    #UNPACK HYPERPARAMETER
    seed = int(hyperparameter['seed'])
    input_size = int(hyperparameter['input_size']) #this is one since we only use lag features to be fed into the LSTM. The exogenous features like calenndar are fed to the fully connected layer, together with the last hidden state of LSTM.
    hidden_size = int(hyperparameter['hidden_size']) #this is the size of hidden state, and we aim to use many to one architecture. Meaning we only take the last hidden state as output, and fed into the fully connected layer.
    num_layers = int(hyperparameter['num_layers']) # we use 1 by default to make it simple. 
    output_size = int(hyperparameter['output_size']) #this is one since we only predict one value.
    batch_size = int(hyperparameter['batch_size']) #using minibatch is important cuz if we train all samples at once, the memory is not enough.
    epochs = int(hyperparameter['epochs'])
    learning_rate = hyperparameter['learning_rate']  # No change for learning rate

    
    #DEFINE MODEL AND TRAINING FUNCTION
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, exog_size, output_size=1):
            super(LSTMModel, self).__init__()
            
            # Define the LSTM layer
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            
            # Define the Fully Connected (FC) layer
            # The FC layer input size is the concatenation of LSTM output and exogenous variables
            self.fc = nn.Linear(hidden_size + exog_size, output_size)  # exog_size is the number of exogenous features

        def forward(self, x, exogenous_data):
            # Pass the input through the LSTM
            out, (h_n, c_n) = self.lstm(x)
            
            # Get the last timestep hidden state (h3)
            last_hidden_state = out[:, -1, :]  # Shape: (batch_size, hidden_size)
            
            # Concatenate the LSTM output (h3) with the exogenous variables (for timestep t+100)
            combined_input = torch.cat((last_hidden_state, exogenous_data), dim=1)  # Shape: (batch_size, hidden_size + exog_size)
            
            # Pass the combined input through the FC layer
            out = self.fc(combined_input)
            return out
        
    def train_lstm_with_minibatches(model, train_loader, epochs, learning_rate=0.001):
        # Define the loss function (Mean Squared Error)
        criterion = nn.MSELoss()
        
        # Define the optimizer (Adam)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            print(f'Epoch [{epoch+1}/{epochs}]')
            start_time = time.time()

            model.train()  # Set model to training mode
            # print(f'I am here')
            
            # Iterate over mini-batches
            batch_no = 1
            for X_lags_batch, X_exog_batch, y_batch in train_loader:
                # print(f'I am here now')
                # Print the loss and time taken for this epoch
                print(f'Epoch [{epoch+1}/{epochs}] and batch [{batch_no}/{len(train_loader)}]')
                batch_no += 1
                # Forward pass
                predictions = model(X_lags_batch, X_exog_batch)
                loss = criterion(predictions, y_batch)

                # Backward pass
                optimizer.zero_grad()  # Zero gradients from previous step
                loss.backward()  # Backpropagate the error
                optimizer.step()  # Update the model's weights
                
            
            
            end_time = time.time()
            epoch_time = end_time - start_time
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, time taken: {epoch_time:.2f} seconds')
            
    def set_seed(seed=seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
            
    # PREPARE TRAIN DATA
    # SEPARATE LAG AND EXOGENOUS FEATURES
    X_lags, X_exog = separate_lag_and_exogenous_features(train_df_X)
    X_lags_tensor = torch.tensor(X_lags.values, dtype=torch.float32)  # Shape: (batch_size, sequence_length, input_size)
    X_exog_tensor = torch.tensor(X_exog.values, dtype=torch.float32)  # Shape: (batch_size, exog_size)
    y_tensor = torch.tensor(train_df_y.values, dtype=torch.float32).view(-1, 1)
    
    total_lag_features = X_lags_tensor.shape[1]  # Number of lag features (columns)
    sequence_length = total_lag_features // input_size
    exog_size = X_exog_tensor.shape[1]  # Number of exogenous features
    
    # Reshaping X_lags_tensor to 3D: (batch_size, sequence_length, input_size)
    X_lags_tensor = X_lags_tensor.view(-1, sequence_length, input_size)

    
    # INITIALIZE MODEL and MAKE TRAINING BATCHES
    set_seed(seed = seed) # Set random seed for reproducibility
    lstm = LSTMModel(input_size, hidden_size, num_layers, exog_size, output_size)
    # Create a TensorDataset with your features and target
    train_data = TensorDataset(X_lags_tensor, X_exog_tensor, y_tensor)
    # Create a DataLoader to handle mini-batching
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # TRAIN MODEL
    train_lstm_with_minibatches(lstm, train_loader, epochs=epochs, learning_rate=learning_rate)

  
    # PACK MODEL
    model = {"lstm": lstm, 'hyperparameter': hyperparameter, "train_df_X": train_df_X, "train_df_y": train_df_y}
  

    return model


# In[ ]:


def produce_forecast_m13_lstm(model, train_df_X, test_df_X):
    """Generate forecasts for train and test sets using a trained LSTM model.

    The function handles lag and exogenous features, applies mini-batching 
    to avoid memory issues, and returns predictions for both train and test sets.

    Args:
        model (dict): Dictionary containing the trained LSTM model and hyperparameters.
        train_df_X (pd.DataFrame): Training predictors including lag and exogenous features.
        test_df_X (pd.DataFrame): Test predictors including lag and exogenous features.

    Returns:
        train_df_y_hat (np.ndarray): Forecasted values for the training set.
        test_df_y_hat (np.ndarray): Forecasted values for the test set.
    """
    
    # UNPACK MODEL
    lstm = model['lstm']
    hyperparameter = model['hyperparameter']
    
    #UNPACK HYPERPARAMETER
    seed = int(hyperparameter['seed'])
    input_size = int(hyperparameter['input_size']) #this is one since we only use lag features to be fed into the LSTM. The exogenous features like calenndar are fed to the fully connected layer, together with the last hidden state of LSTM.
    hidden_size = int(hyperparameter['hidden_size']) #this is the size of hidden state, and we aim to use many to one architecture. Meaning we only take the last hidden state as output, and fed into the fully connected layer.
    num_layers = int(hyperparameter['num_layers']) # we use 1 by default to make it simple. 
    output_size = int(hyperparameter['output_size']) #this is one since we only predict one value.
    batch_size = int(hyperparameter['batch_size']) #using minibatch is important cuz if we train all samples at once, the memory is not enough.
    epochs = int(hyperparameter['epochs'])
    learning_rate = hyperparameter['learning_rate']  # No change for learning rate
    
    # PRODUCE FORECAST
    def produce_forecast(lstm, X):
        # Convert X into X_lag and X_exog
        X_lags, X_exog = separate_lag_and_exogenous_features(X)
        X_lags_tensor = torch.tensor(X_lags.values, dtype=torch.float32)  # Shape: (batch_size, sequence_length, input_size)
        X_exog_tensor = torch.tensor(X_exog.values, dtype=torch.float32)  # Shape: (batch_size, exog_size)
        # y_tensor = torch.tensor(train_df_y.values, dtype=torch.float32).view(-1, 1) to be deleted.
        
        total_lag_features = X_lags_tensor.shape[1]  # Number of lag features (columns)
        sequence_length = total_lag_features // input_size
        exog_size = X_exog_tensor.shape[1]  # Number of exogenous features
        
        # Reshaping X_lags_tensor to 3D: (batch_size, sequence_length, input_size)
        X_lags_tensor = X_lags_tensor.view(-1, sequence_length, input_size)
        
        #predictions = lstm(X_lags_tensor, X_exog_tensor) #this doesn't work because of the batch size is too big, not enough memory.
        predictions = []
        for i in range(0, len(X_lags_tensor), batch_size):
            # Get the current minibatch for both X_lags_tensor and X_exog_tensor
            batch_X_lags = X_lags_tensor[i:i+batch_size]
            batch_X_exog = X_exog_tensor[i:i+batch_size]
            
            with torch.no_grad():
                # Make predictions for the minibatch
                batch_pred = lstm(batch_X_lags, batch_X_exog)
            
            # Store the predictions for the current batch
            predictions.append(batch_pred)

        # Concatenate all predictions to get the full result
        predictions = torch.cat(predictions, dim=0)

        
        return predictions.detach().numpy()
    
    train_df_y_hat = produce_forecast(lstm, train_df_X)
    test_df_y_hat = produce_forecast(lstm, test_df_X)
    
    return train_df_y_hat, test_df_y_hat


# In[ ]:


def separate_lag_and_exogenous_features(train_df_X, target_column='y', lag_prefix='y_lag'):
    """
    Split a dataframe into lag features and exogenous variables.

    Args:
        train_df_X (pd.DataFrame): DataFrame containing lagged features, exogenous variables, and possibly the target.
        target_column (str, optional): Name of the target column to exclude from exogenous features. Defaults to 'y'.
        lag_prefix (str, optional): Prefix that identifies lagged features. Defaults to 'y_lag'.

    Returns:
        X_lags (pd.DataFrame): DataFrame containing only columns that are lag features.
        X_exog (pd.DataFrame): DataFrame containing only exogenous variables (excluding target and lag features).
    """
    
    # Identify lag features (columns that start with 'y_lag')
    lag_features = [col for col in train_df_X.columns if col.startswith(lag_prefix)]
    
    # Identify exogenous variables (everything except the target and lag features)
    exog_features = [col for col in train_df_X.columns if col not in [target_column] + lag_features]
    
    # Create dataframes for lag features and exogenous features
    X_lags = train_df_X[lag_features]
    X_exog = train_df_X[exog_features]
    
    return X_lags, X_exog

