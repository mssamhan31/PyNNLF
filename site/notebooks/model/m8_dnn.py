#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

def train_model_m8_dnn(hyperparameter, train_df_X, train_df_y):
    ''' Train and test a linear model for point forecasting. 
    
    Args:
        hyperparameter (dict) : hyperparameter value of the model consisting of number of features
        train_df_X (DataFrame) : features matrix for training
        train_df_y (DataFrame) : target matrix for training
    
    Returns:
        model (dict) : trained model with all features
    '''
    
    # UNPACK HYPERPARAMETER
    seed = hyperparameter['seed']
    torch.manual_seed(seed)  # Set seed for PyTorch

    n_hidden = hyperparameter['n_hidden']
    hidden_size = hyperparameter['hidden_size']
    activation_function = hyperparameter['activation_function']
    learning_rate = hyperparameter['learning_rate']
    solver = hyperparameter['solver']
    epochs = hyperparameter['epochs']
    
    # Use proper format for X and y
    X = torch.tensor(train_df_X.values, dtype=torch.float32)
    y = torch.tensor(train_df_y.values, dtype=torch.float32).view(-1, 1) 
    
    # Define the DNN model
    class DNNModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, n_hidden, activation_function):
            super(DNNModel, self).__init__()
            self.layers = nn.ModuleList()
            self.activation_function = activation_function

            # Input layer
            self.layers.append(nn.Linear(input_size, hidden_size))
            
            # Hidden layers
            for _ in range(n_hidden - 1):
                self.layers.append(nn.Linear(hidden_size, hidden_size))
            
            # Output layer
            self.layers.append(nn.Linear(hidden_size, output_size))
            
        def forward(self, x):
            for i, layer in enumerate(self.layers[:-1]):  # Iterate through hidden layers
                x = layer(x)
                if self.activation_function == 'relu':
                    x = nn.ReLU()(x)
                elif self.activation_function == 'sigmoid':
                    x = torch.sigmoid(x)
                elif self.activation_function == 'tanh':
                    x = torch.tanh(x)
            
            # Apply the output layer without activation function
            x = self.layers[-1](x)
            return x
        
    # Model initialization
    input_size = X.shape[1]
    output_size = y.shape[1]
    model_dnn = DNNModel(input_size, hidden_size, output_size, n_hidden, activation_function)
    
    if solver == 'adam':
        optimizer = optim.Adam(model_dnn.parameters(), lr=learning_rate)
    elif solver == 'sgd':
        optimizer = optim.SGD(model_dnn.parameters(), lr=learning_rate)
    else:
        raise ValueError('Solver not found')
    
    # Loss function
    criterion = nn.MSELoss()  # Mean Squared Error loss for regression
    
    # TRAIN MODEL
    # Training loop
    for epoch in range(epochs):
        model_dnn.train()
        
        # Forward pass
        output = model_dnn(X)
        loss = criterion(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
  
    # PACK MODEL
    model = {"model_dnn": model_dnn}
  
    return model

def produce_forecast_m8_dnn(model, train_df_X, test_df_X):
    """Create forecast at the train and test set using the trained model

    Args:
        model (dict): all parameters of the trained model
        train_df_X (DataFrame): predictors of train set
        test_df_X (DataFrame): predictors of test set

    Returns:
        train_df_y_hat (DataFrame) : forecast result at train set
        test_df_y_hat (DataFrame) : forecast result at test set
    """
    
    # UNPACK MODEL
    model_dnn = model["model_dnn"]

    # PREPARE FORMAT
    train_df_X_tensor = torch.tensor(train_df_X.values, dtype=torch.float32)
    test_df_X_tensor = torch.tensor(test_df_X.values, dtype=torch.float32)

    # PRODUCE FORECAST
    # Switch model to evaluation mode for inference
    model_dnn.eval()

    # TRAIN SET FORECAST
    with torch.no_grad():  # Disable gradient calculation to save memory
        train_df_y_hat_tensor = model_dnn(train_df_X_tensor)

    # TEST SET FORECAST
    with torch.no_grad():  # Disable gradient calculation to save memory
        test_df_y_hat_tensor = model_dnn(test_df_X_tensor)
        
    # Create DataFrames of result
    train_df_y_hat = pd.DataFrame(train_df_y_hat_tensor.numpy(), index=train_df_X.index, columns=['y_hat'])
    test_df_y_hat = pd.DataFrame(test_df_y_hat_tensor.numpy(), index=test_df_X.index, columns=['y_hat'])
    
    return train_df_y_hat, test_df_y_hat

