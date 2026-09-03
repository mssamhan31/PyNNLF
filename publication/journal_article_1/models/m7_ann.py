# IMPORT IMPORTANT LIBRARY

"""Artificial neural network (ANN) net load forecasting model, one hidden layer.

Inputs:  training and test feature frames prepared by the engine, plus the
         hyperparameter mapping for this model.
Outputs: a fitted model object, and a forecast of net load in kilowatts (kW).
Key steps: scale the features, then train a single hidden layer PyTorch multilayer perceptron.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import os

def train_model_m7_ann(hyperparameter, train_df_X, train_df_y):
    ''' Train and test a linear model for point forecasting. 
        
    Args:
        hyperparameter (df) : hyperparameter value of the model consisting of number of features
        train_df_X (df) : features matrix for training
        train_df_y (df) : target matrix for training

    
    Returns:
        model (model) : trained model with all features
    '''
    
    #UNPACK HYPERPARAMETER

    # Set random seed for reproducibility
    def set_seed(seed):
        """Seed Python, NumPy and PyTorch so this model trains reproducibly.

        Also sets PYTHONHASHSEED, which affects hash ordering in the same process.

        Args:
            seed (int): value used to seed every random number generator.

        Returns:
            None.
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        
    seed = int(hyperparameter['seed'])

    hidden_size = hyperparameter['hidden_size']
    activation_function = hyperparameter['activation_function']
    learning_rate = hyperparameter['learning_rate']
    # learning_rate = 0.001
    solver = hyperparameter['solver']
    epochs = hyperparameter['epochs']
    
    # Use proper format for X and y
    X = torch.tensor(train_df_X.values, dtype=torch.float32)
    y = torch.tensor(train_df_y.values, dtype=torch.float32).view(-1, 1) 
    
    # Define the ANN model
    class ANNModel(nn.Module):
        """A artificial neural network (ANN) with one hidden layer for net load forecasting.

        Passes the lag sequence through the network, concatenates the final hidden
        state with the exogenous features, then maps the result to a single output
        through a fully connected layer.
        """
        def __init__(self, input_size, hidden_size, output_size):
            super(ANNModel, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.relu = nn.ReLU()  # Activation function

        def forward(self, x):
            """Run one forward pass.

            Called by PyTorch; do not call directly.

            Args:
                x (torch.Tensor): batch of lag feature sequences.

            Returns:
                torch.Tensor: predicted net load, in kilowatts (kW).
            """
            x = self.fc1(x)
            if activation_function == 'relu':
                x = self.relu(x)
            elif activation_function == 'sigmoid':
                x = torch.sigmoid(x)
            else:
                x = torch.tanh(x)
            x = self.fc2(x)
            return x
        
    # Model initialization
    input_size = X.shape[1]
    output_size = y.shape[1]
    
    set_seed(seed)
    
    model_ann = ANNModel(input_size, hidden_size, output_size)
    if solver == 'adam':
        optimizer = optim.Adam(model_ann.parameters(), lr=learning_rate)
    elif solver == 'sgd':
        optimizer = optim.SGD(model_ann.parameters(), lr=learning_rate)
    else:
        raise ValueError('Solver not found')
    
    # Loss function
    criterion = nn.MSELoss()  # Mean Squared Error loss for regression
    
    #TRAIN MODEL
    # Training loop
    for epoch in range(epochs):
        model_ann.train()
        
        # Forward pass
        output = model_ann(X)
        loss = criterion(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
  
    # PACK MODEL
    model = {"model_ann": model_ann}
  

    return model

def produce_forecast_m7_ann(model, train_df_X, test_df_X):
    """Create forecast at the train and test set using the trained model

    Args:
        model (dictionary): all parameters of the trained model
        train_df_X (df): predictors of train set
        test_df_X (df): predictors of test set

    Returns:
        train_df_y_hat (df) : forecast result at train set
        test_df_y_hat (df) : forecast result at test set
        
    """
    
    # UNPACK MODEL
    model_ann = model["model_ann"]

    # PREPARE FORMAT
    train_df_X_tensor = torch.tensor(train_df_X.values, dtype=torch.float32)
    test_df_X_tensor = torch.tensor(test_df_X.values, dtype=torch.float32)

    # PRODUCE FORECAST
    # Switch model to evaluation mode for inference
    model_ann.eval()

    # TRAIN SET FORECAST
    with torch.no_grad():  # Disable gradient calculation to save memory
        train_df_y_hat_tensor = model_ann(train_df_X_tensor)

    # TEST SET FORECAST
    with torch.no_grad():  # Disable gradient calculation to save memory
        test_df_y_hat_tensor = model_ann(test_df_X_tensor)
        
    # Create DataFrames of result
    train_df_y_hat = pd.DataFrame(train_df_y_hat_tensor, index=train_df_X.index, columns=['y_hat'])
    test_df_y_hat = pd.DataFrame(test_df_y_hat_tensor, index=test_df_X.index, columns=['y_hat'])
    
    return train_df_y_hat, test_df_y_hat


# # MESSY
