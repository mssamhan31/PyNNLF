#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# IMPORT IMPORTANT LIBRARY
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# In[ ]:


def train_model_m7_ann(hyperparameter, train_df_X, train_df_y):
    """Train an artificial neural network (ANN) for point forecasting.

    Args:
        hyperparameter (dict): Hyperparameters for the ANN including:
            - 'seed': Random seed for reproducibility.
            - 'hidden_size': Number of neurons in the hidden layer.
            - 'activation_function': Activation function ('relu', 'sigmoid', 'tanh').
            - 'learning_rate': Learning rate for the optimizer.
            - 'solver': Optimizer type ('adam' or 'sgd').
            - 'epochs': Number of training epochs.
        train_df_X (pd.DataFrame): Predictor variables for training.
        train_df_y (pd.DataFrame): Target variable for training.

    Returns:
        model (dict): Contains the trained ANN model under key 'model_ann'.
    """
    
    #UNPACK HYPERPARAMETER

    # Set random seed for reproducibility
    def set_seed(seed):
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
        def __init__(self, input_size, hidden_size, output_size):
            super(ANNModel, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.relu = nn.ReLU()  # Activation function

        def forward(self, x):
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


# In[ ]:


def produce_forecast_m7_ann(model, train_df_X, test_df_X):
    """Generate forecasts for train and test sets using a trained ANN model.

    Args:
        model (dict): Contains the trained ANN under key 'model_ann'.
        train_df_X (pd.DataFrame): Predictor variables for the training set.
        test_df_X (pd.DataFrame): Predictor variables for the test set.

    Returns:
        train_df_y_hat (pd.DataFrame): Forecasted values for the training set.
        test_df_y_hat (pd.DataFrame): Forecasted values for the test set.
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
