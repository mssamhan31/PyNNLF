"""Neural basis expansion analysis (N-BEATS) net load forecasting model.

Inputs:  training and test feature frames prepared by the engine, plus the
         hyperparameter mapping for this model.
Outputs: a fitted model object, and a forecast of net load in kilowatts (kW).
Key steps: build stacked fully connected blocks that emit backcast and forecast terms, then
           train the stack in PyTorch.
"""



def train_model_m18_nbeats(hyperparameter, train_df_X, train_df_y):
    """
    Train and test an NBeats model for point forecasting.
    Uses NBeats architecture for predicting time series with lag+exogenous features.
    
    Args:
        hyperparameter (dict) : model hyperparameters
        train_df_X (DataFrame) : predictors for training
        train_df_y (DataFrame) : target for training
    
    Returns:
        model : trained PyTorch NBeats model
    """
    # ---- Unpack hyperparameters ----
    input_size = train_df_X.shape[1]
    output_size = int(hyperparameter['output_size'])
    hidden_size = int(hyperparameter['hidden_size'])
    num_blocks = int(hyperparameter['num_blocks'])
    num_layers = int(hyperparameter['num_layers'])
    lr = hyperparameter['lr']
    epochs = int(hyperparameter['epochs'])
    seed = int(hyperparameter['seed'])

    # ---- Set seeds for reproducibility ----
    import torch
    import numpy as np
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ---- Define NBeats model inside the function ----
    import torch.nn as nn
    class NBeatsModel(nn.Module):
        """A neural basis expansion analysis (N-BEATS) stack for net load forecasting.

        Passes the lag sequence through the stack, concatenates the final hidden
        state with the exogenous features, then maps the result to a single output
        through a fully connected layer.
        """
        def __init__(self, input_size, output_size, hidden_size, num_blocks, num_layers):
            super(NBeatsModel, self).__init__()
            blocks = []
            for _ in range(num_blocks):
                block = []
                for layer_idx in range(num_layers):
                    block.append(nn.Linear(input_size if layer_idx==0 else hidden_size, hidden_size))
                    block.append(nn.ReLU())
                block.append(nn.Linear(hidden_size, output_size))
                blocks.append(nn.Sequential(*block))
            self.blocks = nn.ModuleList(blocks)

        def forward(self, x):
            """Run one forward pass.

            Called by PyTorch; do not call directly.

            Args:
                x (torch.Tensor): batch of lag feature sequences.

            Returns:
                torch.Tensor: predicted net load, in kilowatts (kW).
            """
            out = 0
            for block in self.blocks:
                out += block(x)
            return out

    model = NBeatsModel(input_size, output_size, hidden_size, num_blocks, num_layers)

    # ---- Training setup ----
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    X_tensor = torch.tensor(train_df_X.values, dtype=torch.float32)
    y_tensor = torch.tensor(train_df_y.values, dtype=torch.float32)

    # ---- Training loop ----
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
    
    return model

def produce_forecast_m18_nbeats(model, train_df_X, test_df_X):
    """
    Create forecast at the train and test set using the trained NBeats model.
    
    Args:
        model : trained NBeats PyTorch model
        train_df_X (DataFrame) : predictors of train set
        test_df_X (DataFrame) : predictors of test set

    Returns:
        train_df_y_hat (DataFrame) : forecast result at train set
        test_df_y_hat (DataFrame) : forecast result at test set
    """
    import torch
    model.eval()
    
    with torch.no_grad():
        X_train_tensor = torch.tensor(train_df_X.values, dtype=torch.float32)
        X_test_tensor = torch.tensor(test_df_X.values, dtype=torch.float32)
        
        y_train_hat = model(X_train_tensor).detach().cpu().numpy()
        y_test_hat  = model(X_test_tensor).detach().cpu().numpy()
    
    import pandas as pd
    train_df_y_hat = pd.DataFrame(y_train_hat, index=train_df_X.index, columns=['y_hat'])
    test_df_y_hat = pd.DataFrame(y_test_hat, index=test_df_X.index, columns=['y_hat'])
    
    return train_df_y_hat, test_df_y_hat

