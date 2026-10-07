"""Neural hierarchical interpolation for time series (N-HiTS) net load forecasting model.

Inputs:  training and test feature frames prepared by the engine, plus the
         hyperparameter mapping for this model.
Outputs: a fitted model object, and a forecast of net load in kilowatts (kW).
Key steps: split the lag block from the exogenous block, normalise each lookback
           window, run a stack of multi-rate MLP blocks joined by backcast
           residuals, then combine the summed block forecast with an exogenous
           embedding in a linear head.

Reference: Challu et al. (2023), "NHITS: Neural Hierarchical Interpolation for
Time Series Forecasting", AAAI. The engine asks for one value at a fixed horizon,
so the paper's output interpolation collapses to the identity; what remains of
the hierarchy is the multi-rate input pooling, one kernel size per block.
"""

import torch
import torch.nn as nn

from pynnlf.model_utils import (
    forecast_in_minibatches,
    prepare_sequence_inputs,
    train_in_minibatches,
)
from pynnlf.reproducibility import seed_everything


class NHiTSBlock(nn.Module):
    """One N-HiTS block: pool the lookback, run an MLP, emit backcast and forecast."""

    def __init__(self, lookback_length, pool_kernel_size, hidden_size, num_layers):
        super().__init__()
        self.lookback_length = lookback_length
        # Max pooling is the multi-rate sampling of the paper: a large kernel sees
        # the slow shape of the window, a kernel of 1 sees every step.
        self.pool = nn.MaxPool1d(pool_kernel_size, stride=pool_kernel_size, ceil_mode=True)
        pooled_length = -(-lookback_length // pool_kernel_size)  # ceiling division
        layers, width = [], pooled_length
        for _ in range(num_layers):
            layers += [nn.Linear(width, hidden_size), nn.ReLU()]
            width = hidden_size
        self.mlp = nn.Sequential(*layers)
        # One backcast coefficient per lookback step, plus a single forecast knot.
        self.theta = nn.Linear(hidden_size, lookback_length + 1)

    def forward(self, x):
        """Run one forward pass. Called by PyTorch; do not call directly.

        Args:
            x (torch.Tensor): (batch, lookback_length) normalised lookback windows.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: backcast (batch, lookback_length)
            and forecast (batch, 1), both in normalised units.
        """
        pooled = self.pool(x.unsqueeze(1)).squeeze(1)
        theta = self.theta(self.mlp(pooled))
        return theta[:, : self.lookback_length], theta[:, self.lookback_length :]


class NHiTSModel(nn.Module):
    """N-HiTS stack with an exogenous branch joined before the linear head."""

    def __init__(
        self,
        lookback_length,
        pool_kernel_sizes,
        hidden_size,
        num_layers,
        exog_size,
        exog_hidden_size,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            NHiTSBlock(lookback_length, kernel, hidden_size, num_layers)
            for kernel in pool_kernel_sizes
        )
        self.exog_net = nn.Sequential(nn.Linear(exog_size, exog_hidden_size), nn.ReLU())
        self.head = nn.Linear(1 + exog_hidden_size, 1)

    def forward(self, lookback, exog):
        """Run one forward pass. Called by PyTorch; do not call directly.

        Args:
            lookback (torch.Tensor): (batch, lookback_length) normalised windows.
            exog (torch.Tensor): (batch, exog_size) standardised exogenous features.

        Returns:
            torch.Tensor: (batch, 1) forecast in normalised units.
        """
        residual = lookback
        forecast = torch.zeros(lookback.shape[0], 1)
        # Each block explains what the previous ones left behind (backcast residual)
        # and adds its own share of the forecast.
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return self.head(torch.cat([forecast, self.exog_net(exog)], dim=1))


def train_model_m19_nhits(hyperparameter, train_df_X, train_df_y, forecast_horizon=None):
    """Train an N-HiTS model for point forecasting.

    Args:
        hyperparameter (dict): seed, hidden_size, num_layers, pool_kernel_sizes,
            exog_hidden_size, batch_size, epochs, learning_rate.
        train_df_X (pd.DataFrame): features matrix for training.
        train_df_y (pd.DataFrame): target for training.
        forecast_horizon (int | None): minutes; unused, accepted for the engine.

    Returns:
        dict: the fitted network, the hyperparameters and the exogenous
        standardisation statistics needed at forecast time.
    """
    seed = int(hyperparameter["seed"])
    hidden_size = int(hyperparameter["hidden_size"])
    num_layers = int(hyperparameter["num_layers"])
    kernels = hyperparameter["pool_kernel_sizes"]
    if not isinstance(kernels, (list, tuple)):
        kernels = str(kernels).split(",")  # allow "8,4,1" as well as a YAML list
    pool_kernel_sizes = [int(kernel) for kernel in kernels]
    exog_hidden_size = int(hyperparameter["exog_hidden_size"])
    batch_size = int(hyperparameter["batch_size"])
    epochs = int(hyperparameter["epochs"])
    learning_rate = float(hyperparameter["learning_rate"])

    inputs = prepare_sequence_inputs(train_df_X, train_df_y)

    seed_everything(seed)
    net = NHiTSModel(
        inputs.lookback.shape[1],
        pool_kernel_sizes,
        hidden_size,
        num_layers,
        inputs.exog.shape[1],
        exog_hidden_size,
    )
    train_in_minibatches(net, inputs, batch_size, epochs, learning_rate)

    return {
        "net": net,
        "hyperparameter": hyperparameter,
        "exog_mean": inputs.exog_mean,
        "exog_std": inputs.exog_std,
    }


def produce_forecast_m19_nhits(model, train_df_X, test_df_X, train_df_y=None, forecast_horizon=None):
    """Create forecasts for the train and test sets using the trained model.

    Args:
        model (dict): output of :func:`train_model_m19_nhits`.
        train_df_X (pd.DataFrame): predictors of the train set.
        test_df_X (pd.DataFrame): predictors of the test set.
        train_df_y (pd.DataFrame | None): unused, accepted for the engine.
        forecast_horizon (int | None): unused, accepted for the engine.

    Returns:
        tuple[np.ndarray, np.ndarray]: train and test forecasts in kilowatts (kW).
    """
    batch_size = int(model["hyperparameter"]["batch_size"])
    train_y_hat = forecast_in_minibatches(model, train_df_X, batch_size)
    test_y_hat = forecast_in_minibatches(model, test_df_X, batch_size)
    return train_y_hat, test_y_hat
