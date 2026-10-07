"""Patch time series Transformer (PatchTST) net load forecasting model.

Inputs:  training and test feature frames prepared by the engine, plus the
         hyperparameter mapping for this model.
Outputs: a fitted model object, and a forecast of net load in kilowatts (kW).
Key steps: split the lag block from the exogenous block, normalise each lookback
           window, cut the window into overlapping patches, encode the patch
           sequence with a Transformer encoder, then combine the flattened
           encoding with an exogenous embedding in a linear head.

Reference: Nie et al. (2023), "A Time Series is Worth 64 Words: Long-term
Forecasting with Transformers", ICLR. Net load is a single channel, so the
paper's channel independence holds trivially; patching is what keeps the
attention cost small on a week-long lookback.
"""

import torch
import torch.nn as nn

from pynnlf.model_utils import (
    forecast_in_minibatches,
    prepare_sequence_inputs,
    train_in_minibatches,
)
from pynnlf.reproducibility import seed_everything


class PatchTSTModel(nn.Module):
    """PatchTST encoder with an exogenous branch joined before the linear head."""

    def __init__(
        self,
        lookback_length,
        patch_length,
        stride,
        d_model,
        num_heads,
        num_layers,
        feedforward_size,
        dropout,
        exog_size,
        exog_hidden_size,
    ):
        super().__init__()
        self.patch_length = patch_length
        self.stride = stride
        # The window is padded at the end by one stride (repeating the last value,
        # as in the paper) so the final patch is complete.
        num_patches = (lookback_length + stride - patch_length) // stride + 1
        self.embed = nn.Linear(patch_length, d_model)
        self.position = nn.Parameter(torch.randn(num_patches, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model, num_heads, feedforward_size, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.exog_net = nn.Sequential(nn.Linear(exog_size, exog_hidden_size), nn.ReLU())
        self.head = nn.Linear(num_patches * d_model + exog_hidden_size, 1)

    def forward(self, lookback, exog):
        """Run one forward pass. Called by PyTorch; do not call directly.

        Args:
            lookback (torch.Tensor): (batch, lookback_length) normalised windows.
            exog (torch.Tensor): (batch, exog_size) standardised exogenous features.

        Returns:
            torch.Tensor: (batch, 1) forecast in normalised units.
        """
        padded = torch.cat([lookback, lookback[:, -1:].expand(-1, self.stride)], dim=1)
        patches = padded.unfold(1, self.patch_length, self.stride)  # (batch, patches, patch_length)
        tokens = self.dropout(self.embed(patches) + self.position)
        encoded = self.encoder(tokens)
        return self.head(torch.cat([encoded.flatten(1), self.exog_net(exog)], dim=1))


def train_model_m20_patchtst(hyperparameter, train_df_X, train_df_y, forecast_horizon=None):
    """Train a PatchTST model for point forecasting.

    Args:
        hyperparameter (dict): seed, patch_length, stride, d_model, num_heads,
            num_layers, feedforward_size, dropout, exog_hidden_size, batch_size,
            epochs, learning_rate.
        train_df_X (pd.DataFrame): features matrix for training.
        train_df_y (pd.DataFrame): target for training.
        forecast_horizon (int | None): minutes; unused, accepted for the engine.

    Returns:
        dict: the fitted network, the hyperparameters and the exogenous
        standardisation statistics needed at forecast time.
    """
    seed = int(hyperparameter["seed"])
    patch_length = int(hyperparameter["patch_length"])
    stride = int(hyperparameter["stride"])
    d_model = int(hyperparameter["d_model"])
    num_heads = int(hyperparameter["num_heads"])
    num_layers = int(hyperparameter["num_layers"])
    feedforward_size = int(hyperparameter["feedforward_size"])
    dropout = float(hyperparameter["dropout"])
    exog_hidden_size = int(hyperparameter["exog_hidden_size"])
    batch_size = int(hyperparameter["batch_size"])
    epochs = int(hyperparameter["epochs"])
    learning_rate = float(hyperparameter["learning_rate"])

    inputs = prepare_sequence_inputs(train_df_X, train_df_y)

    seed_everything(seed)
    net = PatchTSTModel(
        inputs.lookback.shape[1],
        patch_length,
        stride,
        d_model,
        num_heads,
        num_layers,
        feedforward_size,
        dropout,
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


def produce_forecast_m20_patchtst(model, train_df_X, test_df_X, train_df_y=None, forecast_horizon=None):
    """Create forecasts for the train and test sets using the trained model.

    Args:
        model (dict): output of :func:`train_model_m20_patchtst`.
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
