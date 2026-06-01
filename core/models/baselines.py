"""Baseline models for ablation comparison against ReHeartNet.

All models accept (B, SEQ_LEN, 1) PPG input and produce (B, SEQ_LEN, 1)
ECG output, matching the ReHeartNet interface so they drop into the same
train_fold / evaluate_fold pipeline without any changes.

Three baselines:
  LinearRegressionBaseline  -- per-window OLS, no deep learning
  SimpleLSTMBaseline        -- 5-layer stacked LSTM, no dense connections
  PlainBiLSTMBaseline       -- 5-layer stacked BiLSTM, no dense connections

The latter two ablate the dense connectivity of ReHeartNet; S-LSTM further
ablates bidirectionality relative to P-BiLSTM.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Linear Regression Baseline
# ---------------------------------------------------------------------------

class LinearRegressionBaseline(nn.Module):
    """Per-window linear mapping: PPG (flattened) → ECG (flattened).

    Implemented as a single Linear layer so it participates in the same
    DataLoader/optimizer loop as the deep models.  Weight initialisation
    mimics OLS when trained with MSE loss to convergence.

    Input:  (B, SEQ_LEN, 1)
    Output: (B, SEQ_LEN, 1)
    """

    def __init__(self, seq_len: int = 1250):
        super().__init__()
        self.seq_len = seq_len
        self.linear = nn.Linear(seq_len, seq_len, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, SEQ_LEN, 1)
        x_flat = x.squeeze(-1)                   # (B, SEQ_LEN)
        out = self.linear(x_flat)                # (B, SEQ_LEN)
        return out.unsqueeze(-1)                 # (B, SEQ_LEN, 1)


# ---------------------------------------------------------------------------
# Simple LSTM Baseline (unidirectional, no dense connections)
# ---------------------------------------------------------------------------

class SimpleLSTMBaseline(nn.Module):
    """5-layer stacked unidirectional LSTM with a linear output head.

    Ablates BOTH bidirectionality AND dense connections relative to ReHeartNet.

    Input:  (B, SEQ_LEN, 1)
    Output: (B, SEQ_LEN, 1)
    """

    def __init__(self, hidden_size: int = 64, num_layers: int = 5, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = 1,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
            bidirectional = False,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, SEQ_LEN, 1)
        out, _ = self.lstm(x)           # (B, SEQ_LEN, hidden_size)
        return self.head(out)           # (B, SEQ_LEN, 1)


# ---------------------------------------------------------------------------
# Plain BiLSTM Baseline (bidirectional, no dense connections)
# ---------------------------------------------------------------------------

class PlainBiLSTMBaseline(nn.Module):
    """5-layer stacked BiLSTM with a linear output head.

    Ablates ONLY the dense connections relative to ReHeartNet; retains
    bidirectionality.  The output dimension of each layer (2*hidden_size)
    is fed directly as input to the next layer.

    Input:  (B, SEQ_LEN, 1)
    Output: (B, SEQ_LEN, 1)
    """

    def __init__(self, hidden_size: int = 64, num_layers: int = 5):
        super().__init__()
        # Build layers manually to keep the same structure as ReHeartNet blocks
        layers = []
        in_size = 1
        for _ in range(num_layers):
            layers.append(
                nn.LSTM(
                    input_size    = in_size,
                    hidden_size   = hidden_size,
                    num_layers    = 1,
                    batch_first   = True,
                    bidirectional = True,
                )
            )
            in_size = hidden_size * 2   # bidirectional doubles the output dim

        self.layers = nn.ModuleList(layers)
        self.head   = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, SEQ_LEN, 1)
        h = x
        for lstm in self.layers:
            h, _ = lstm(h)              # (B, SEQ_LEN, 2*hidden_size)
        return self.head(h)             # (B, SEQ_LEN, 1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY = {
    "linear":    LinearRegressionBaseline,
    "lstm":      SimpleLSTMBaseline,
    "bilstm":    PlainBiLSTMBaseline,
}


def get_model(name: str, hidden_size: int = 64, seq_len: int = 1250) -> nn.Module:
    """Return a model instance by short name.

    Args:
        name:        One of "linear", "lstm", "bilstm", "reheartnet".
        hidden_size: Hidden units for LSTM-based models.
        seq_len:     Sequence length for the linear baseline.

    Returns:
        Initialised nn.Module with the ReHeartNet-compatible interface.
    """
    name = name.lower()
    if name == "reheartnet":
        from core.models.reheartnet import ReHeartNet  # noqa: PLC0415
        return ReHeartNet(hidden_size=hidden_size)
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(_REGISTRY)+ ['reheartnet']}")
    cls = _REGISTRY[name]
    if name == "linear":
        return cls(seq_len=seq_len)
    return cls(hidden_size=hidden_size)
