# models/lstm_baseline.py — Standard LSTM baseline for S0 (no personalization)

import torch
import torch.nn as nn


class BaselineLSTM(nn.Module):
    """Standard LSTM for S0 baseline — single global model, no personalization.
    Architecture mirrors FairPFLModel but with all layers as 'shared'."""

    def __init__(self, input_dim=46, hidden1=128, hidden2=64,
                 num_classes=8, num_heads=4, dropout=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden1, batch_first=True, num_layers=1)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True, num_layers=1)
        self.attention = nn.MultiheadAttention(hidden2, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        attn_out, attn_w = self.attention(out, out, out)
        out = self.norm(attn_out + out).mean(dim=1)
        return self.classifier(out), attn_w

    # For baseline, ALL params are "shared" (sent to server)
    def get_personal_params(self):
        return {}

    def get_shared_params(self):
        return dict(self.named_parameters())
