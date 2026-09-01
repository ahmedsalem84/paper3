# models/fairpfl_model.py — Core FairPFL model architecture
# Two-level: PersonalizedLayer (device-specific) + SharedGlobalModel (aggregated)

import torch
import torch.nn as nn


class PersonalizedLayer(nn.Module):
    """Device-type-specific personalization layer (LSTM).
    These parameters stay on-device and are NOT sent to the server.

    Architecture: LSTM(input_dim→hidden_dim) + Dropout
    LSTM is fully FL-compatible (no running stats like BatchNorm).
    Centralized test: LSTM F1=0.694 vs MLP F1=0.675 on CIC-IoT-2023.
    """

    def __init__(self, input_dim=46, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            dropout=0,
            num_layers=1,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args: x: (batch_size, 1, input_dim) or (batch_size, seq_len, input_dim)
        Returns: (batch_size, seq_len, hidden_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        output, (h_n, c_n) = self.lstm(x)
        return self.dropout(output)


class AttentionLayer(nn.Module):
    """Multi-head attention over LSTM outputs."""

    def __init__(self, hidden_dim=64, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        attn_output, attn_weights = self.attention(x, x, x)
        attn_output = self.norm(attn_output + x)
        output = attn_output.mean(dim=1)
        return output, attn_weights


class SharedGlobalModel(nn.Module):
    """Shared global layers — transmitted to server for aggregation.
    Architecture: LSTM(input_dim→hidden_dim) + MultiheadAttention + Classifier.
    LSTM has no running stats — fully FL-compatible for aggregation."""

    def __init__(self, input_dim=128, hidden_dim=64, num_classes=8,
                 num_heads=4, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1,
        )
        self.attention = AttentionLayer(hidden_dim, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        """
        Args: x: (batch_size, seq_len, input_dim) from PersonalizedLayer
        Returns: logits (batch_size, num_classes), attn_weights
        """
        lstm_out, _ = self.lstm(x)
        attended, attn_weights = self.attention(lstm_out)
        logits = self.classifier(attended)
        return logits, attn_weights


# ---- Legacy LSTM classes (kept for ablation study) ----

class PersonalizedLayerLSTM(nn.Module):
    """[LEGACY] LSTM-based personalization layer. Kept for ablation study."""

    def __init__(self, input_dim=46, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            dropout=0,
            num_layers=1,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return self.dropout(output)


class AttentionLayer(nn.Module):
    """[LEGACY] Multi-head attention over LSTM outputs. Kept for ablation study."""

    def __init__(self, hidden_dim=64, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        attn_output, attn_weights = self.attention(x, x, x)
        attn_output = self.norm(attn_output + x)
        output = attn_output.mean(dim=1)
        return output, attn_weights


class SharedGlobalModelLSTM(nn.Module):
    """[LEGACY] LSTM-based shared global model. Kept for ablation study."""

    def __init__(self, input_dim=128, hidden_dim=64, num_classes=8,
                 num_heads=4, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1,
        )
        self.attention = AttentionLayer(hidden_dim, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attended, attn_weights = self.attention(lstm_out)
        logits = self.classifier(attended)
        return logits, attn_weights


class FairPFLModel(nn.Module):
    """Combined FairPFL model for a single device-type cluster.
    Used for S5 ablation (single-dimension device-type personalization)."""

    def __init__(self, input_dim=46, personal_dim=128, shared_dim=64,
                 num_classes=8, num_heads=4, dropout=0.3):
        super().__init__()
        self.personal = PersonalizedLayer(input_dim, personal_dim, dropout)
        self.shared = SharedGlobalModel(personal_dim, shared_dim,
                                        num_classes, num_heads, dropout)

    def forward(self, x):
        personal_out = self.personal(x)
        logits, attn_weights = self.shared(personal_out)
        return logits, attn_weights

    def get_personal_params(self):
        """Return personalized classifier parameters (stay on device).

        FedPer-style split: only the classifier head is personal.
        This captures device-type-specific decision boundaries while
        allowing the full feature extractor to benefit from global
        aggregation across all clusters.

        Keys are FULL MODEL-SCOPED (e.g. 'shared.classifier.0.weight')
        so they match state_dict() and can be correctly saved/loaded from disk.
        """
        params = {}
        for k, v in self.shared.classifier.named_parameters():
            params[f'shared.classifier.{k}'] = v
        return params

    def get_shared_params(self):
        """Return shared feature extractor parameters (sent to server).

        FedPer-style split: LSTM1 (personal module) + LSTM2 + Attention
        are ALL shared globally. This ensures 99% of parameters benefit
        from collaborative learning across 9 clusters, producing a
        strong global feature representation.

        Keys are FULL MODEL-SCOPED so they match state_dict() for
        direct loading via base.load_state_dict().
        """
        params = {}
        # Personal LSTM1 — shared globally despite module name
        for k, v in self.personal.named_parameters():
            params[f'personal.{k}'] = v
        # Shared LSTM2 + Attention (exclude classifier)
        for k, v in self.shared.named_parameters():
            if not k.startswith('classifier'):
                params[f'shared.{k}'] = v
        return params

    def set_shared_params(self, global_params):
        """Load shared parameters received from server.

        Handles full-model-scoped keys spanning both personal and shared
        sub-modules (FedPer-style split).
        """
        state = self.state_dict()
        for key, value in global_params.items():
            if key in state:
                state[key] = value
        self.load_state_dict(state)


class AdaptiveFairPFLModel(nn.Module):
    """FairPFL with adaptive depth per cluster based on JS-divergence.
    Depth d=1/2/3 controls how many layers are personalized vs shared."""

    def __init__(self, depth, input_dim=37, hidden1=128, hidden2=64,
                 num_classes=8, num_heads=4, dropout=0.3):
        super().__init__()
        self.depth = depth

        # Personalized layers (always includes LSTM Layer 1)
        self.personal_lstm1 = nn.LSTM(
            input_dim, hidden1, batch_first=True, num_layers=1)

        if depth >= 2:
            self.personal_lstm2 = nn.LSTM(
                hidden1, hidden2, batch_first=True, num_layers=1)

        if depth >= 3:
            self.personal_attention = nn.MultiheadAttention(
                hidden2, num_heads, batch_first=True)
            self.personal_norm = nn.LayerNorm(hidden2)

        # Shared layers (what remains after personalization)
        if depth == 1:
            self.shared_lstm2 = nn.LSTM(
                hidden1, hidden2, batch_first=True, num_layers=1)
            self.shared_attention = nn.MultiheadAttention(
                hidden2, num_heads, batch_first=True)
            self.shared_norm = nn.LayerNorm(hidden2)
        elif depth == 2:
            self.shared_attention = nn.MultiheadAttention(
                hidden2, num_heads, batch_first=True)
            self.shared_norm = nn.LayerNorm(hidden2)
        # depth==3: everything personalized except classifier

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        out, _ = self.personal_lstm1(x)
        out = self.dropout(out)

        if self.depth >= 2:
            out, _ = self.personal_lstm2(out)
        else:
            out, _ = self.shared_lstm2(out)

        if self.depth >= 3:
            attn_out, attn_w = self.personal_attention(out, out, out)
            out = self.personal_norm(attn_out + out)
        else:
            attn_out, attn_w = self.shared_attention(out, out, out)
            out = self.shared_norm(attn_out + out)

        out = out.mean(dim=1)
        return self.classifier(out), attn_w

    def get_personal_params(self):
        """Return all personalized parameters.

        Keys are FULL MODEL-SCOPED (e.g. 'personal_lstm1.weight_ih_l0')
        so they match state_dict() and can be correctly saved/loaded from disk.
        """
        params = {f'personal_lstm1.{k}': v
                  for k, v in self.personal_lstm1.named_parameters()}
        if self.depth >= 2:
            for k, v in self.personal_lstm2.named_parameters():
                params[f'personal_lstm2.{k}'] = v
        if self.depth >= 3:
            for k, v in self.personal_attention.named_parameters():
                params[f'personal_attention.{k}'] = v
            for k, v in self.personal_norm.named_parameters():
                params[f'personal_norm.{k}'] = v
        return params

    def get_shared_params(self):
        """Return all shared parameters."""
        params = {}
        if self.depth == 1:
            for k, v in self.shared_lstm2.named_parameters():
                params[f'shared_lstm2.{k}'] = v
            for k, v in self.shared_attention.named_parameters():
                params[f'shared_attention.{k}'] = v
            for k, v in self.shared_norm.named_parameters():
                params[f'shared_norm.{k}'] = v
        elif self.depth == 2:
            for k, v in self.shared_attention.named_parameters():
                params[f'shared_attention.{k}'] = v
            for k, v in self.shared_norm.named_parameters():
                params[f'shared_norm.{k}'] = v
        for k, v in self.classifier.named_parameters():
            params[f'classifier.{k}'] = v
        return params
