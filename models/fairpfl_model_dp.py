# models/fairpfl_model_dp.py — DP-compatible model variants using DPLSTM
# CRITICAL: Standard nn.LSTM is NOT compatible with Opacus.
# These models use opacus.layers.DPLSTM (single-layer, no dropout, no bidirectional).
# Use these ONLY for DP scenarios (S5, ε sweep).
# For non-DP scenarios, use models from fairpfl_model.py with nn.LSTM.

import torch
import torch.nn as nn

try:
    from opacus.layers import DPLSTM, DPMultiheadAttention
    DPLSTM_AVAILABLE = True
except ImportError:
    DPLSTM_AVAILABLE = False
    print("WARNING: opacus.layers.DPLSTM not available. DP models will fail.")


class PersonalizedLayerDP(nn.Module):
    """Device-type personalization layer — DP compatible.
    Uses DPLSTM instead of nn.LSTM."""

    def __init__(self, input_dim=46, hidden_dim=128):
        super().__init__()
        if not DPLSTM_AVAILABLE:
            raise RuntimeError("DPLSTM required. Install: pip install opacus")
        # DPLSTM: single layer, no dropout, not bidirectional
        self.lstm = DPLSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1,
        )

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return output


class SharedGlobalModelDP(nn.Module):
    """Shared global layers — DP compatible.
    Uses DPLSTM + DPMultiheadAttention + classifier.
    Both LSTM and Attention use Opacus DP-compatible replacements
    so that make_private() does not raise ShouldReplaceModuleError."""

    def __init__(self, input_dim=128, hidden_dim=64, num_classes=8,
                 num_heads=4, dropout=0.3):
        super().__init__()
        if not DPLSTM_AVAILABLE:
            raise RuntimeError("DPLSTM required for DP model.")
        self.lstm = DPLSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1,
        )
        # DPMultiheadAttention is Opacus's drop-in replacement for
        # nn.MultiheadAttention — required to avoid ShouldReplaceModuleError
        self.attention = DPMultiheadAttention(
            hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, attn_w = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.norm(attn_out + lstm_out).mean(dim=1)
        logits = self.classifier(out)
        return logits, attn_w


class FairPFLModelDP(nn.Module):
    """FairPFL model — DP compatible variant.
    Uses DPLSTM for both personal and shared layers.
    For DP scenarios: S5, ε sweep experiments."""

    def __init__(self, input_dim=46, personal_dim=128, shared_dim=64,
                 num_classes=8, num_heads=4, dropout=0.3):
        super().__init__()
        self.personal = PersonalizedLayerDP(input_dim, personal_dim)
        self.shared = SharedGlobalModelDP(
            personal_dim, shared_dim, num_classes, num_heads, dropout)

    def forward(self, x):
        personal_out = self.personal(x)
        logits, attn_weights = self.shared(personal_out)
        return logits, attn_weights

    def get_personal_params(self):
        """Return personalized layer parameters with full model-scoped keys.

        Keys are prefixed with 'personal.' to match state_dict() for
        correct disk save/load across Flower rounds.
        """
        return {f'personal.{k}': v for k, v in self.personal.named_parameters()}

    def get_shared_params(self):
        # When Opacus wraps self.shared in a GradSampleModule, named_parameters()
        # would return '_module.'-prefixed keys.  Always iterate the inner module
        # so callers always receive clean, standard parameter names.
        shared = self.shared
        if hasattr(shared, '_module'):
            shared = shared._module
        return dict(shared.named_parameters())

    def set_shared_params(self, global_params):
        shared = self.shared
        if hasattr(shared, '_module'):
            shared = shared._module
        shared_state = shared.state_dict()
        for key, value in global_params.items():
            if key in shared_state:
                shared_state[key] = value
        shared.load_state_dict(shared_state)


class DualFairPFLModelDP(nn.Module):
    """Dual FairPFL with DP — uses DPLSTM for device personalization.
    Attack head uses standard nn.Linear (no LSTM needed)."""

    def __init__(self, input_dim=46, personal_dim=128, shared_dim=64,
                 num_classes=8, num_heads=4, dropout=0.3):
        super().__init__()
        from models.dual_personalizer import DualPersonalizedHead

        if not DPLSTM_AVAILABLE:
            raise RuntimeError("DPLSTM required for DP model.")

        self.device_personal = DPLSTM(
            input_dim, personal_dim, batch_first=True, num_layers=1)
        self.shared_lstm = DPLSTM(
            personal_dim, shared_dim, batch_first=True, num_layers=1)
        # DPMultiheadAttention is the Opacus-compatible drop-in replacement
        self.shared_attention = DPMultiheadAttention(
            shared_dim, num_heads, batch_first=True)
        self.shared_norm = nn.LayerNorm(shared_dim)
        self.attack_personal = DualPersonalizedHead(shared_dim, 7, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.device_personal(x)
        out = self.dropout(out)
        out, _ = self.shared_lstm(out)
        attn_out, attn_w = self.shared_attention(out, out, out)
        out = self.shared_norm(attn_out + out).mean(dim=1)
        fine_logits, cat_logits = self.attack_personal(out)
        return fine_logits, attn_w, cat_logits

    def get_personal_params(self):
        """Return personalized parameters with full model-scoped keys.

        device_personal keys are prefixed with 'device_personal.' and
        attack_personal keys with 'attack_personal.' to match state_dict().
        """
        params = {f'device_personal.{k}': v
                  for k, v in self.device_personal.named_parameters()}
        for k, v in self.attack_personal.named_parameters():
            params[f'attack_personal.{k}'] = v
        return params

    def get_shared_params(self):
        params = {}
        for k, v in self.shared_lstm.named_parameters():
            params[f'shared_lstm.{k}'] = v
        for k, v in self.shared_attention.named_parameters():
            params[f'shared_attention.{k}'] = v
        for k, v in self.shared_norm.named_parameters():
            params[f'shared_norm.{k}'] = v
        return params

    def set_shared_params(self, global_params):
        """Load shared parameters received from server (key-based, DPLSTM-safe)."""
        for name in ['shared_lstm', 'shared_attention', 'shared_norm']:
            if hasattr(self, name):
                sub = getattr(self, name)
                sub_state = sub.state_dict()
                prefix = f'{name}.'
                updated = False
                for key in list(sub_state.keys()):
                    full_key = f'{prefix}{key}'
                    if full_key in global_params:
                        value = global_params[full_key]
                        if not isinstance(value, torch.Tensor):
                            value = torch.tensor(value)
                        sub_state[key] = value
                        updated = True
                if updated:
                    # Sync DPLSTM standard → internal names before loading
                    import re
                    pattern = re.compile(r'^(.*?)weight_ih_l(\d+)$')
                    for k in list(sub_state.keys()):
                        m = pattern.match(k)
                        if m:
                            pfx, layer = m.group(1), m.group(2)
                            for std, intl in [
                                (f'{pfx}weight_ih_l{layer}', f'{pfx}l{layer}.ih.weight'),
                                (f'{pfx}bias_ih_l{layer}',   f'{pfx}l{layer}.ih.bias'),
                                (f'{pfx}weight_hh_l{layer}', f'{pfx}l{layer}.hh.weight'),
                                (f'{pfx}bias_hh_l{layer}',   f'{pfx}l{layer}.hh.bias'),
                            ]:
                                if std in sub_state and intl in sub_state:
                                    sub_state[intl] = sub_state[std]
                    sub.load_state_dict(sub_state, strict=False)
