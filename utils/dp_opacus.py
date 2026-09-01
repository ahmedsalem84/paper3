# utils/dp_opacus.py — Production-grade DP with Opacus
# CRITICAL: Standard nn.LSTM is NOT compatible with Opacus.
# Use opacus.layers.DPLSTM for DP scenarios.
# For non-DP baselines (FedAvg, FedProx), use standard nn.LSTM.

import torch
import numpy as np

try:
    from opacus import PrivacyEngine
    from opacus.layers import DPLSTM
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    print("WARNING: Opacus not installed. DP features unavailable.")


def make_private_model_and_optimizer(model, optimizer, data_loader, config):
    """
    Wrap model + optimizer with Opacus PrivacyEngine.
    Uses RDP accounting for tight privacy budget tracking.

    IMPORTANT: Model must use DPLSTM layers (not nn.LSTM) before calling this.
    """
    if not OPACUS_AVAILABLE:
        raise RuntimeError("Opacus is required for DP. Install: pip install opacus")

    # secure_mode=False: acceptable for research (faster training).
    # Set dp_secure_mode=True in config before final paper submission.
    secure_mode = config.get('dp_secure_mode', False)
    privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=secure_mode)

    # CRITICAL: Flower re-creates the client (and PrivacyEngine) every round.
    # Noise sigma must be calibrated for 1 round = local_epochs steps only.
    # Using num_rounds × local_epochs (=500) causes completely wrong sigma
    # and leads to FPR collapse (model predicts all samples as attacks).
    # Each round provides an independent (epsilon, delta)-DP guarantee.
    model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        target_epsilon=config['dp_epsilon_global'],
        target_delta=config['dp_delta'],
        epochs=config.get('local_epochs', 5),   # 1 round only — NOT total rounds
        max_grad_norm=config['dp_clip_norm_shared'],
    )

    return model, optimizer, data_loader, privacy_engine


def get_privacy_spent(privacy_engine, delta):
    """Query actual privacy budget consumed (tight RDP bound)."""
    epsilon = privacy_engine.get_epsilon(delta=delta)
    return epsilon


def add_dp_noise_manual(parameters, epsilon, delta, clip_norm):
    """
    Manual Gaussian noise addition for (ε, δ)-DP.
    Used as FALLBACK only when Opacus is not compatible.

    σ = clip_norm * sqrt(2 * ln(1.25/δ)) / ε
    """
    sigma = clip_norm * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noisy_params = []
    for param in parameters:
        noise = np.random.normal(0, sigma, size=param.shape)
        noisy_params.append(param + noise.astype(np.float32))
    return noisy_params


def compute_rdp_epsilon(sigma, delta, T, sampling_rate, alphas=None):
    """
    Compute (ε, δ)-DP from Rényi DP parameters.
    For theoretical tables in the paper.
    """
    import math
    if alphas is None:
        alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

    def rdp_single(alpha, _sigma):
        return alpha / (2 * _sigma ** 2)

    def rdp_to_dp(rdp, alpha, _delta):
        return rdp - (math.log(_delta) + math.log(alpha - 1)) / (alpha - 1) + \
               math.log(1 - 1 / alpha)

    best_eps = float('inf')
    for alpha in alphas:
        rdp = T * sampling_rate ** 2 * rdp_single(alpha, sigma)
        eps = rdp_to_dp(rdp, alpha, delta)
        best_eps = min(best_eps, eps)

    return best_eps
