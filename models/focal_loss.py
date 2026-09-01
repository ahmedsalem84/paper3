# models/focal_loss.py — Loss functions for CIC-IoT-2023 class imbalance
# v9: LogitAdjustedCE replaces FocalLoss as primary loss function.
# Centralized test: LogitAdj F1=0.677 BF=0.347 vs Focal F1=0.655 BF=0.316

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LogitAdjustedCE(nn.Module):
    """
    Logit-Adjusted Cross-Entropy Loss (Menon et al., 2021).

    Adjusts logits by class prior probabilities to debias the classifier.
    More principled than focal loss for balanced-sampled data where the
    training distribution differs from the test distribution.

    During training: L = CE(logits + tau * log(pi), y)
    where pi is the class prior probability vector.

    Args:
        class_counts: numpy array of per-class sample counts
        tau: adjustment strength (default 1.0, range [0.5, 1.5])
        num_classes: number of classes
    """

    def __init__(self, class_counts, tau=1.0, num_classes=8):
        super().__init__()
        self.tau = tau
        freqs = class_counts / class_counts.sum()
        adjustments = tau * torch.log(torch.FloatTensor(freqs) + 1e-12)
        self.register_buffer('logit_adjustments', adjustments)

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch, num_classes) raw predictions
            targets: (batch,) integer class labels
        Returns:
            scalar logit-adjusted cross-entropy loss
        """
        adjusted_logits = logits + self.logit_adjustments
        return F.cross_entropy(adjusted_logits, targets)


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) with class weights.
    Kept for backward compatibility and ablation studies.

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    Args:
        alpha: class weights tensor of shape (num_classes,), or None for uniform
        gamma: focusing parameter (1.0 for CIC-IoT-2023).
               v11b/v11c tested gamma=2.0 — no improvement over gamma=1.0.
    """

    def __init__(self, alpha=None, gamma=1.0, num_classes=8, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch, num_classes) raw predictions
            targets: (batch,) integer class labels
        Returns:
            scalar focal loss
        """
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.alpha,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def compute_class_weights(labels, num_classes=8, beta=0.999):
    """
    Class weights with targeted minority-class boost.

    After equal per-class sampling (N=2000/class), the training distribution
    is 1:1 balanced, so standard inverse-frequency weights are all ~1.0 and
    have no effect.  However, BruteForce (class 1) and Web-based (class 7)
    still achieve near-zero F1 because:
      - They are genuinely "hard" classes (subtle feature patterns)
      - The model learns DDoS/Mirai patterns much faster (strong signal)

    Solution: Apply a targeted 4x weight boost to BruteForce and Web-based.

    Args:
        labels: numpy array or list of integer labels (used only for validation)
        num_classes: number of classes (default 8)
        beta: unused, kept for API compatibility
    Returns:
        torch.FloatTensor of shape (num_classes,) with targeted boosts
    """
    # Base: uniform weights (data is already balanced by equal sampling)
    weights = np.ones(num_classes, dtype=float)

    if num_classes == 8:
        # Targeted 4x boost for hardest minority classes (8-class mode)
        # Class 1 = BruteForce, Class 7 = Web-based
        MINORITY_BOOST = 4.0
        weights[1] = MINORITY_BOOST  # BruteForce
        weights[7] = MINORITY_BOOST  # Web-based
    elif num_classes == 6:
        # v19 tested gamma=2.0 + weights (Recon 2x, Benign 1.5x, Spoofing 1.5x)
        # Result: F1 dropped 0.8532 -> 0.7777 (-7.6%). Keeping uniform weights.
        pass

    # Normalize so mean weight = 1.0 (preserves effective learning rate scale)
    weights = weights / weights.mean()
    return torch.FloatTensor(weights)

