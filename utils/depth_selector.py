# utils/depth_selector.py — Adaptive personalization depth via JS-divergence

import numpy as np
from scipy.spatial.distance import jensenshannon


def compute_js_divergence(cluster_labels, global_labels, num_classes=8):
    """
    Compute Jensen-Shannon divergence between cluster and global label distributions.

    Returns:
        float: JS divergence in [0, 1] (0 = identical, 1 = maximally different)
    """
    cluster_hist = np.histogram(cluster_labels, bins=num_classes,
                                range=(0, num_classes))[0]
    global_hist = np.histogram(global_labels, bins=num_classes,
                               range=(0, num_classes))[0]

    # Normalize with Laplace smoothing
    eps = 1e-10
    cluster_prob = (cluster_hist + eps) / (cluster_hist.sum() + num_classes * eps)
    global_prob = (global_hist + eps) / (global_hist.sum() + num_classes * eps)

    # Squared JS divergence
    return jensenshannon(cluster_prob, global_prob) ** 2


def select_personalization_depth(cluster_data, global_labels, num_classes=8,
                                 tau1=0.1, tau2=0.3):
    """
    Determine personalization depth per cluster based on heterogeneity.

    Depth levels:
        d=1: Only LSTM Layer 1 personalized (low heterogeneity)
        d=2: LSTM Layer 1 + partial Layer 2 (moderate)
        d=3: LSTM Layer 1 + Layer 2 + Attention (high heterogeneity)

    Args:
        cluster_data: dict {cluster_id: {'train_y': labels}}
        global_labels: all training labels combined
        tau1, tau2: thresholds for depth selection
    Returns:
        dict: {cluster_id: depth_level}
    """
    depths = {}
    for k, splits in cluster_data.items():
        js_div = compute_js_divergence(splits['train_y'], global_labels, num_classes)

        if js_div <= tau1:
            depths[k] = 1
        elif js_div <= tau2:
            depths[k] = 2
        else:
            depths[k] = 3

        print(f"Cluster {k}: JS_div={js_div:.4f} → depth={depths[k]}")

    return depths
