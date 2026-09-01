# metrics/fairness_metrics.py — All fairness metrics for FairPFL paper

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from models.dual_personalizer import BENIGN_CLASS_ID


def compute_all_fairness_metrics(cluster_predictions, cluster_labels,
                                 num_clusters=9, num_classes=8):
    """
    Compute all fairness metrics needed for the paper.

    Args:
        cluster_predictions: dict {cluster_id: np.array of predictions}
        cluster_labels: dict {cluster_id: np.array of true labels}
        num_classes: number of classes for per-class F1 (default 8)
    Returns:
        dict with all fairness metrics including bi-level fairness (Metric 6)
    """
    tpr_per_cluster = {}
    fpr_per_cluster = {}
    fnr_per_cluster = {}
    f1_per_cluster = {}
    per_class_f1_per_cluster = {}

    for k in cluster_predictions:
        y_true = cluster_labels[k]
        y_pred = cluster_predictions[k]

        if len(y_true) == 0:
            continue

        # Binary: attack vs benign for TPR/FPR/FNR
        y_true_bin = (y_true != BENIGN_CLASS_ID).astype(int)
        y_pred_bin = (y_pred != BENIGN_CLASS_ID).astype(int)

        cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0

        tpr_per_cluster[k] = tp / (tp + fn + 1e-10)
        fpr_per_cluster[k] = fp / (fp + tn + 1e-10)
        fnr_per_cluster[k] = fn / (fn + tp + 1e-10)
        f1_per_cluster[k] = f1_score(y_true, y_pred, average='macro',
                                     zero_division=0)

        # Per-class F1 for Metric 6 (intra-client class Gini, Eq. 27)
        pcf1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1_per_cluster[k] = pcf1.tolist()

    if not f1_per_cluster:
        return {
            'demographic_parity': 0, 'equalized_odds': 0,
            'min_f1': 0, 'std_f1': 0, 'avg_f1': 0, 'gini_f1': 0,
            'gini_class_worst': 0, 'gini_class_avg': 0,
            'f1_per_cluster': {}, 'tpr_per_cluster': {},
            'per_class_f1_per_cluster': {},
        }

    # 1. Demographic Parity: DP = max|TPR_i - TPR_j|
    tpr_vals = list(tpr_per_cluster.values())
    dp = max(tpr_vals) - min(tpr_vals) if len(tpr_vals) >= 2 else 0

    # 2. Equalized Odds: EO = max(|FPR_i - FPR_j| + |FNR_i - FNR_j|)
    fpr_vals = list(fpr_per_cluster.values())
    fnr_vals = list(fnr_per_cluster.values())
    eo = 0
    for i in range(len(fpr_vals)):
        for j in range(i + 1, len(fpr_vals)):
            eo_ij = abs(fpr_vals[i] - fpr_vals[j]) + \
                    abs(fnr_vals[i] - fnr_vals[j])
            eo = max(eo, eo_ij)

    # 3. F1 statistics
    f1_vals = list(f1_per_cluster.values())
    min_f1 = min(f1_vals)
    std_f1 = float(np.std(f1_vals))
    avg_f1 = float(np.mean(f1_vals))

    # 4. Gini coefficient (inter-client, Eq. 23)
    gini = compute_gini(f1_vals)

    # 5. Metric 6: Intra-Client Class Fairness (Eq. 27)
    gini_class_worst, gini_class_avg = compute_intra_client_class_gini(
        per_class_f1_per_cluster)

    return {
        'demographic_parity': float(dp),
        'equalized_odds': float(eo),
        'min_f1': float(min_f1),
        'std_f1': std_f1,
        'avg_f1': avg_f1,
        'gini_f1': float(gini),
        'gini_class_worst': float(gini_class_worst),
        'gini_class_avg': float(gini_class_avg),
        'f1_per_cluster': f1_per_cluster,
        'tpr_per_cluster': tpr_per_cluster,
        'per_class_f1_per_cluster': per_class_f1_per_cluster,
    }


def compute_gini(values):
    """Compute Gini coefficient of a list of values (Eq. 23)."""
    values = np.sort(np.array(values, dtype=float))
    n = len(values)
    if n == 0 or np.sum(values) == 0:
        return 0
    index = np.arange(1, n + 1)
    return float(
        (2 * np.sum(index * values) - (n + 1) * np.sum(values)) /
        (n * np.sum(values) + 1e-10)
    )


def compute_intra_client_class_gini(per_class_f1_per_cluster):
    """
    Metric 6 (Eq. 27): Intra-Client Class Fairness.

    Computes the Gini coefficient of per-class F1 scores WITHIN each cluster,
    then returns the worst-case (max) and average across clusters.

    This captures whether a cluster detects all attack types equally well,
    or if it fails on minority classes (e.g., BruteForce at 0.03% prevalence).

    Args:
        per_class_f1_per_cluster: dict {cluster_id: list of per-class F1 scores}

    Returns:
        (gini_class_worst, gini_class_avg): worst-case and average intra-client
        class Gini coefficients
    """
    if not per_class_f1_per_cluster:
        return 0.0, 0.0

    gini_per_cluster = {}
    for cid, pcf1 in per_class_f1_per_cluster.items():
        if len(pcf1) >= 2:
            gini_per_cluster[cid] = compute_gini(pcf1)

    if not gini_per_cluster:
        return 0.0, 0.0

    gini_vals = list(gini_per_cluster.values())
    return float(max(gini_vals)), float(np.mean(gini_vals))

