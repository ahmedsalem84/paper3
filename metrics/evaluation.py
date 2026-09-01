# metrics/evaluation.py — Complete evaluation for all paper metrics

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix,
)
from metrics.fairness_metrics import compute_all_fairness_metrics
from models.dual_personalizer import BENIGN_CLASS_ID

# 8-class taxonomy (UNB alphabetical order)
# [0] Benign, [1] BruteForce, [2] DDoS, [3] DoS,
# [4] Mirai,  [5] Recon,      [6] Spoofing, [7] Web-based
CLASS_NAMES_8 = {
    0: 'Benign',
    1: 'BruteForce',
    2: 'DDoS',
    3: 'DoS',
    4: 'Mirai',
    5: 'Recon',
    6: 'Spoofing',
    7: 'Web-based',
}


def evaluate_all(cluster_predictions, cluster_labels, model=None, config=None):
    """
    Compute every metric needed for the paper.

    Args:
        cluster_predictions: dict {cluster_id: np.array}
        cluster_labels: dict {cluster_id: np.array}
        model: optional — for communication cost calculation
        config: optional — experiment config

    Returns:
        dict with all metrics
    """
    results = {}
    num_classes = config.get('num_classes', 8) if config else 8

    # === 1. Overall Detection Metrics ===
    all_preds = np.concatenate(list(cluster_predictions.values()))
    all_labels = np.concatenate(list(cluster_labels.values()))

    results['accuracy'] = float(accuracy_score(all_labels, all_preds))
    results['macro_f1'] = float(f1_score(all_labels, all_preds,
                                         average='macro', zero_division=0))
    results['weighted_f1'] = float(f1_score(all_labels, all_preds,
                                            average='weighted', zero_division=0))
    results['macro_precision'] = float(precision_score(
        all_labels, all_preds, average='macro', zero_division=0))
    results['macro_recall'] = float(recall_score(
        all_labels, all_preds, average='macro', zero_division=0))

    # Per-class F1
    per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    results['per_class_f1'] = per_class.tolist()

    # Per-class named F1 (for paper tables)
    for cls_id, cls_name in CLASS_NAMES_8.items():
        if cls_id < len(per_class):
            results[f'f1_{cls_name}'] = float(per_class[cls_id])

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    results['confusion_matrix'] = cm.tolist()

    # === 2. Per-Cluster Metrics ===
    cluster_f1s = {}
    for k in cluster_predictions:
        cluster_f1s[k] = float(f1_score(
            cluster_labels[k], cluster_predictions[k],
            average='macro', zero_division=0))

    results['per_cluster_f1'] = cluster_f1s
    results['min_cluster_f1'] = float(min(cluster_f1s.values()))
    results['max_cluster_f1'] = float(max(cluster_f1s.values()))
    results['mean_cluster_f1'] = float(np.mean(list(cluster_f1s.values())))
    results['std_cluster_f1'] = float(np.std(list(cluster_f1s.values())))

    # === 3. Fairness Metrics (including Metric 6: bi-level fairness) ===
    fairness = compute_all_fairness_metrics(
        cluster_predictions, cluster_labels, num_classes=num_classes)
    results['demographic_parity'] = fairness['demographic_parity']
    results['equalized_odds'] = fairness['equalized_odds']
    results['gini_f1'] = fairness['gini_f1']
    results['gini_class_worst'] = fairness['gini_class_worst']  # Metric 6 (Eq. 27)
    results['gini_class_avg'] = fairness['gini_class_avg']

    # === 4. Per-Attack-Class Metrics (8-class: each class is already a category) ===
    for cls_id, cls_name in CLASS_NAMES_8.items():
        if cls_id == BENIGN_CLASS_ID:
            continue  # Skip benign for attack-specific metrics
        mask = (all_labels == cls_id)
        if mask.any():
            cls_preds = all_preds[mask]
            cls_labels = all_labels[mask]
            results[f'cat_{cls_name}_accuracy'] = float(
                accuracy_score(cls_labels, cls_preds))
            results[f'cat_{cls_name}_f1'] = float(
                f1_score(cls_labels, cls_preds, average='macro', zero_division=0))

    # === 5. Communication Metrics ===
    if model is not None:
        shared_params = model.get_shared_params() if hasattr(model, 'get_shared_params') \
                        else dict(model.named_parameters())
        results['params_per_round_mb'] = float(sum(
            p.numel() * 4 / 1e6 for p in shared_params.values()))
        num_rounds = config.get('num_rounds', 100) if config else 100
        results['total_communication_mb'] = results['params_per_round_mb'] * num_rounds * 2

    # === 6. Model Size ===
    if model is not None:
        results['model_size_mb'] = float(sum(
            p.numel() * 4 / 1e6 for p in model.parameters()))

    return results

