#!/usr/bin/env python3
"""
Fairness Metrics Analysis for FairPFL Paper
============================================
Computes comprehensive fairness metrics across all FL methods:
- Demographic Parity (DP) gap
- Equalized Odds (EO) gap  
- Equal Opportunity (EOP) gap
- Worst-client F1
- Inter-client F1 variance / std
- F1 gap (max - min across clients)
- Per-class fairness (Gini coefficient)
- Per-cluster fairness visualization data
"""

import json
import numpy as np
import os
import sys

# ─── Configuration ───────────────────────────────────────────
RESULTS_MAP = {
    # noDP methods
    'FairPFL': {
        42:  'results_v20_test/FairPFL_noDP/seed_42/round_metrics.json',
        123: 'results_v20_seed123/FairPFL_noDP/seed_123/round_metrics.json',
        456: 'results_v20_seed456/FairPFL_noDP/seed_456/round_metrics.json',
    },
    'FedAvg': {
        42:  'results_baselines_6class/FedAvg/seed_42/round_metrics.json',
        123: 'results_baselines_6class/FedAvg/seed_123/round_metrics.json',
        456: 'results_baselines_6class/FedAvg/seed_456/round_metrics.json',
    },
    'FedProx': {
        42:  'results_baselines_6class/FedProx/seed_42/round_metrics.json',
        123: 'results_baselines_6class/FedProx/seed_123/round_metrics.json',
        456: 'results_baselines_6class/FedProx/seed_456/round_metrics.json',
    },
    'qFFL': {
        42:  'results_baselines_6class/qFFL/seed_42/round_metrics.json',
        123: 'results_baselines_6class/qFFL/seed_123/round_metrics.json',
        456: 'results_baselines_6class/qFFL/seed_456/round_metrics.json',
    },
    'Ditto': {
        42:  'results_baselines_6class/Ditto/seed_42/round_metrics.json',
        123: 'results_baselines_6class/Ditto/seed_123/round_metrics.json',
        456: 'results_baselines_6class/Ditto/seed_456/round_metrics.json',
    },
    'pFedMe': {
        42:  'results_baselines_6class/pFedMe/seed_42/round_metrics.json',
        123: 'results_baselines_6class/pFedMe/seed_123/round_metrics.json',
        456: 'results_baselines_6class/pFedMe/seed_456/round_metrics.json',
    },
    # DP methods
    'FairPFL_DP': {
        42:  'results_v20_dp/FairPFL_DP/FairPFL_DP/seed_42/round_metrics.json',
        123: 'results_v20_dp/FairPFL_DP/FairPFL_DP/seed_123/round_metrics.json',
        456: 'results_v20_dp/FairPFL_DP/FairPFL_DP/seed_456/round_metrics.json',
    },
    'FedAvg_DP': {
        42:  'results_v20_dp/FedAvg_DP_real/FedAvg_DP/seed_42/round_metrics.json',
        123: 'results_v20_dp/FedAvg_DP_real/FedAvg_DP/seed_123/round_metrics.json',
        456: 'results_v20_dp/FedAvg_DP_real/FedAvg_DP/seed_456/round_metrics.json',
    },
}

CLASS_NAMES = ['Benign', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing']


def load_final_round(path):
    """Load the last round from a round_metrics.json file."""
    with open(path) as f:
        data = json.load(f)
    return data[-1]


def compute_fairness_metrics(round_data):
    """Compute all fairness metrics from a single round's data."""
    metrics = {}
    
    # Per-cluster F1 values
    pc_f1 = round_data.get('per_cluster_f1', {})
    f1_vals = np.array(list(pc_f1.values()))
    
    # Per-cluster TPR and FPR
    pc_tpr = round_data.get('per_cluster_tpr', {})
    pc_fpr = round_data.get('per_cluster_fpr', {})
    tpr_vals = np.array(list(pc_tpr.values()))
    fpr_vals = np.array(list(pc_fpr.values()))
    
    # Per-cluster accuracy
    pc_acc = round_data.get('per_cluster_accuracy', {})
    acc_vals = np.array(list(pc_acc.values()))
    
    # Per-cluster precision and recall
    pc_prec = round_data.get('per_cluster_precision', {})
    pc_rec = round_data.get('per_cluster_recall', {})
    
    # ─── 1. Macro-F1 ───
    metrics['macro_f1'] = round_data.get('avg_f1', 0)
    
    # ─── 2. Worst-Client F1 ───
    metrics['worst_client_f1'] = float(f1_vals.min()) if len(f1_vals) > 0 else 0
    metrics['best_client_f1'] = float(f1_vals.max()) if len(f1_vals) > 0 else 0
    
    # ─── 3. F1 Gap (max - min) ───
    metrics['f1_gap'] = float(f1_vals.max() - f1_vals.min()) if len(f1_vals) > 0 else 0
    
    # ─── 4. Inter-Client F1 Variance / Std ───
    metrics['f1_std'] = float(f1_vals.std()) if len(f1_vals) > 0 else 0
    metrics['f1_variance'] = float(f1_vals.var()) if len(f1_vals) > 0 else 0
    
    # ─── 5. Demographic Parity Gap ───
    # DP = max difference in positive prediction rates across clients
    # In multi-class: use FPR as proxy (probability of predicting attack when benign)
    if len(fpr_vals) > 0:
        metrics['dp_gap'] = float(fpr_vals.max() - fpr_vals.min())
    else:
        metrics['dp_gap'] = 0
    
    # ─── 6. Equal Opportunity Gap ───
    # EOP = max difference in TPR across clients
    if len(tpr_vals) > 0:
        metrics['eop_gap'] = float(tpr_vals.max() - tpr_vals.min())
    else:
        metrics['eop_gap'] = 0
    
    # ─── 7. Equalized Odds Gap ───
    # EO = max(TPR gap, FPR gap)
    metrics['eo_gap'] = max(metrics['eop_gap'], metrics['dp_gap'])
    
    # ─── 8. Accuracy Parity ───
    if len(acc_vals) > 0:
        metrics['acc_gap'] = float(acc_vals.max() - acc_vals.min())
        metrics['acc_std'] = float(acc_vals.std())
    
    # ─── 9. Gini Coefficient of F1 ───
    if len(f1_vals) > 0:
        sorted_f1 = np.sort(f1_vals)
        n = len(sorted_f1)
        index = np.arange(1, n + 1)
        metrics['gini_f1'] = float((2 * np.sum(index * sorted_f1) - (n + 1) * np.sum(sorted_f1)) / (n * np.sum(sorted_f1)))
    
    # ─── 10. Per-class fairness (worst class F1 across clusters) ───
    pcc_f1 = round_data.get('per_cluster_per_class_f1', {})
    if pcc_f1:
        n_classes = len(CLASS_NAMES)
        class_f1_across_clusters = []
        for ci in range(n_classes):
            vals = [pcc_f1[cid][ci] for cid in pcc_f1 if ci < len(pcc_f1[cid])]
            if vals:
                class_f1_across_clusters.append({
                    'class': CLASS_NAMES[ci],
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'min': float(np.min(vals)),
                    'max': float(np.max(vals)),
                    'gap': float(np.max(vals) - np.min(vals)),
                })
        metrics['per_class_fairness'] = class_f1_across_clusters
        # Worst class gap
        metrics['worst_class_gap'] = max(c['gap'] for c in class_f1_across_clusters)
        metrics['worst_class_f1'] = min(c['mean'] for c in class_f1_across_clusters)
    
    # Raw per-cluster data for visualization
    metrics['_per_cluster_f1'] = pc_f1
    metrics['_per_cluster_tpr'] = pc_tpr
    metrics['_per_cluster_fpr'] = pc_fpr
    
    return metrics


def _require_results(run_map, what):
    """Fail loudly if no result files exist.

    Without this the script averages over nothing and still emits output that
    looks real: zero-filled tables, or figures drawn from empty series. Reading
    those as results is the failure mode worth preventing.
    """
    import os, sys
    if not [p for m in run_map.values() for p in m.values() if os.path.exists(p)]:
        sys.exit(
            f"No result files found, so {what} cannot be produced.\n"
            "This script reads round_metrics.json from the directories created by the\n"
            "training runs, under the names used for the paper (results_v20_test/,\n"
            "results_v20_seed123/, results_v20_seed456/, results_baselines_6class/,\n"
            "results_v20_ablation/, results_v20_dp/). Run the experiments first (see\n"
            "the README), or edit the run map at the top of this file."
        )


def main():
    _require_results(RESULTS_MAP, 'the fairness panel')
    print("=" * 80)
    print("  FAIRNESS METRICS ANALYSIS — FairPFL v20")
    print("=" * 80)
    
    all_results = {}
    
    for method, seeds in RESULTS_MAP.items():
        seed_metrics = []
        for seed, path in seeds.items():
            if not os.path.exists(path):
                print(f"  ⚠️ Missing: {path}")
                continue
            rd = load_final_round(path)
            fm = compute_fairness_metrics(rd)
            seed_metrics.append(fm)
        
        if not seed_metrics:
            continue
            
        # Average across seeds
        avg = {}
        keys = [k for k in seed_metrics[0] if not k.startswith('_') and k != 'per_class_fairness']
        for k in keys:
            vals = [m[k] for m in seed_metrics if k in m]
            avg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
        
        all_results[method] = {'avg': avg, 'seeds': seed_metrics}
    
    # ─── Print Summary Table ───
    print("\n" + "─" * 100)
    print(f"{'Method':15s} │ {'Macro-F1':>10s} │ {'Worst-F1':>10s} │ {'F1 Gap':>8s} │ {'F1 Std':>8s} │ {'DP Gap':>8s} │ {'EOP Gap':>8s} │ {'EO Gap':>8s} │ {'Gini':>6s}")
    print("─" * 100)
    
    for method in ['FairPFL', 'qFFL', 'FedAvg', 'FedProx', 'Ditto', 'pFedMe', 'FairPFL_DP', 'FedAvg_DP']:
        if method not in all_results:
            continue
        a = all_results[method]['avg']
        print(f"{method:15s} │ {a['macro_f1']['mean']:>8.4f}±{a['macro_f1']['std']:.3f}"
              f" │ {a['worst_client_f1']['mean']:>8.4f}±{a['worst_client_f1']['std']:.3f}"
              f" │ {a['f1_gap']['mean']:>6.4f}"
              f" │ {a['f1_std']['mean']:>6.4f}"
              f" │ {a['dp_gap']['mean']:>6.4f}"
              f" │ {a['eop_gap']['mean']:>6.4f}"
              f" │ {a['eo_gap']['mean']:>6.4f}"
              f" │ {a['gini_f1']['mean']:>5.4f}")
    
    print("─" * 100)
    
    # ─── Per-class fairness for FairPFL vs FedAvg ───
    print("\n\n═══ PER-CLASS FAIRNESS (Inter-cluster gap) ═══\n")
    print(f"{'Class':10s} │ {'FairPFL Gap':>12s} │ {'FedAvg Gap':>12s} │ {'qFFL Gap':>12s} │ {'Winner':>8s}")
    print("─" * 70)
    
    for ci, cname in enumerate(CLASS_NAMES):
        row = {}
        for method in ['FairPFL', 'FedAvg', 'qFFL']:
            if method in all_results:
                gaps = []
                for sm in all_results[method]['seeds']:
                    if 'per_class_fairness' in sm and ci < len(sm['per_class_fairness']):
                        gaps.append(sm['per_class_fairness'][ci]['gap'])
                row[method] = np.mean(gaps) if gaps else 0
        
        winner = min(row, key=row.get) if row else ''
        print(f"{cname:10s} │ {row.get('FairPFL', 0):>10.4f}   │ {row.get('FedAvg', 0):>10.4f}   │ {row.get('qFFL', 0):>10.4f}   │ {'✓ ' + winner:>8s}")
    
    # ─── DP Impact on Fairness ───
    print("\n\n═══ DP IMPACT ON FAIRNESS ═══\n")
    print(f"{'Metric':20s} │ {'FairPFL noDP':>14s} │ {'FairPFL DP':>14s} │ {'FedAvg noDP':>14s} │ {'FedAvg DP':>14s}")
    print("─" * 85)
    
    for metric in ['macro_f1', 'worst_client_f1', 'f1_gap', 'dp_gap', 'eop_gap', 'eo_gap', 'gini_f1']:
        vals = {}
        for method in ['FairPFL', 'FairPFL_DP', 'FedAvg', 'FedAvg_DP']:
            if method in all_results:
                vals[method] = all_results[method]['avg'][metric]['mean']
        
        print(f"{metric:20s} │ {vals.get('FairPFL', 0):>12.4f}   │ {vals.get('FairPFL_DP', 0):>12.4f}   "
              f"│ {vals.get('FedAvg', 0):>12.4f}   │ {vals.get('FedAvg_DP', 0):>12.4f}")
    
    # ─── Save to JSON ───
    output = {}
    for method, data in all_results.items():
        output[method] = data['avg']
    
    with open('fairness_metrics.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Results saved to fairness_metrics.json")


if __name__ == '__main__':
    main()
