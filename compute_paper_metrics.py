#!/usr/bin/env python3
"""
compute_paper_metrics.py
========================
Builds the table data for §VI of paper/FairPFL_full_paper.md.
Adds beyond compute_fairness_metrics.py:
  - Jain's fairness index of per-cluster F1
  - Bottom-quintile (bottom 2 of 9 clusters) F1, with delta vs FedAvg
  - Paired t-test p-value vs FedAvg (3 seeds)
  - Per-class F1 across clusters (worst-class, per-class gap)
  - Sensitivity sweep (q_max, mu, W) summary
Writes paper_metrics.json + prints a markdown summary.
"""
import json, os, numpy as np
from scipy import stats

CLASS_NAMES = ['Benign', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing']

RUNS = {
  'FairPFL': {42:'results_v20_test/FairPFL_noDP/seed_42/round_metrics.json',
              123:'results_v20_seed123/FairPFL_noDP/seed_123/round_metrics.json',
              456:'results_v20_seed456/FairPFL_noDP/seed_456/round_metrics.json'},
  'FedAvg':  {s:f'results_baselines_6class/FedAvg/seed_{s}/round_metrics.json' for s in (42,123,456)},
  'FedProx': {s:f'results_baselines_6class/FedProx/seed_{s}/round_metrics.json' for s in (42,123,456)},
  'qFFL':    {s:f'results_baselines_6class/qFFL/seed_{s}/round_metrics.json' for s in (42,123,456)},
  'Ditto':   {s:f'results_baselines_6class/Ditto/seed_{s}/round_metrics.json' for s in (42,123,456)},
  'pFedMe':  {s:f'results_baselines_6class/pFedMe/seed_{s}/round_metrics.json' for s in (42,123,456)},
  'FairPFL_DP':{s:f'results_v20_dp/FairPFL_DP/FairPFL_DP/seed_{s}/round_metrics.json' for s in (42,123,456)},
  'FedAvg_DP':{s:f'results_v20_dp/FedAvg_DP_real/FedAvg_DP/seed_{s}/round_metrics.json' for s in (42,123,456)},
}

ABLATIONS = {a: {s:f'results_v20_ablation/{a}/{a}/seed_{s}/round_metrics.json'
                for s in (42,123,456)}
             for a in ['A1_no_fairness','A2_no_personal','A3_fixed_depth','A4_no_warmup','A5_no_proximal']}

def _preflight(run_map):
    """Stop with a clear message instead of emitting a file full of zeros.

    Every path below is relative to the repository root and refers to a results
    directory produced by the training runs. If none of them exists, the script
    would otherwise average over nothing and write paper_metrics.json filled
    with 0.0000, which looks like a real result. Fail loudly instead.
    """
    import sys
    found = [p for m in run_map.values() for p in m.values() if os.path.exists(p)]
    if not found:
        sys.exit(
            "No result files found.\n"
            "This script reads round_metrics.json from the directories produced by\n"
            "the training runs, and expects the names used for the paper:\n"
            "  results_v20_test/, results_v20_seed123/, results_v20_seed456/  (FairPFL)\n"
            "  results_baselines_6class/   results_v20_ablation/   results_v20_dp/\n"
            "Run the experiments first (see the README), or edit the RUNS map at the\n"
            "top of this file to point at your own --results_dir names."
        )


_preflight(RUNS)

SENSITIVITY = {
  'qmax_1.0':'results_sensitivity/qmax_1.0/FairPFL_noDP/seed_42/round_metrics.json',
  'qmax_2.0':'results_sensitivity/qmax_2.0/FairPFL_noDP/seed_42/round_metrics.json',
  'qmax_5.0':'results_sensitivity/qmax_5.0/FairPFL_noDP/seed_42/round_metrics.json',
  'mu_0.000':'results_sensitivity/mu_0.000/FairPFL_noDP/seed_42/round_metrics.json',
  'mu_0.010':'results_sensitivity/mu_0.010/FairPFL_noDP/seed_42/round_metrics.json',
  'mu_0.100':'results_sensitivity/mu_0.100/FairPFL_noDP/seed_42/round_metrics.json',
  'warmup_0':'results_sensitivity/warmup_0/FairPFL_noDP/seed_42/round_metrics.json',
  'warmup_10':'results_sensitivity/warmup_10/FairPFL_noDP/seed_42/round_metrics.json',
  'warmup_40':'results_sensitivity/warmup_40/FairPFL_noDP/seed_42/round_metrics.json',
}

def load_last(path):
    return json.load(open(path))[-1]

def jain(vals):
    v = np.asarray(vals, dtype=float)
    s1 = v.sum(); s2 = (v*v).sum()
    return float(s1*s1 / (len(v) * s2)) if s2 > 0 else 0.0

def bottom_quintile_f1(per_cluster_f1):
    vals = sorted(per_cluster_f1.values())
    k = max(1, int(round(len(vals)*0.20)))   # 9 clusters → 2 (rounded)
    return float(np.mean(vals[:k])), k

def seed_summary(seeds_map):
    out = {'macro_f1':[], 'worst_f1':[], 'f1_gap':[], 'jain':[], 'gini':[],
           'bottom_q':[], 'per_cluster':[], 'per_class':[]}
    for s,p in seeds_map.items():
        if not os.path.exists(p):
            continue
        r = load_last(p)
        pc = {int(k):float(v) for k,v in r.get('per_cluster_f1',{}).items()}
        out['macro_f1'].append(r.get('avg_f1',0))
        out['worst_f1'].append(min(pc.values()) if pc else 0)
        out['f1_gap'].append(max(pc.values())-min(pc.values()) if pc else 0)
        out['jain'].append(jain(list(pc.values())))
        out['gini'].append(r.get('gini_f1',0))
        bq, k = bottom_quintile_f1(pc); out['bottom_q'].append(bq)
        out['per_cluster'].append(pc)
        pcc = r.get('per_cluster_per_class_f1',{})
        # pcc format: {"0": [c0_f1, c1_f1, ...], ...} for that cluster
        out['per_class'].append({int(k):v for k,v in pcc.items()})
    return out

def mean_std(xs):
    if not xs: return (0.0, 0.0)
    return float(np.mean(xs)), float(np.std(xs))

# ── 1. Main + DP table ────────────────────────────────────────────
print("=" * 110)
print(f"{'Method':14s} {'Macro-F1':>16s} {'Worst-F1':>16s} {'F1-gap':>10s} {'Jain':>10s} {'Bot-Q':>10s} {'p vs FA':>10s}")
print("─" * 110)

summary = {}
for m, sm in RUNS.items():
    d = seed_summary(sm)
    summary[m] = d
    mF = mean_std(d['macro_f1']); wF = mean_std(d['worst_f1'])
    fg = mean_std(d['f1_gap']);  jn = mean_std(d['jain'])
    bq = mean_std(d['bottom_q'])
    print(f"{m:14s} {mF[0]:>8.4f}±{mF[1]:.4f}  {wF[0]:>8.4f}±{wF[1]:.4f}  "
          f"{fg[0]:>8.4f}  {jn[0]:>8.4f}  {bq[0]:>8.4f}", end='')
    if m != 'FedAvg' and len(d['macro_f1']) == 3 and len(summary.get('FedAvg',{}).get('macro_f1',[])) == 3:
        t, p = stats.ttest_rel(d['macro_f1'], summary['FedAvg']['macro_f1'])
        print(f"  {p:>8.4f}")
    else:
        print(f"  {'—':>8s}")

# ── 2. Ablation ────────────────────────────────────────────────────
print()
print("ABLATIONS (vs full FairPFL)")
print(f"{'Scenario':18s} {'Macro-F1':>16s} {'Worst-F1':>16s} {'F1-gap':>10s} {'Δ F1':>8s} {'Δ Worst':>10s}")
ablation_summary = {}
base = summary['FairPFL']
for a, sm in ABLATIONS.items():
    d = seed_summary(sm)
    ablation_summary[a] = d
    mF = mean_std(d['macro_f1']); wF = mean_std(d['worst_f1']); fg = mean_std(d['f1_gap'])
    dF = mF[0] - np.mean(base['macro_f1']); dW = wF[0] - np.mean(base['worst_f1'])
    print(f"{a:18s} {mF[0]:>8.4f}±{mF[1]:.4f}  {wF[0]:>8.4f}±{wF[1]:.4f}  {fg[0]:>8.4f}  {dF:>+8.4f}  {dW:>+8.4f}")

# ── 3. Sensitivity ─────────────────────────────────────────────────
print()
print("SENSITIVITY (seed 42, default = qmax=3.0, mu=0.001, W=20 → F1=0.8648)")
print(f"{'Config':12s} {'Macro-F1':>10s} {'Worst-F1':>10s} {'F1-gap':>10s} {'Δ F1':>8s}")
sens_summary = {}
DEFAULT_F1 = 0.8648
for c, p in SENSITIVITY.items():
    if not os.path.exists(p):
        print(f"{c:12s} MISSING"); continue
    r = load_last(p)
    pc = {int(k):float(v) for k,v in r.get('per_cluster_f1',{}).items()}
    F = r.get('avg_f1',0)
    W = min(pc.values()) if pc else 0
    G = (max(pc.values())-min(pc.values())) if pc else 0
    sens_summary[c] = {'macro_f1':F,'worst_f1':W,'f1_gap':G,'jain':jain(list(pc.values()))}
    print(f"{c:12s} {F:>10.4f} {W:>10.4f} {G:>10.4f} {F-DEFAULT_F1:>+8.4f}")

# ── 4. Per-class F1 across clusters (worst-class, gap) ─────────────
print()
print("PER-CLASS F1 (mean across 9 clusters x 3 seeds)")
print(f"{'Method':14s} {'Benign':>8s} {'DDoS':>8s} {'DoS':>8s} {'Mirai':>8s} {'Recon':>8s} {'Spoofing':>9s} {'Worst':>9s} {'WorstCls':>10s}")
per_class_summary = {}
for m, d in summary.items():
    # aggregate per-class F1 across seeds and clusters
    cls_means = []
    for ci in range(6):
        vals = []
        for seed_pcc in d['per_class']:
            for cid, cls_f1s in seed_pcc.items():
                if ci < len(cls_f1s):
                    vals.append(cls_f1s[ci])
        cls_means.append(float(np.mean(vals)) if vals else 0.0)
    worst_idx = int(np.argmin(cls_means))
    per_class_summary[m] = {'per_class_mean': cls_means,
                            'worst_class': CLASS_NAMES[worst_idx],
                            'worst_class_f1': cls_means[worst_idx]}
    row = " ".join(f"{v:>8.4f}" for v in cls_means)
    print(f"{m:14s} {row}  {cls_means[worst_idx]:>9.4f}  {CLASS_NAMES[worst_idx]:>10s}")

# ── Save ───────────────────────────────────────────────────────────
output = {
  'main': {m: {
      'macro_f1_mean': float(np.mean(d['macro_f1'])) if d['macro_f1'] else 0.0,
      'macro_f1_std':  float(np.std(d['macro_f1'])) if d['macro_f1'] else 0.0,
      'worst_f1_mean': float(np.mean(d['worst_f1'])) if d['worst_f1'] else 0.0,
      'worst_f1_std':  float(np.std(d['worst_f1'])) if d['worst_f1'] else 0.0,
      'f1_gap_mean':   float(np.mean(d['f1_gap'])) if d['f1_gap'] else 0.0,
      'jain_mean':     float(np.mean(d['jain'])) if d['jain'] else 0.0,
      'jain_std':      float(np.std(d['jain'])) if d['jain'] else 0.0,
      'gini_mean':     float(np.mean(d['gini'])) if d['gini'] else 0.0,
      'bottom_q_mean': float(np.mean(d['bottom_q'])) if d['bottom_q'] else 0.0,
      'bottom_q_std':  float(np.std(d['bottom_q'])) if d['bottom_q'] else 0.0,
      'seeds_macro_f1': d['macro_f1'],
      'seeds_worst_f1': d['worst_f1'],
    } for m, d in summary.items()
  },
  'ablation': {a: {
      'macro_f1_mean': float(np.mean(d['macro_f1'])) if d['macro_f1'] else 0.0,
      'macro_f1_std':  float(np.std(d['macro_f1'])) if d['macro_f1'] else 0.0,
      'worst_f1_mean': float(np.mean(d['worst_f1'])) if d['worst_f1'] else 0.0,
      'worst_f1_std':  float(np.std(d['worst_f1'])) if d['worst_f1'] else 0.0,
      'f1_gap_mean':   float(np.mean(d['f1_gap'])) if d['f1_gap'] else 0.0,
    } for a, d in ablation_summary.items()
  },
  'sensitivity': sens_summary,
  'per_class':   per_class_summary,
}
# paired t-test vs FedAvg
fa = summary['FedAvg']['macro_f1']
ttests = {}
for m,d in summary.items():
    if m == 'FedAvg' or len(d['macro_f1']) != 3: continue
    t, p = stats.ttest_rel(d['macro_f1'], fa)
    ttests[m] = {'t': float(t), 'p': float(p)}
output['ttest_vs_fedavg'] = ttests

with open('paper_metrics.json','w') as f:
    json.dump(output, f, indent=2)
print("\n✅ paper_metrics.json yazıldı.")
