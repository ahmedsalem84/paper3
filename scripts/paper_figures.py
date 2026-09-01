#!/usr/bin/env python3
"""
paper_figures.py
================
Generates the four figures referenced in paper/FairPFL_full_paper.md:

  Figure 3  Per-cluster F1 boxplot (9 clusters x 3 seeds, one box per method)
  Figure 4  Convergence stability (per-round Macro-F1 with +/- 1 std band, 3 seeds)
  Figure 5  Per-class F1 heatmap (method x class)
  Figure 6  Macro-F1 vs F1-gap trade-off scatter

Outputs PNGs under paper/figures/. Inputs are the same round_metrics.json files
listed in results.md.
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# IEEE-ish style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': ':',
})

OUT = 'paper/figures'   # created in __main__, after the results check

CLASS_NAMES = ['Benign', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing']

RUNS = {
  'FairPFL': {42:'results_v20_test/FairPFL_noDP/seed_42/round_metrics.json',
              123:'results_v20_seed123/FairPFL_noDP/seed_123/round_metrics.json',
              456:'results_v20_seed456/FairPFL_noDP/seed_456/round_metrics.json'},
  'qFFL':    {s:f'results_baselines_6class/qFFL/seed_{s}/round_metrics.json' for s in (42,123,456)},
  'FedAvg':  {s:f'results_baselines_6class/FedAvg/seed_{s}/round_metrics.json' for s in (42,123,456)},
  'FedProx': {s:f'results_baselines_6class/FedProx/seed_{s}/round_metrics.json' for s in (42,123,456)},
  'Ditto':   {s:f'results_baselines_6class/Ditto/seed_{s}/round_metrics.json' for s in (42,123,456)},
  'pFedMe':  {s:f'results_baselines_6class/pFedMe/seed_{s}/round_metrics.json' for s in (42,123,456)},
  'FairPFL_DP':{s:f'results_v20_dp/FairPFL_DP/FairPFL_DP/seed_{s}/round_metrics.json' for s in (42,123,456)},
  'FedAvg_DP':{s:f'results_v20_dp/FedAvg_DP_real/FedAvg_DP/seed_{s}/round_metrics.json' for s in (42,123,456)},
}

# Stable, print-safe color palette (CB-friendly)
COLORS = {
  'FairPFL':    '#1f77b4',
  'qFFL':       '#ff7f0e',
  'FedAvg':     '#2ca02c',
  'FedProx':    '#9467bd',
  'Ditto':      '#8c564b',
  'pFedMe':     '#e377c2',
  'FairPFL_DP': '#17becf',
  'FedAvg_DP':  '#bcbd22',
}

def load_last(p):
    return json.load(open(p))[-1]

def load_history(p):
    return json.load(open(p))

# ─────────────────────────────────────────────────────────────────
# Figure 3. Per-cluster F1 boxplot
# ─────────────────────────────────────────────────────────────────
def figure3():
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    order = ['FairPFL', 'qFFL', 'FedAvg', 'FedProx', 'Ditto', 'pFedMe', 'FairPFL_DP', 'FedAvg_DP']
    data, labels, colors = [], [], []
    for m in order:
        vals = []
        for s, p in RUNS[m].items():
            if not os.path.exists(p): continue
            pc = load_last(p).get('per_cluster_f1', {})
            vals.extend(float(v) for v in pc.values())
        data.append(vals)
        labels.append(m.replace('_', '-'))
        colors.append(COLORS[m])

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                    medianprops=dict(color='black', linewidth=1.2),
                    flierprops=dict(marker='o', markersize=3, alpha=0.4))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.5); patch.set_edgecolor('black')
    ax.set_ylabel('Per-cluster Macro-F1')
    ax.set_title('Per-cluster F1 distribution across 9 device-type clusters (3 seeds)')
    ax.set_ylim(0.3, 1.0)
    ax.axhline(0.8675, color='#1f77b4', linestyle='--', alpha=0.5, linewidth=0.9,
               label='FairPFL mean (0.8675)')
    ax.axhline(0.8488, color='#2ca02c', linestyle='--', alpha=0.5, linewidth=0.9,
               label='FedAvg mean (0.8488)')
    ax.legend(loc='lower left', framealpha=0.9)
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig3_boxplot_per_cluster_f1.png')
    plt.close(fig)
    print('  wrote fig3_boxplot_per_cluster_f1.png')

# ─────────────────────────────────────────────────────────────────
# Figure 4. Convergence stability
# ─────────────────────────────────────────────────────────────────
def figure4():
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    methods = ['FairPFL', 'qFFL', 'FedAvg']
    for m in methods:
        all_curves = []
        for s, p in RUNS[m].items():
            if not os.path.exists(p): continue
            h = load_history(p)
            rounds = [r['round'] for r in h]
            f1s = [r.get('avg_f1', 0) for r in h]
            all_curves.append((rounds, f1s))
        if not all_curves: continue
        # align on the shortest length
        min_len = min(len(c[1]) for c in all_curves)
        rounds = all_curves[0][0][:min_len]
        arr = np.array([c[1][:min_len] for c in all_curves])
        mean = arr.mean(axis=0); std = arr.std(axis=0)
        ax.plot(rounds, mean, color=COLORS[m], label=m, linewidth=1.6)
        ax.fill_between(rounds, mean - std, mean + std, color=COLORS[m], alpha=0.18)
    ax.set_xlabel('Federation round')
    ax.set_ylabel('Macro-F1 (validation set)')
    ax.set_title('Convergence stability: per-round Macro-F1 with ±1 σ band over 3 seeds')
    ax.set_xlim(0, 200)
    ax.legend(loc='lower right', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig4_convergence_stability.png')
    plt.close(fig)
    print('  wrote fig4_convergence_stability.png')

# ─────────────────────────────────────────────────────────────────
# Figure 5. Per-class F1 heatmap
# ─────────────────────────────────────────────────────────────────
def figure5():
    methods = ['FairPFL', 'qFFL', 'FedAvg', 'FedProx', 'Ditto', 'pFedMe',
               'FairPFL_DP', 'FedAvg_DP']
    mat = np.zeros((len(methods), 6))
    for i, m in enumerate(methods):
        cls_means = [[] for _ in range(6)]
        for s, p in RUNS[m].items():
            if not os.path.exists(p): continue
            pcc = load_last(p).get('per_cluster_per_class_f1', {})
            for cid, cls_f1s in pcc.items():
                for ci, v in enumerate(cls_f1s[:6]):
                    cls_means[ci].append(v)
        mat[i] = [np.mean(vals) if vals else 0.0 for vals in cls_means]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=0.4, vmax=1.0)
    ax.set_xticks(np.arange(6)); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([m.replace('_', '-') for m in methods])
    ax.set_title('Per-class Macro-F1 (mean across 9 clusters × 3 seeds)')
    for i in range(len(methods)):
        for j in range(6):
            txt_color = 'black' if 0.6 < mat[i, j] < 0.92 else 'white'
            ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                    color=txt_color, fontsize=7.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Macro-F1')
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig5_per_class_heatmap.png')
    plt.close(fig)
    print('  wrote fig5_per_class_heatmap.png')

# ─────────────────────────────────────────────────────────────────
# Figure 6. Macro-F1 vs F1-gap trade-off scatter
# ─────────────────────────────────────────────────────────────────
def figure6():
    methods = ['FairPFL', 'qFFL', 'FedAvg', 'FedProx', 'Ditto', 'pFedMe',
               'FairPFL_DP', 'FedAvg_DP']
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for m in methods:
        f1s, gaps = [], []
        for s, p in RUNS[m].items():
            if not os.path.exists(p): continue
            r = load_last(p)
            pc = r.get('per_cluster_f1', {})
            f1s.append(r.get('avg_f1', 0))
            if pc:
                vals = [float(v) for v in pc.values()]
                gaps.append(max(vals) - min(vals))
        if not f1s: continue
        ax.scatter(np.mean(gaps), np.mean(f1s), s=140, color=COLORS[m],
                   edgecolor='black', linewidth=0.8, alpha=0.85, zorder=3)
        # error ellipse via std bars
        ax.errorbar(np.mean(gaps), np.mean(f1s),
                    xerr=np.std(gaps), yerr=np.std(f1s),
                    color=COLORS[m], alpha=0.5, capsize=2, zorder=2)
        # annotation
        ax.annotate(m.replace('_', '-'),
                    (np.mean(gaps), np.mean(f1s)),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8)
    ax.set_xlabel('Inter-cluster F1 gap (max − min)  ← fairer')
    ax.set_ylabel('Macro-F1  ↑ better')
    ax.set_title('Macro-F1 vs inter-cluster fairness (3-seed means with ±1 σ)')
    ax.set_xlim(0.04, 0.24)
    ax.set_ylim(0.65, 0.90)
    # ideal corner
    ax.annotate('ideal corner',
                (0.05, 0.88), fontsize=8, color='gray', style='italic')
    ax.plot([0.045], [0.885], marker='*', color='gold', markersize=12,
            markeredgecolor='black', zorder=4)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig6_tradeoff_f1_vs_gap.png')
    plt.close(fig)
    print('  wrote fig6_tradeoff_f1_vs_gap.png')

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


if __name__ == '__main__':
    _require_results(RUNS, 'the figures')
    os.makedirs(OUT, exist_ok=True)
    print('Generating figures to', OUT)
    figure3()
    figure4()
    figure5()
    figure6()
    print('done.')
