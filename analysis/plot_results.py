# analysis/plot_results.py — Generate all paper figures (300dpi PNG)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os

# Publication-ready style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'figure.figsize': (8, 5),
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def plot_convergence(results_dir, scenarios, output_path):
    """Figure 2: Convergence curves (loss vs round)."""
    fig, ax = plt.subplots()
    for scenario in scenarios:
        path = os.path.join(results_dir, scenario, 'seed_42', 'history.json')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            history = json.load(f)
        rounds = [h['round'] for h in history]
        losses = [h.get('train_loss', 0) for h in history]
        ax.plot(rounds, losses, label=scenario, linewidth=1.5)

    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Distributed Loss')
    ax.set_title('Convergence Comparison')
    ax.legend(fontsize=10)
    fig.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_cluster_f1(results, methods, output_path, num_clusters=9):
    """Figure 3: Per-cluster F1 grouped bar chart."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(num_clusters)
    width = 0.12

    for i, method in enumerate(methods):
        if method in results:
            f1_scores = [results[method].get(f'cluster_{k}_f1', 0)
                         for k in range(num_clusters)]
            ax.bar(x + i * width, f1_scores, width, label=method)

    ax.set_xlabel('Device-Type Cluster')
    ax.set_ylabel('F1-Score')
    ax.set_xticks(x + width * len(methods) / 2)
    ax.set_xticklabels([f'C{k}' for k in range(num_clusters)])
    ax.legend()
    ax.set_ylim(0.8, 1.0)
    fig.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_fairness_accuracy_tradeoff(results, output_path):
    """Figure 4: Fairness-accuracy scatter plot."""
    fig, ax = plt.subplots()

    for method, metrics in results.items():
        dp = metrics.get('demographic_parity', 0)
        acc = metrics.get('accuracy', 0) * 100
        ax.scatter(dp, acc, s=100, label=method, zorder=5)
        ax.annotate(method, (dp, acc), fontsize=9,
                    xytext=(5, 5), textcoords='offset points')

    ax.set_xlabel('Demographic Parity (lower = fairer)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Fairness-Accuracy Trade-off')
    ax.legend()
    fig.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_q_sensitivity(q_results, output_path):
    """Figure 5: q sensitivity dual-axis plot."""
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    q_vals = sorted(q_results.keys())
    accs = [q_results[q].get('accuracy', 0) * 100 for q in q_vals]
    dps = [q_results[q].get('demographic_parity', 0) for q in q_vals]

    ax1.plot(q_vals, accs, 'b-o', label='Accuracy', linewidth=2)
    ax2.plot(q_vals, dps, 'r-s', label='DP', linewidth=2)

    ax1.set_xlabel('Fairness Parameter q')
    ax1.set_ylabel('Accuracy (%)', color='b')
    ax2.set_ylabel('Demographic Parity', color='r')
    ax1.set_title('Effect of q on Performance')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    fig.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_privacy_pareto(eps_results, output_path):
    """Figure 6: Privacy-accuracy Pareto frontier."""
    fig, ax = plt.subplots()

    eps_vals = sorted(eps_results.keys())
    accs = [eps_results[e].get('accuracy', 0) * 100 for e in eps_vals]

    ax.plot(eps_vals, accs, 'g-o', linewidth=2, markersize=8)
    ax.fill_between(eps_vals, [a - 1 for a in accs], [a + 1 for a in accs],
                    alpha=0.2, color='g')

    ax.set_xlabel('Total Privacy Budget (ε)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Privacy-Accuracy Trade-off')
    fig.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")
