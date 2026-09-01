# analysis/latex_tables.py — Generate LaTeX tables for paper

import os
import csv
import json


def generate_main_results_table(results_dict, output_dir):
    """Table 1: Main comparison results across 7 methods."""
    methods = ['FedAvg', 'FedProx', 'qFFL', 'Ditto', 'pFedMe',
               'FairPFL_noDP', 'FairPFL_DP']

    header = ['Method', 'Accuracy', 'Macro-F1', 'Min F1', 'DP↓', 'EO↓', 'Gini↓']

    rows = []
    for method in methods:
        if method in results_dict:
            r = results_dict[method]
            rows.append([
                method,
                f"{r.get('accuracy', 0) * 100:.1f}",
                f"{r.get('macro_f1', 0) * 100:.1f}",
                f"{r.get('min_cluster_f1', 0) * 100:.1f}",
                f"{r.get('demographic_parity', 0):.4f}",
                f"{r.get('equalized_odds', 0):.4f}",
                f"{r.get('gini_f1', 0):.4f}",
            ])

    # Save CSV
    csv_path = os.path.join(output_dir, 'table1_main_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    # Generate LaTeX
    latex = _to_latex(header, rows, "Main Comparison Results",
                      "tab:main_results")

    tex_path = os.path.join(output_dir, 'table1_main_results.tex')
    with open(tex_path, 'w') as f:
        f.write(latex)

    print(f"Saved: {csv_path}, {tex_path}")


def generate_ablation_table(results_dict, output_dir):
    """Table 2: Ablation study S0-S5."""
    scenarios = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']
    labels = {
        'S0': 'Baseline (no personal.)',
        'S1': '+ Personalization',
        'S2': '+ Fairness only',
        'S3': '+ Fair + Personal',
        'S4': 'No augmentation',
        'S5': '+ Differential Privacy',
    }

    header = ['Config', 'Description', 'Acc.', 'F1', 'Min F1', 'DP↓']
    rows = []
    for s in scenarios:
        if s in results_dict:
            r = results_dict[s]
            rows.append([
                s, labels.get(s, s),
                f"{r.get('accuracy', 0) * 100:.1f}",
                f"{r.get('macro_f1', 0) * 100:.1f}",
                f"{r.get('min_cluster_f1', 0) * 100:.1f}",
                f"{r.get('demographic_parity', 0):.4f}",
            ])

    csv_path = os.path.join(output_dir, 'table2_ablation.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Saved: {csv_path}")


def _to_latex(header, rows, caption, label):
    """Convert to LaTeX tabular."""
    cols = 'l' + 'c' * (len(header) - 1)
    lines = [
        f"\\begin{{table}}[t]",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\centering",
        f"\\begin{{tabular}}{{{cols}}}",
        f"\\toprule",
        " & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\",
        f"\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend([
        f"\\bottomrule",
        f"\\end{{tabular}}",
        f"\\end{{table}}",
    ])
    return "\n".join(lines)
