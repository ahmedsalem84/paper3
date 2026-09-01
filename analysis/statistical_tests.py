# analysis/statistical_tests.py — Statistical significance for paper tables

from scipy import stats
import numpy as np


def paired_ttest(method1_scores, method2_scores, alpha=0.05):
    """Paired t-test between two methods across seeds (n=5)."""
    if len(method1_scores) < 2:
        return {'t_statistic': 0, 'p_value': 1.0,
                'significant': False, 'effect_size': 0}

    t_stat, p_value = stats.ttest_rel(method1_scores, method2_scores)
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': bool(p_value < alpha),
        'effect_size': float(cohen_d(method1_scores, method2_scores)),
    }


def cohen_d(group1, group2):
    """Effect size (Cohen's d)."""
    n1, n2 = len(group1), len(group2)
    var1 = np.var(group1, ddof=1) if n1 > 1 else 0
    var2 = np.var(group2, ddof=1) if n2 > 1 else 0
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / max(n1 + n2 - 2, 1))
    if pooled_std == 0:
        return 0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def wilcoxon_test(method1_scores, method2_scores, alpha=0.05):
    """Non-parametric alternative (for small n=5)."""
    if len(method1_scores) < 5:
        return {'statistic': 0, 'p_value': 1.0, 'significant': False}
    try:
        stat, p_value = stats.wilcoxon(method1_scores, method2_scores)
        return {'statistic': float(stat), 'p_value': float(p_value),
                'significant': bool(p_value < alpha)}
    except ValueError:
        return {'statistic': 0, 'p_value': 1.0, 'significant': False}
