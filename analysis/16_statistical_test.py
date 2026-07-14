"""Statistical significance test: is the growth graph different from Simple CNN?

Uses the controlled baseline data (3 seeds each) and performs:
1. Two-sample t-test
2. Effect size (Cohen's d)
3. 95% confidence intervals
"""
from __future__ import annotations
import sys, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy import stats


def main():
    # Load the controlled baseline data
    with open("results/analysis/01_controlled_baseline.json") as f:
        data = json.load(f)

    simple = data["simple_cnn"]["accs"]
    growth = data["growth_graph"]["accs"]

    print("=== Statistical Analysis ===")
    print(f"Simple CNN:   {simple} (n={len(simple)})")
    print(f"Growth Graph: {growth} (n={len(growth)})")
    print()

    # Descriptive stats
    s_mean, s_std = np.mean(simple), np.std(simple, ddof=1)
    g_mean, g_std = np.mean(growth), np.std(growth, ddof=1)
    print(f"Simple CNN:   mean={s_mean:.4f}, std={s_std:.4f}, sem={s_std/np.sqrt(len(simple)):.4f}")
    print(f"Growth Graph: mean={g_mean:.4f}, std={g_std:.4f}, sem={g_std/np.sqrt(len(growth)):.4f}")

    # 95% CI
    s_ci = stats.t.interval(0.95, len(simple)-1, loc=s_mean, scale=stats.sem(simple))
    g_ci = stats.t.interval(0.95, len(growth)-1, loc=g_mean, scale=stats.sem(growth))
    print(f"Simple CNN   95% CI: [{s_ci[0]:.4f}, {s_ci[1]:.4f}]")
    print(f"Growth Graph 95% CI: [{g_ci[0]:.4f}, {g_ci[1]:.4f}]")

    # Two-sample t-test (Welch's — unequal variance)
    t_stat, p_value = stats.ttest_ind(simple, growth, equal_var=False)
    print(f"\nWelch's t-test: t={t_stat:.4f}, p={p_value:.4f}")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((s_std**2 + g_std**2) / 2)
    cohens_d = (s_mean - g_mean) / pooled_std
    print(f"Cohen's d: {cohens_d:.4f} ({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'})")

    # Interpretation
    print(f"\n=== Interpretation ===")
    diff = s_mean - g_mean
    print(f"Difference: {diff:+.4f} (Simple CNN is {'better' if diff > 0 else 'worse'})")
    if p_value < 0.05:
        print(f"The difference is statistically significant (p < 0.05).")
    else:
        print(f"The difference is NOT statistically significant (p >= 0.05).")
        print(f"With only {len(simple)} seeds, we lack power to detect a {abs(diff)*100:.1f}% difference.")
    print(f"\nTo achieve significance, we would need ~{max(10, int((s_std**2 + g_std**2) / (diff**2) * 8))} seeds (rough estimate).")


if __name__ == "__main__":
    main()
