"""Stage 2a, Part 6 item 5: the coverage figure.

Places the 138 empirical datasets inside the synthetic cloud in the most
important metric dimensions, with a coverage statistic per panel, and lists any
empirical region the synthetic corpus fails to cover.

Per the standing constraint, the figure is drawn from tables on disk, not from
in-memory state: it reads TABLE_2a_EmpiricalMetrics_final.csv and the corpus
metrics written by b1_coverage.py. Run b1_coverage.py first.

Writes outputs/figures/CompareUQMethods_FIG_MetricCoverage.png and .pdf
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from _common import ROOT, TABLES, write

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus as CORP

FIGDIR = os.path.join(ROOT, 'outputs', 'figures')
PAIRS = [('coeffvar', 'skewness'), ('coeffvar', 'crit_bw_1'),
         ('skewness', 'kurtosis'), ('n', 'w_v_uw_wasserstein'),
         ('entropy', 'weight_outliers'), ('fit_norm_SW', 'fit_lognorm_SW')]
LOGX = {'n'}
LABELS = {
    'coeffvar': 'Coefficient of variation', 'skewness': 'Skewness',
    'kurtosis': 'Excess kurtosis', 'crit_bw_1': 'Critical bandwidth (sd units)',
    'n': 'Dataset size n', 'w_v_uw_wasserstein': 'W1, uniform vs variable',
    'entropy': 'Entropy', 'weight_outliers': 'Weight of outliers',
    'fit_norm_SW': 'Normal fit (Shapiro-Wilk)',
    'fit_lognorm_SW': 'Lognormal fit (Shapiro-Wilk)',
}


def coverage_fraction(emp, syn, a, b, nbins=26):
    """Share of empirical datasets landing in a 2-D bin the synthetic corpus
    also occupies. A blunt statistic, but it is the one the figure shows."""
    e = emp[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
    s = syn[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(e) == 0 or len(s) == 0:
        return np.nan, e.index[:0]
    lo = np.minimum(e.min().to_numpy(), s.min().to_numpy())
    hi = np.maximum(e.max().to_numpy(), s.max().to_numpy())
    edges = [np.linspace(lo[i], hi[i], nbins + 1) for i in range(2)]
    Hs, _, _ = np.histogram2d(s[a], s[b], bins=edges)
    ie = np.clip(np.digitize(e[a], edges[0]) - 1, 0, nbins - 1)
    je = np.clip(np.digitize(e[b], edges[1]) - 1, 0, nbins - 1)
    inside = Hs[ie, je] > 0
    return float(inside.mean()), e.index[~inside]


if __name__ == '__main__':
    emp = pd.read_csv(os.path.join(TABLES, 'TABLE_2a_EmpiricalMetrics_final.csv')
                      ).set_index('material')
    metrics_all = pd.read_parquet(os.path.join(CORP.active_dir(), 'metrics.parquet'))
    syn = metrics_all[~metrics_all.is_probe]

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7),
                             gridspec_kw=dict(hspace=0.38, wspace=0.28))
    rows, uncovered = [], {}
    for ax, (a, b) in zip(axes.ravel(), PAIRS):
        frac, miss = coverage_fraction(emp, syn, a, b)
        rows.append(dict(metric_x=a, metric_y=b, empirical_covered=frac,
                         n_uncovered=len(miss),
                         uncovered='|'.join(sorted(miss))))
        uncovered[(a, b)] = list(miss)
        ax.scatter(syn[a], syn[b], s=2, alpha=0.08, color='tab:blue', linewidths=0,
                   rasterized=True, label='Synthetic (10,000)')
        ax.scatter(emp[a], emp[b], s=14, color='tab:orange', edgecolors='black',
                   linewidths=0.3, zorder=3, label='Empirical (138)')
        if len(miss):
            ax.scatter(emp.loc[miss, a], emp.loc[miss, b], s=46, facecolors='none',
                       edgecolors='crimson', linewidths=1.0, zorder=4,
                       label='Empirical, not covered')
        if a in LOGX:
            ax.set_xscale('log')
        ax.set_xlabel(LABELS.get(a, a), fontsize=8)
        ax.set_ylabel(LABELS.get(b, b), fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(f'{frac*100:.1f} pct of empirical datasets covered', fontsize=8)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axes[0, 0].legend(fontsize=7, loc='best', framealpha=0.9)
    fig.suptitle('Empirical ECC datasets inside the synthetic metric space', fontsize=11)
    os.makedirs(FIGDIR, exist_ok=True)
    for ext, dpi in (('png', 600), ('pdf', 600)):
        fig.savefig(os.path.join(FIGDIR, f'CompareUQMethods_FIG_MetricCoverage.{ext}'),
                    bbox_inches='tight', dpi=dpi)
    plt.close(fig)

    cov = pd.DataFrame(rows)
    write(cov, 'TABLE_2a_CoverageFigureStats.csv')
    pd.set_option('display.width', 250, 'display.max_colwidth', 80)
    print(cov[['metric_x', 'metric_y', 'empirical_covered', 'n_uncovered']].to_string(
        index=False, float_format=lambda v: f'{v:,.4f}'))
    print('\nempirical datasets not covered, by panel:')
    for k, v in uncovered.items():
        if v:
            print(f'  {k[0]} vs {k[1]}: {", ".join(sorted(v))}')
    print(f'\nwrote {FIGDIR}/CompareUQMethods_FIG_MetricCoverage.png and .pdf')
