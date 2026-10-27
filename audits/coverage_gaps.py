"""Do the two coverage gaps of Figure 1 threaten the recommendation?

Fifteen of the 882 empirical points in the coverage figure fall outside the
synthetic range. Three are the three largest ready-mix classes, which hold more
EPDs than any synthetic dataset (14,366 to 31,025 against 9,978). Eleven belong
to the six most dispersed empirical categories (coefficient of variation ranks
1 to 6, all with 118 to 398 EPDs). The corpus cannot score a method on data it
does not contain, so the question is whether the trend at the corpus's own edge
points toward a different recommendation beyond it.

Both questions are asked of the per-dataset W1 against the market-weighted true
distribution (`w1_market` in TABLE_MethodScores.csv), with uniform weights,
because that is what the size rule a practitioner can follow uses. Nothing is
refitted and no randomness reaches a result except the bootstrap on the medians,
which has its own literal-seeded Generator.

Replaces decisions 137 and 138, which asked the same questions on a corpus the
study no longer uses.

    python audits/coverage_gaps.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import corpus                      # noqa: E402
import empirical as EMP            # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
KDE, LOGN = 'KDE, Uniform', 'Lognormal, Uniform'
N_BOOT = 2000


def per_dataset():
    """One row per synthetic dataset: n, CV, the two uniform W1s, the closest method."""
    d = pd.read_csv(os.path.join(ROOT, 'outputs', 'tables', 'TABLE_MethodScores.csv'))
    d = d[d.arm == 'synthetic']
    m = pd.read_parquet(os.path.join(corpus.active_dir(), 'metrics.parquet'))
    m = m[~m.is_probe][['dataset', 'coeffvar']]
    d = d.merge(m, on='dataset')
    w = (d.pivot_table(index=['dataset', 'n', 'coeffvar'], columns='method',
                       values='w1_market').reset_index())
    best = d.loc[d.groupby('dataset').w1_market.idxmin()].set_index('dataset').method
    w['closest'] = w.dataset.map(best)
    w['ratio'] = w[KDE] / w[LOGN]
    return w


def summarize(g, rng, **labels):
    r = np.log(g.ratio.to_numpy())
    boot = [np.median(rng.choice(r, len(r))) for _ in range(N_BOOT)]
    return dict(**labels, n_datasets=len(g),
                kde_beats_lognormal_pct=100 * float((g.ratio < 1).mean()),
                median_ratio=float(np.exp(np.median(r))),
                median_ratio_lo=float(np.exp(np.percentile(boot, 2.5))),
                median_ratio_hi=float(np.exp(np.percentile(boot, 97.5))),
                closest_kde_pct=100 * float(g.closest.str.startswith('KDE').mean()),
                closest_lognormal_pct=100 * float(g.closest.str.startswith('Logn').mean()),
                closest_normal_pct=100 * float(g.closest.str.startswith('Normal').mean()))


def main():
    w = per_dataset()
    rng = np.random.default_rng(20261026)
    rows = []
    for lo, hi in [(100, 999), (1000, 1999), (2000, 3999), (4000, 9999)]:
        rows.append(summarize(w[(w.n >= lo) & (w.n <= hi)], rng, check='size',
                              group=f'n {lo}-{hi}'))
    band = w[(w.n >= 100) & (w.n <= 999)]
    for lo, hi, lab in [(0, .5, 'bottom half'), (.5, .9, '50th-90th pct'),
                        (.9, .98, '90th-98th pct'), (.98, 1, 'top 2 pct')]:
        a, z = band.coeffvar.quantile([lo, hi]).to_numpy()
        rows.append(summarize(band[(band.coeffvar >= a) & (band.coeffvar <= z)],
                              rng, check='dispersion, n 100-999',
                              group=f'CV {lab} ({a:.2f} to {z:.2f})'))
    out = pd.DataFrame(rows)
    out['corpus'] = os.path.basename(corpus.active_dir()).replace('corpus_', '')
    out['weight_rho'] = float(EMP.WEIGHT_RHO)
    os.makedirs(TABLES, exist_ok=True)
    path = os.path.join(TABLES, 'TABLE_CoverageGaps.csv')
    out.to_csv(path, index=False)
    pd.set_option('display.width', 220)
    print(out.drop(columns=['corpus', 'weight_rho']).round(3).to_string(index=False))
    print(f'wrote {os.path.relpath(path, ROOT)}')


if __name__ == '__main__':
    main()
