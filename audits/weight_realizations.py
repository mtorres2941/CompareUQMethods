"""One weight draw or many: which claims depend on which? Stage 2h.

Stage 2a-3 found that redrawing the Dirichlet weights of the SAME datasets from
the SAME distribution, changing nothing else, moves `w_v_uw_wasserstein` by up
to 1.02 in absolute terms per dataset, excess kurtosis by 365.8 and the
coefficient of variation by 4.09, while every unweighted column stays
bit-identical. The arm-level DISTRIBUTION is far more stable and that is what
the study rests on -- but the manuscript does not currently distinguish a
per-dataset weighted statistic from the distribution of them.

THE TWO QUANTITIES, and every weighted claim in the paper is one or the other.

    PER DATASET   "this category's uniform-to-variable distance is 0.13".
                  One draw from a distribution. Moves when the seed moves.
    ARM LEVEL     "the median category's distance is 0.093", "the coefficient
                  of variation predicts it at Spearman +0.73". A property of
                  the DISTRIBUTION of draws, which a single realization
                  estimates with error that averages down over 147 categories.

This measures both: the per-dataset spread across independent realizations, and
the realization-to-realization spread of every arm-level statistic the paper
quotes. The second is what says whether a headline is safe.

    conda run -n compareuq python audits/weight_realizations.py [n_real]
"""
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import empirical                            # noqa: E402
import weighting as WG                      # noqa: E402
from customstats import empirical_metadata  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')

#: Independent weight realizations of the WHOLE ARM. Each one is a complete
#: alternative version of the empirical arm's market shares.
N_REALIZATIONS = 25

#: The weighted characteristics the paper reports. Their unweighted twins are
#: bit-identical across realizations by construction and are the control.
WEIGHTED = ('w_v_uw_wasserstein', 'coeffvar', 'kurtosis', 'skewness',
            'crit_bw_1', 'entropy', 'fit_norm_SF', 'fit_lognorm_SF',
            'weight_outliers')
UNWEIGHTED = ('coeffvar_uw', 'kurtosis_uw', 'skewness_uw')


def main(n_real=N_REALIZATIONS):
    os.makedirs(TABLES, exist_ok=True)
    pd.set_option('display.width', 220)
    ds, _ = empirical.prepare(np.random.default_rng(20260912))
    values = {m: np.asarray(x, float) for m, (x, _) in ds.items()}

    rows = []
    for r in range(n_real):
        rng = np.random.default_rng(90000 + r)
        for mat, x in values.items():
            w = rng.dirichlet(np.ones(len(x)))
            m = empirical_metadata(x, w)
            m.update(realization=r, dataset=str(mat), n=len(x))
            rows.append(m)
        print(f'  realization {r + 1}/{n_real}', flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TABLES, 'TABLE_WeightRealizations.csv.gz'),
             index=False)
    report(d, n_real)
    return d


def report(d, n_real):
    print()
    print('=' * 78)
    print('1. PER DATASET: how far one realization moves a category\'s own number')
    print('=' * 78)
    have = [c for c in WEIGHTED if c in d]
    per = d.groupby('dataset')[have].agg(['median', 'std', 'min', 'max'])
    out = []
    for c in have:
        s = per[c]
        out.append(dict(characteristic=c,
                        median_across_datasets=float(s['median'].median()),
                        typical_sd=float(s['std'].median()),
                        worst_sd=float(s['std'].max()),
                        worst_range=float((s['max'] - s['min']).max()),
                        typical_rel_sd=float(
                            (s['std'] / s['median'].abs()).median())))
    print(pd.DataFrame(out).to_string(index=False,
                                      float_format=lambda v: f'{v:.4f}'))
    print()
    print('THE CONTROL: the unweighted twins must not move at all, because the')
    print('weights do not enter them. A non-zero number here would mean the')
    print('measurement is picking up something other than the weight draw.')
    ctrl = [c for c in UNWEIGHTED if c in d]
    print(d.groupby('dataset')[ctrl].std().max()
          .to_string(float_format=lambda v: f'{v:.3g}'))

    print()
    print('=' * 78)
    print('2. ARM LEVEL: how far the STATISTICS THE PAPER QUOTES move')
    print('=' * 78)
    print(f'Each row is one arm-level statistic, recomputed on {n_real}')
    print('independent realizations of the whole arm. This is the number that')
    print('says whether a headline is safe, and it is not the one above.')
    print()
    stats = []
    for r, g in d.groupby('realization'):
        row = dict(realization=r)
        for c in have:
            row[f'{c}__median'] = float(g[c].median())
            row[f'{c}__mean'] = float(g[c].mean())
        # The paper's own practitioner law: separation against dispersion and
        # size (decision 96). Both its coefficients are arm-level statistics.
        ok = (g.w_v_uw_wasserstein > 0) & (g.coeffvar > 0) & (g.n > 0)
        gg = g[ok]
        if len(gg) > 20:
            X = np.column_stack([np.ones(len(gg)), np.log(gg.n),
                                 np.log(gg.coeffvar)])
            y = np.log(gg.w_v_uw_wasserstein)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            pred = X @ beta
            row['law_log_n_exponent'] = float(beta[1])
            row['law_log_cv_exponent'] = float(beta[2])
            row['law_r2'] = float(1 - np.var(y - pred) / np.var(y))
            row['spearman_cv'] = float(
                gg.w_v_uw_wasserstein.rank().corr(gg.coeffvar.rank()))
            row['spearman_logn'] = float(
                gg.w_v_uw_wasserstein.rank().corr(np.log(gg.n).rank()))
        stats.append(row)
    s = pd.DataFrame(stats)
    s.to_csv(os.path.join(TABLES, 'TABLE_WeightRealizationStats.csv'),
             index=False)
    cols = [c for c in s.columns if c != 'realization']
    summ = pd.DataFrame(dict(statistic=cols,
                             mean=[s[c].mean() for c in cols],
                             sd=[s[c].std() for c in cols],
                             lo=[s[c].min() for c in cols],
                             hi=[s[c].max() for c in cols]))
    summ['rel_sd'] = summ.sd / summ['mean'].abs()
    print(summ.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print()
    print('WHICH CLAIMS DEPEND ON WHICH. A claim about ONE category carries the')
    print('per-dataset spread of part 1 and should be stated as a distribution')
    print('rather than a number. A claim about the arm -- a median, a')
    print('correlation, the size-and-dispersion law -- carries the much smaller')
    print('spread of part 2 and is safe to quote as a number.')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_REALIZATIONS)
