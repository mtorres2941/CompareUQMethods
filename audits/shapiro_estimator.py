"""How much does switching the uniform column to Shapiro-Francia move?

THE DEFECT. `customstats.shapiro_wilk_weighted` returned the true Shapiro-Wilk
W when the weights were uniform and a Shapiro-Francia W' when they were not.
So `fit_norm_SW` and `fit_norm_SW_uw` were two DIFFERENT statistics, and four
panels of the main characteristic figure -- the normal and lognormal fits under
each weighting -- compared them as though they were one. The same held for
`fit_lognorm_SW`.

THE DECISION, from the author: use Shapiro-Francia for BOTH columns. It is the
only one of the two with a weighted form, so it is the only choice under which
the uniform-versus-variable comparison is a comparison of one statistic under
two weightings, which is what those panels claim to show. The columns are
renamed `fit_norm_SF` and `fit_lognorm_SF` so that the name says which
statistic it holds. Decision 125.

WHAT THIS SCRIPT MEASURES, three things:

1. How far the uniform columns move on the real empirical arm and on the
   synthetic corpus, per dataset and per size stratum. Only the `_uw` columns
   can move; the variable-weighted ones were already Shapiro-Francia.
2. Whether the docstring's equivalence claim -- "indistinguishable for
   n >= 20" -- holds where this study needs it. It does not: the smallest
   stratum is n = 3 to 9.
3. What the two defects in `_royston_pvalue` were worth, against scipy's own
   p-value. Nothing reported depends on it, because only the statistic is
   kept, but a known-wrong p-value should not sit in a public deposit.

    conda run -n compareuq python audits/shapiro_estimator.py
"""
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, HERE)

import corpus  # noqa: E402
import customstats as CS  # noqa: E402
import empirical  # noqa: E402
from _common import write  # noqa: E402

STRATA = ((3, 9), (10, 99), (100, 999), (1000, 10 ** 9))


def stratum_of(n):
    for i, (lo, hi) in enumerate(STRATA, start=1):
        if lo <= n <= hi:
            return i
    return len(STRATA)


def per_dataset(name, values, n):
    """Both statistics on one dataset, under uniform weights only.

    The variable-weighted columns cannot move: they were already
    Shapiro-Francia. So the whole of the change is in the `_uw` pair.
    """
    x = np.asarray(values, float)
    lx = np.log(x)
    return dict(
        dataset=name, n=int(n), stratum=stratum_of(int(n)),
        norm_SW=CS.shapiro_wilk_scipy(x)[0],
        norm_SF=CS.shapiro_francia_weighted(x)[0],
        lognorm_SW=CS.shapiro_wilk_scipy(lx)[0],
        lognorm_SF=CS.shapiro_francia_weighted(lx)[0],
    )


def arm_table(rows, arm):
    df = pd.DataFrame(rows)
    df['arm'] = arm
    df['d_norm'] = df.norm_SF - df.norm_SW
    df['d_lognorm'] = df.lognorm_SF - df.lognorm_SW
    return df


def summarize(df):
    out = []
    for (arm, stratum), g in df.groupby(['arm', 'stratum']):
        row = dict(arm=arm, stratum=stratum, n_datasets=len(g),
                   n_min=int(g.n.min()), n_max=int(g.n.max()))
        for col, tag in (('d_norm', 'fit_norm_SF_uw'),
                         ('d_lognorm', 'fit_lognorm_SF_uw')):
            row[f'{tag}__mean_change'] = float(g[col].mean())
            row[f'{tag}__median_abs_change'] = float(g[col].abs().median())
            row[f'{tag}__max_abs_change'] = float(g[col].abs().max())
        out.append(row)
    for arm, g in df.groupby('arm'):
        row = dict(arm=arm, stratum='all', n_datasets=len(g),
                   n_min=int(g.n.min()), n_max=int(g.n.max()))
        for col, tag in (('d_norm', 'fit_norm_SF_uw'),
                         ('d_lognorm', 'fit_lognorm_SF_uw')):
            row[f'{tag}__mean_change'] = float(g[col].mean())
            row[f'{tag}__median_abs_change'] = float(g[col].abs().median())
            row[f'{tag}__max_abs_change'] = float(g[col].abs().max())
        out.append(row)
    return pd.DataFrame(out)


def royston_table(rng, reps=200):
    """What the two p-value defects were worth, against scipy's own p-value.

    `_royston_pvalue` is now correct, so this compares the CORRECTED function
    with scipy and reproduces, in the `old_*` columns, what the two defective
    branches returned. Both old forms are recomputed here rather than quoted,
    so the table stands on its own after the source is fixed.
    """
    rows = []
    for n in (4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 50, 100, 500, 2000, 5000):
        fixed, old, ref = [], [], []
        for i in range(reps):
            x = rng.normal(size=n) if i % 2 else rng.lognormal(size=n)
            res = stats.shapiro(x)
            W = float(res.statistic)
            ref.append(float(res.pvalue))
            fixed.append(CS._royston_pvalue(W, n))
            old.append(_old_royston(W, n))
        fixed, old, ref = np.array(fixed), np.array(old), np.array(ref)
        rows.append(dict(
            n=n,
            median_scipy=float(np.median(ref)),
            median_old=float(np.median(old)),
            median_fixed=float(np.median(fixed)),
            max_abs_error_old=float(np.max(np.abs(old - ref))),
            max_abs_error_fixed=float(np.max(np.abs(fixed - ref))),
        ))
    return pd.DataFrame(rows)


def _old_royston(W, n):
    """The pre-Stage-2f `_royston_pvalue`, verbatim, so the table can show what
    it returned after the source has been corrected."""
    W = float(W)
    n = int(n)
    if n == 3:
        p = (6.0 / np.pi) * (np.arcsin(np.sqrt(W)) - np.arcsin(np.sqrt(0.75)))
        return max(float(p), 1e-99)
    y = np.log(1.0 - W)
    u = np.log(n)
    mu = -1.5861 - 0.31082 * u - 0.083751 * u ** 2 + 0.0038915 * u ** 3
    lu = np.log(u)
    sigma = np.exp(-0.4803 - 0.082676 * lu + 0.0030302 * lu ** 2)
    if 4 <= n <= 11:
        gamma = 0.459 * n - 2.273
        z = (y - gamma - mu) / sigma
    else:
        z = (y - mu) / sigma
    return max(float(1.0 - stats.norm.cdf(z)), 1e-99)


def equivalence_table(rng, reps=400):
    """Is the "indistinguishable for n >= 20" claim true where it matters?"""
    rows = []
    for n in (3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 100, 500, 2000):
        d = []
        for _ in range(reps):
            x = rng.lognormal(sigma=0.6, size=n)
            d.append(CS.shapiro_francia_weighted(x)[0] - CS.shapiro_wilk_scipy(x)[0])
        d = np.array(d)
        rows.append(dict(n=n, median_abs_diff=float(np.median(np.abs(d))),
                         mean_diff=float(d.mean()),
                         p95_abs_diff=float(np.percentile(np.abs(d), 95)),
                         max_abs_diff=float(np.max(np.abs(d)))))
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(20260919)

    print('empirical arm ...')
    # The weights are irrelevant here -- only the UNIFORM column can move --
    # so the stream this consumes reaches nothing reported.
    emp, _report = empirical.prepare(rng.spawn(1)[0])
    erows = [per_dataset(k, x, len(x)) for k, (x, _w) in emp.items()]

    print('synthetic corpus ...')
    _metrics, values, _meta = corpus.load_corpus()
    srows = [per_dataset(str(dsid), g.value.to_numpy(), len(g))
             for dsid, g in values.groupby('dataset_id', observed=True)]

    df = pd.concat([arm_table(erows, 'empirical'), arm_table(srows, 'synthetic')],
                   ignore_index=True)
    write(df, 'AUDIT_ShapiroEstimatorPerDataset.csv')
    summary = summarize(df)
    write(summary, 'AUDIT_ShapiroEstimatorByStratum.csv')
    print(summary.to_string(index=False))

    eq = equivalence_table(rng)
    write(eq, 'AUDIT_ShapiroEquivalence.csv')
    print()
    print(eq.to_string(index=False))

    roy = royston_table(rng)
    write(roy, 'AUDIT_RoystonPValue.csv')
    print()
    print(roy.to_string(index=False))


if __name__ == '__main__':
    main()
