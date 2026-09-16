"""Why does the KDE not win on the empirical arm? Three mechanisms, measured.

The headline finding this examines is that the
KDE is not the best-fitting method on the 149 empirical datasets. That is a
statement about numbers; this script is the diagnosis, and the diagnosis matters
much more than the finding, because two of the three mechanisms are defects in
how the comparison is SPECIFIED rather than facts about kernel density
estimation.

  1. DATASET SIZE. The KDE's rank improves monotonically with n on BOTH arms,
     and the lognormal's degrades. The two arms do not actually disagree: the
     synthetic corpus allocates datasets equally across size strata, which
     over-represents large datasets about 3.4-fold relative to the empirical
     size mix, and large datasets are exactly where the KDE wins. Reweighting
     the corpus to the empirical mix flips its winner.

  2. THE BANDWIDTH RULE. `BW_METHOD = 'scott'`, 1.06 * sd * n_eff ** -0.2,
     oversmooths right-skewed data. Silverman's robust rule,
     0.9 * min(sd, IQR/1.34) * n_eff ** -0.2, beats it on essentially every
     empirical dataset. THE AUTHOR'S OWN KL2 PAPER USES SILVERMAN AND JUSTIFIES
     IT; this study uses Scott. CLAUDE.md decision 9 and discrepancy entry 10.

  3. THE CRITERION. W1 falls monotonically as the bandwidth shrinks, all the way
     to a few percent of any standard rule, because a KDE with a vanishing
     bandwidth converges on the empirical distribution it is being scored
     against. So mechanism 2 is CONFOUNDED: W1 prefers Silverman partly because
     Silverman is smaller, not because it is a better density estimate. This is
     the sharpest form of "W1 is an in-sample criterion with no complexity
     penalty" and it makes an earlier revision's out-of-sample comparison decisive rather
     than optional.

NOTHING HERE IS ADOPTED. The bandwidth rule belongs to an earlier revision and the
criterion to an earlier revision. This script measures; it changes no default.

    conda run -n compareuq python audits/why_kde_loses.py
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

import empirical  # noqa: E402
import families as F  # noqa: E402
import fitting as FT  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
OUTPUTS = os.path.join(ROOT, 'outputs', 'tables')
SEED = 42
PEWT = list(FT.PEWT)
BANDS = [0, 9, 99, 999, 10 ** 9]
BAND_LABELS = ['n 3-9', 'n 10-99', 'n 100-999', 'n >=1000']
#: Multiples of Scott's bandwidth, for the criterion check.
BW_MULTIPLES = (1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.10, 0.05, 0.02, 0.01)


def banded(df):
    return df.assign(band=pd.cut(df.n, bins=BANDS, labels=BAND_LABELS))


def mechanism_1():
    """Dataset size, on both arms, and the post-stratified comparison."""
    e = banded(pd.read_excel(
        os.path.join(OUTPUTS, 'TABLE_EmpiricalECCMetricsAndW1.xlsx'),
        index_col=0))
    s = banded(pd.read_excel(
        os.path.join(OUTPUTS, 'TABLE_SyntheticECCMetricsAndW1.xlsx'),
        index_col=0))

    print('=' * 78)
    print('1. DATASET SIZE')
    print('=' * 78)
    out = []
    for arm, d in (('empirical', e), ('synthetic', s)):
        r = d[PEWT].rank(axis=1).assign(band=d.band)
        t = r.groupby('band', observed=True).mean()
        t.insert(0, 'datasets', d.groupby('band', observed=True).size())
        print(f'--- {arm}: mean RANK of each method, by dataset size ---')
        print(t.to_string(float_format=lambda v: f'{v:.2f}'))
        print()
        out.append(t.assign(arm=arm).reset_index())
    pd.concat(out).to_csv(os.path.join(TABLES, 'TABLE_RankBySize.csv'),
                          index=False)

    share = e.groupby('band', observed=True).size() / len(e)
    per = s[PEWT].rank(axis=1).assign(band=s.band).groupby(
        'band', observed=True).mean()
    post = (per.T * share).T.sum() / share.sum()
    comp = pd.DataFrame({
        'synthetic, equal allocation': s[PEWT].rank(axis=1).mean(),
        'synthetic, reweighted to the empirical size mix': post,
        'empirical, as observed': e[PEWT].rank(axis=1).mean(),
    }).sort_values('empirical, as observed')
    comp.to_csv(os.path.join(TABLES, 'TABLE_PostStratifiedRank.csv'))
    print('empirical size mix, percent: '
          + ', '.join(f'{k} {v*100:.1f}' for k, v in share.items()))
    print('synthetic size mix is 25 percent in each band, by design.')
    print()
    print('MEAN RANK over the six methods:')
    print(comp.to_string(float_format=lambda v: f'{v:.3f}'))
    print()
    print('THE TWO ARMS DO NOT DISAGREE. Reweighting the corpus to the real')
    print('size mix moves KDE, Variable from 2.12 to 2.25 and Lognormal,')
    print('Variable from 2.19 to 2.02, which is the empirical ordering. The')
    print('corpus\'s equal allocation across strata is a design choice for')
    print('precision, not a claim about how common each size is.')


def mechanism_2_and_3(ds):
    print()
    print('=' * 78)
    print('2. THE BANDWIDTH RULE')
    print('=' * 78)
    rows = []
    for name, (x, w) in ds.items():
        r = dict(dataset=name, n=len(x))
        for wt, ww in (('Uniform', FT.uniform_weights(x)), ('Variable', w)):
            for bw in ('scott', 'silverman'):
                m, p = FT.fit_kde(x, ww, bw_method=bw)
                r[f'kde_{bw}_{wt}'] = FT.score_w1_model(m, x, w)
                r[f'h_over_sd_{bw}_{wt}'] = p['bandwidth'] / FT.weighted_std(x, ww)
                r[f'mass_below_{bw}_{wt}'] = m.mass_below
            m3, _ = FT.fit_family('lognormal_3p', x, ww, 'mle')
            r[f'lognormal_{wt}'] = FT.score_w1_model(m3, x, w)
        rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TABLES, 'TABLE_BandwidthDiagnostic.csv'),
             index=False)
    for wt in ('Uniform', 'Variable'):
        sc, si, lg = f'kde_scott_{wt}', f'kde_silverman_{wt}', f'lognormal_{wt}'
        print(f'--- {wt} weighting, 149 empirical datasets ---')
        print(f'  mean W1     Scott {d[sc].mean():.4f}   '
              f'Silverman {d[si].mean():.4f}   lognormal {d[lg].mean():.4f}')
        print(f'  median W1   Scott {d[sc].median():.4f}   '
              f'Silverman {d[si].median():.4f}   lognormal {d[lg].median():.4f}')
        print(f'  Silverman beats Scott on {(d[si] < d[sc]).mean()*100:.1f} pct')
        print(f'  beats the lognormal: Silverman {(d[si] < d[lg]).mean()*100:.1f} pct, '
              f'Scott {(d[sc] < d[lg]).mean()*100:.1f} pct')
        print(f'  median bandwidth / sd   Scott {d[f"h_over_sd_scott_{wt}"].median():.3f}   '
              f'Silverman {d[f"h_over_sd_silverman_{wt}"].median():.3f}')
        print(f'  mean model mass below zero   Scott '
              f'{d[f"mass_below_scott_{wt}"].mean():.4f}   Silverman '
              f'{d[f"mass_below_silverman_{wt}"].mean():.4f}')
        print()
    print('A Gaussian KDE inflates the fitted variance by sqrt(1 + (h/sd)^2).')
    print('At Scott\'s median h/sd of 0.56 that is 15 percent of extra spread')
    print('the data does not have, and it is what pushes 9.4 percent of the')
    print('model\'s mass below zero on an arm whose support is (0, inf).')

    print()
    print('=' * 78)
    print('3. THE CRITERION REWARDS UNDERSMOOTHING, WHICH CONFOUNDS 2')
    print('=' * 78)
    rows = []
    for name, (x, w) in ds.items():
        base = FT.weighted_bw(x, w, bw_method='scott')
        r = {'dataset': name, 'n': len(x)}
        for mlt in BW_MULTIPLES:
            mod = F.Truncated(F.WeightedKDE(x, w, base * mlt), label='kde')
            r[f'x{mlt}'] = FT.score_w1_model(mod, x, w)
        rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TABLES, 'TABLE_BandwidthSweepW1.csv'), index=False)
    cols = [f'x{m}' for m in BW_MULTIPLES]
    print('W1 as the bandwidth shrinks, as a multiple of Scott, variable weights:')
    print(pd.DataFrame({'mean W1': d[cols].mean(),
                        'median W1': d[cols].median()}
                       ).to_string(float_format=lambda v: f'{v:.5f}'))
    print()
    print('bandwidth multiple that MINIMIZES W1, per dataset:')
    print(d[cols].idxmin(axis=1).value_counts().to_string())
    print()
    print('W1 keeps falling to roughly 2 percent of Scott\'s bandwidth, where a')
    print('KDE is almost the empirical distribution it is being scored against.')
    print('It turns back up only at 1 percent, and that is the 1,000-point')
    print('scoring grid running out of resolution, not a real optimum.')
    print()
    print('SO: W1 cannot arbitrate between methods of different flexibility, and')
    print('the Scott-against-Silverman comparison above is confounded by it.')
    print('What survives the confound is the DIRECTION: Scott oversmooths this')
    print('data badly, and the KDE still loses to the lognormal under Scott even')
    print('though the criterion is biased in the KDE\'s favour. an earlier revision owns the')
    print('out-of-sample answer; an earlier revision owns the bandwidth.')


def main():
    os.makedirs(TABLES, exist_ok=True)
    mechanism_1()
    ds, _ = empirical.prepare(np.random.default_rng(SEED).spawn(1)[0])
    mechanism_2_and_3(ds)


if __name__ == '__main__':
    main()
