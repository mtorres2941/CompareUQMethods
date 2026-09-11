"""Stage 2a: what the 138 empirical ECC datasets actually look like.

This is the target the generator has to cover, so it is measured before the
generator is redesigned rather than after. Weights are redrawn at alpha = 1
per the Part 0 item 2 decision.

Writes TABLE_2a_EmpiricalEnvelope.csv and TABLE_2a_EmpiricalMetrics_alpha1.csv
"""
import numpy as np, pandas as pd, sys, os
from _common import load_empirical, write

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from customstats import empirical_metadata

SEED = 20260911
ALPHA = 1

if __name__ == '__main__':
    d = load_empirical()
    rng = np.random.default_rng(SEED)
    rows, met = [], {}
    for mat in d:
        data = d[mat]['data']
        w = rng.dirichlet(np.ones_like(data) * ALPHA)
        met[mat] = empirical_metadata(data / np.mean(data), w)
        raw = data / np.mean(data)
        rows.append(dict(material=mat, n=len(data),
                         min_over_mean=float(raw.min()), max_over_mean=float(raw.max()),
                         ratio_max_min=float(raw.max() / raw.min())))
    dfm = pd.DataFrame(met).T.astype(float)
    dfm.index.name = 'material'
    dfm.reset_index().to_csv(
        '../../outputs/tables/stage2a/TABLE_2a_EmpiricalMetrics_alpha1.csv', index=False)
    print('wrote TABLE_2a_EmpiricalMetrics_alpha1.csv', dfm.shape)

    env = []
    for c in dfm.columns:
        s = dfm[c].replace([np.inf, -np.inf], np.nan).dropna()
        env.append(dict(metric=c, n_defined=int(len(s)), n_missing=int(len(dfm) - len(s)),
                        min=float(s.min()), p01=float(s.quantile(.01)),
                        p05=float(s.quantile(.05)), median=float(s.median()),
                        p95=float(s.quantile(.95)), p99=float(s.quantile(.99)),
                        max=float(s.max()), mean=float(s.mean()), sd=float(s.std())))
    env = pd.DataFrame(env)
    write(env, 'TABLE_2a_EmpiricalEnvelope.csv')
    pd.set_option('display.width', 250)
    print(env.to_string(index=False, float_format=lambda v: f'{v:,.4f}'))

    sz = pd.DataFrame(rows)
    print('\n--- dataset sizes ---')
    print(sz.n.describe(percentiles=[.05, .25, .5, .75, .95, .99]).to_string())
    strata = [(3, 9), (10, 99), (100, 999), (1000, 9999), (10000, 10**9)]
    print('\nstratum   count   share')
    for lo, hi in strata:
        c = int(((sz.n >= lo) & (sz.n <= hi)).sum())
        print(f'{lo:>6}-{hi if hi < 10**8 else "inf":<7} {c:>4}   {c/len(sz)*100:5.1f}%')
    print('\nlargest 8:', sorted(sz.n)[-8:])
    print('\n--- near-zero values, motivating the multiplicative cleaning bound ---')
    print(sz.sort_values('min_over_mean').head(12).to_string(index=False))
    print(f"\ndatasets with min < 1% of mean : {(sz.min_over_mean < 0.01).sum()} of {len(sz)}")
    print(f"datasets with min < 0.1% of mean: {(sz.min_over_mean < 0.001).sum()} of {len(sz)}")
