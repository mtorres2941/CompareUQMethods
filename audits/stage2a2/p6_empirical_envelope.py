"""The empirical envelope on the new data, for setting genconfig from measurement.

Every range in src/genconfig.py names an empirical measurement. Those
measurements were taken on the 2026-03 arm, so a new extract invalidates them
whether or not anything else changes. This recomputes them, and reports the old
value beside the new so a moved target is visible rather than quietly adopted.

    conda run -n compareuq python p6_empirical_envelope.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import empirical  # noqa: E402
import genconfig as G  # noqa: E402
import modality as MD  # noqa: E402
from customstats import empirical_metadata  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'stage2a2')
SEED = 42

#: The values the Stage 2a configuration was set from, for side-by-side reading.
OLD = dict(coeffvar_median=0.600, coeffvar_log10_sd=0.2913,
           coeffvar_min=0.0066, coeffvar_max=2.40,
           skewness_median=1.055, skewness_min=-1.44, skewness_max=4.618,
           n_median=37, unimodal_share=0.819)


def main():
    ds, _ = empirical.prepare(np.random.default_rng(SEED).spawn(1)[0])
    met = pd.DataFrame({m: empirical_metadata(x, w)
                        for m, (x, w) in ds.items()}).T.astype(float)
    met.index.name = 'material'
    met.to_csv(os.path.join(TABLES, 'TABLE_2a2_EmpiricalEnvelope.csv'))

    rng = np.random.default_rng(0)
    modes = np.array([MD.n_modes_silverman(x, rng=rng, nboot=100)
                      for x, _ in ds.values()])

    cv = met["coeffvar"].replace([np.inf, -np.inf], np.nan).dropna()
    cv = cv[cv > 0]
    lcv = np.log10(cv)
    print(f'{len(met)} empirical datasets\n')

    print('=== coefficient of variation ===')
    print(f'  median      {cv.median():.4f}   (Stage 2a arm {OLD["coeffvar_median"]})')
    print(f'  log10 sd    {lcv.std():.4f}   (Stage 2a arm {OLD["coeffvar_log10_sd"]})')
    print(f'  log10 mean  {lcv.mean():.4f}   median log10 {lcv.median():.4f}')
    print(f'  min / max   {cv.min():.4f} / {cv.max():.4f}   '
          f'(Stage 2a arm {OLD["coeffvar_min"]} / {OLD["coeffvar_max"]})')
    print(f'  genconfig range now brackets it: '
          f'{10**G.DEFAULT.cv_log10_lo:.4f} to {10**G.DEFAULT.cv_log10_hi:.4f}  '
          f'-> {"YES" if 10**G.DEFAULT.cv_log10_lo < cv.min() and 10**G.DEFAULT.cv_log10_hi > cv.max() else "NO"}')

    print('\n=== skewness ===')
    sk = met["skewness"].replace([np.inf, -np.inf], np.nan).dropna()
    print(f'  median      {sk.median():.4f}   (Stage 2a arm {OLD["skewness_median"]})')
    print(f'  min / max   {sk.min():.4f} / {sk.max():.4f}   '
          f'(Stage 2a arm {OLD["skewness_min"]} / {OLD["skewness_max"]})')
    print(f'  p05 / p95   {sk.quantile(.05):.4f} / {sk.quantile(.95):.4f}')

    print('\n=== kurtosis (excess) ===')
    ku = met["kurtosis"].replace([np.inf, -np.inf], np.nan).dropna()
    print(f'  median {ku.median():.3f}   min {ku.min():.3f}   max {ku.max():.3f}   '
          f'p95 {ku.quantile(.95):.3f}')

    print('\n=== dataset size ===')
    n = met["n"]
    print(f'  median {n.median():.0f}   (Stage 2a arm {OLD["n_median"]})   '
          f'min {n.min():.0f}   max {n.max():.0f}')
    share = {}
    for st in G.DEFAULT.strata:
        share[st.name] = float(((n >= st.n_lo) & (n <= st.n_hi)).mean())
    above = float((n > G.DEFAULT.strata[-1].n_hi).mean())
    print('  share per stratum (EMPIRICAL_STRATUM_SHARE in genconfig):')
    for k, v in share.items():
        old = G.EMPIRICAL_STRATUM_SHARE[k]
        print(f'    {k:<14} {v:.4f}   (was {old:.4f})')
    print(f'    above 9999     {above:.4f}')

    print('\n=== modality, Silverman ===')
    for k in range(1, 7):
        print(f'  {k} mode(s)   {(modes == k).mean()*100:5.1f}%')
    print(f'  unimodal    {(modes == 1).mean()*100:5.1f}%   '
          f'(Stage 2a arm {OLD["unimodal_share"]*100:.1f}%)')
    print(f'  crit_bw_1   median {met["crit_bw_1"].median():.4f}   '
          f'min {met["crit_bw_1"].min():.4f}   max {met.crit_bw_1.max():.4f}')

    print('\n=== other characteristics, median / min / max ===')
    for c in ['entropy', 'weight_outliers', 'fit_norm_SW', 'fit_lognorm_SW',
              'w_v_uw_wasserstein', 'modality_index']:
        s = met[c].replace([np.inf, -np.inf], np.nan).dropna()
        print(f'  {c:<20} {s.median():8.4f}  {s.min():8.4f}  {s.max():8.4f}')

    pd.DataFrame(dict(material=met.index, n_modes=modes)).to_csv(
        os.path.join(TABLES, 'TABLE_2a2_EmpiricalModality.csv'), index=False)


if __name__ == '__main__':
    main()
