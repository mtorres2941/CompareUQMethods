"""Empirical component overlap on the new arm, for genconfig and Table 1.

The overlap range in src/genconfig.py cites a measurement of the empirical
datasets: a BIC-selected Gaussian mixture fitted to each one, with the
Maitra-Melnykov average pairwise overlap computed from the fit. That is a proxy,
since the empirical datasets have no known components, but it puts both arms on
the same footing. It was measured on the 2026-03 arm, so it is remeasured here.

    conda run -n compareuq python p7_empirical_overlap.py
"""
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import empirical  # noqa: E402
import mixture as M  # noqa: E402
import modality as MD  # noqa: E402
from scipy import stats  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'stage2a2')
SEED = 42
NMAX = 20_000       # cap per dataset; the BIC fit on 86,770 points is not worth it


def fitted_overlap(x, rng, kmax=5):
    x = np.asarray(x, float)
    if len(x) > NMAX:
        x = rng.choice(x, NMAX, replace=False)
    if len(x) < 6 or np.std(x) <= 0:
        return np.nan, 1
    k, pi, mu, sd = MD.fit_mixture_bic(x, rng, kmax=kmax)
    if k < 2:
        return 0.0, 1
    comps = [stats.norm(loc=m, scale=max(s, 1e-9)) for m, s in zip(mu, sd)]
    return M.average_overlap(comps, pi, grid_n=1501), k


if __name__ == '__main__':
    ds, _ = empirical.prepare(np.random.default_rng(SEED).spawn(1)[0])
    rng = np.random.default_rng(SEED)
    rows = []
    for mat, (x, _) in ds.items():
        ov, k = fitted_overlap(x, rng)
        rows.append(dict(material=mat, n=len(x), k_bic=k, overlap=ov))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TABLES, 'TABLE_2a2_EmpiricalOverlap.csv'), index=False)
    o = d.overlap.dropna()
    print(f'{len(d)} datasets, {o.notna().sum()} with an overlap')
    print(f'  BIC multimodal   {(d.k_bic > 1).mean()*100:.1f} pct  '
          f'(Stage 2a arm 79.0 pct)')
    print(f'  overlap median   {o.median():.4f}   (Stage 2a arm 0.0218)')
    print(f'  quartiles        {o.quantile(.25):.4f} / {o.median():.4f} / '
          f'{o.quantile(.75):.4f}   (Stage 2a arm 0.0034 / 0.0330 / 0.1247)')
    print(f'  95th pct / max   {o.quantile(.95):.4f} / {o.max():.4f}   '
          f'(Stage 2a arm 0.4528 / 0.6719)')
    print(f'  k_bic counts     {d.k_bic.value_counts().sort_index().to_dict()}')
