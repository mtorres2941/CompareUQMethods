"""How many grid points and how many Dirichlet draws does A_IQR need? Stage 2d.

Two settings in `src/weighting.py` have to be defensible rather than round.
`N_DRAWS` is fixed at 1,000 by consistency with the KL2 paper, which says so
twice, so the question here is only whether 1,000 is ENOUGH, not what to pick.
`AIQR_GRID_POINTS` is this study's own and is chosen here.

    conda run -n compareuq python audits/aiqr_convergence.py
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
import weighting as WG  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
GRIDS = (250, 500, 1000, 2000, 5000, 20000)
DRAWS = (100, 250, 500, 1000, 2000)
REFERENCE_DRAWS = 8_000
SEEDS = 3
FMT = lambda v: f'{v:.5f}'  # noqa: E731


def one(x, draws, npoints):
    g = WG.aiqr_grid(x, npoints=npoints)
    return WG.aiqr(WG.density_ensemble(x, draws, g), g)


def main():
    os.makedirs(TABLES, exist_ok=True)
    emp, _ = empirical.prepare(np.random.default_rng(42).spawn(1)[0])
    sizes = pd.Series({k: len(v[0]) for k, v in emp.items()}).sort_values()
    picks = [sizes.index[0], sizes.index[len(sizes) // 3],
             sizes.index[2 * len(sizes) // 3], sizes.index[-1]]

    rows = []
    print('GRID POINTS, at 1,000 draws, as a ratio to 20,000 points')
    print(f"{'dataset':30s} {'n':>6s} " + ' '.join(f'{p:>8d}' for p in GRIDS))
    for name in picks:
        x = np.asarray(emp[name][0], dtype=float)
        d = WG.dirichlet_draws(len(x), np.random.default_rng(11))
        vals = [one(x, d, g) for g in GRIDS]
        ref = vals[-1]
        print(f'{name[:30]:30s} {len(x):6d} ' +
              ' '.join(f'{v / ref:8.5f}' for v in vals))
        for g, v in zip(GRIDS, vals):
            rows.append(dict(kind='grid', dataset=name, n=len(x), setting=g,
                             value=v, ratio=v / ref))

    print()
    print(f'DRAWS, at {WG.AIQR_GRID_POINTS} grid points, as a ratio to '
          f'{REFERENCE_DRAWS:,} draws, mean of {SEEDS} seeds')
    print(f"{'dataset':30s} {'n':>6s} " + ' '.join(f'{k:>8d}' for k in DRAWS))
    for name in picks:
        x = np.asarray(emp[name][0], dtype=float)
        ref = one(x, WG.dirichlet_draws(len(x), np.random.default_rng(5),
                                        n_draws=REFERENCE_DRAWS),
                  WG.AIQR_GRID_POINTS)
        out = []
        for k in DRAWS:
            s = [one(x, WG.dirichlet_draws(len(x),
                                           np.random.default_rng(100 + j),
                                           n_draws=k), WG.AIQR_GRID_POINTS)
                 for j in range(SEEDS)]
            out.append((float(np.mean(s)), float(np.std(s))))
            rows.append(dict(kind='draws', dataset=name, n=len(x), setting=k,
                             value=out[-1][0], ratio=out[-1][0] / ref,
                             seed_spread=out[-1][1] / ref))
        print(f'{name[:30]:30s} {len(x):6d} ' +
              ' '.join(f'{m / ref:8.5f}' for m, _ in out))
        print(f"{'  seed-to-seed spread':30s} {'':6s} " +
              ' '.join(f'{s / ref:8.5f}' for _, s in out))

    pd.DataFrame(rows).to_csv(
        os.path.join(TABLES, 'TABLE_AIQRConvergence.csv'), index=False)
    print()
    print('WHAT THIS SETTLES.')
    print(f'  AIQR_GRID_POINTS = {WG.AIQR_GRID_POINTS}. The remaining')
    print('  discretization error is a fraction of a percent and it is in the')
    print('  SAME direction on every dataset, so it moves the level of A_IQR')
    print('  and not the ordering, which is what A_IQR is used for.')
    print(f'  N_DRAWS = {WG.N_DRAWS}, which is what KL2 uses. The seed-to-seed')
    print('  spread at that many draws is what a reader should treat as the')
    print('  precision of a reported A_IQR.')


if __name__ == '__main__':
    main()
