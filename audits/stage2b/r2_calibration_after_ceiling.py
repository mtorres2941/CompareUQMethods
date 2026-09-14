"""Did the plausibility ceiling move the generator calibration? Measured.

Stage 2b, task 1, the second gate. The ceiling changes the empirical arm, which
is the reference the corpus was tuned against, so the tuning objective has to be
rescored. The criterion is the one Stage 2a-3 used and is not reinvented here:
the active corpus scored against the arm BEFORE and AFTER the ceiling, with the
same weighted objective, against the seed-to-seed noise of 0.0066 measured in
`audits/stage2a2/p10_config_noise.py`.

Inside the noise, nothing happens. Outside it, this stage STOPS AND REPORTS:
reopening generation is an author decision (decision 47, and the Stage 2a-3
handoff section 7), not a patch inside Stage 2b.

    conda run -n compareuq python audits/stage2b/r2_calibration_after_ceiling.py [label]
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
sys.path.insert(0, os.path.join(ROOT, 'audits', 'stage2a'))
sys.path.insert(0, os.path.join(ROOT, 'audits', 'stage2a2'))
sys.path.insert(0, os.path.join(ROOT, 'audits', 'stage2a3'))

import empirical  # noqa: E402
import modality as MD  # noqa: E402
from customstats import empirical_metadata  # noqa: E402
from q3_corpus_vs_split_arm import (NOISE_MODE_TV, NOISE_OBJECTIVE_SD, SEED,  # noqa: E402
                                    corpus_modality, objective)

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'stage2b')


def arm(ceiling):
    """The empirical arm with and without the plausibility ceiling.

    Same seed and same spawn position as `q3.arm`, so the Dirichlet weights are
    the ones the rest of the stage sees and the only thing that differs between
    the two calls is the ceiling.
    """
    ds, _ = empirical.prepare(np.random.default_rng(SEED).spawn(1)[0],
                              ceiling=ceiling)
    met = pd.DataFrame({m: empirical_metadata(x, w)
                        for m, (x, w) in ds.items()}).T.astype(float)
    met.index.name = 'material'
    rng = np.random.default_rng(0)
    modes = np.array([MD.n_modes_silverman(x, rng=rng, nboot=100)
                      for x, _ in ds.values()])
    vis = np.array([MD.n_modes_visible(x) for x, _ in ds.values()])
    return met.reset_index(), modes, vis


def main(label):
    os.makedirs(TABLES, exist_ok=True)
    met, modes, vis = corpus_modality(label)
    print(f'corpus {label}: {len(met):,} datasets\n', flush=True)

    rows, detail = [], []
    for ceiling in [False, True]:
        emet, emodes, evis = arm(ceiling)
        name = f'{"with ceiling" if ceiling else "no ceiling"} ({len(emet)})'
        d, s = objective(emet, emodes, evis, met, modes, vis)
        rows.append(dict(empirical_arm=name, n_datasets=len(emet), **s))
        detail.append(d.assign(empirical_arm=name))
    out = pd.DataFrame(rows)
    out.insert(0, 'corpus', label)
    out.to_csv(os.path.join(TABLES,
                            f'TABLE_2b_CalibrationAfterCeiling_{label}.csv'),
               index=False)
    arms = list(out.empirical_arm)
    wide = pd.concat(detail).pivot(index='metric', columns='empirical_arm',
                                   values='w1_standardized')
    wide['change'] = wide[arms[1]] - wide[arms[0]]
    wide = wide.sort_values(arms[1], ascending=False)
    wide.to_csv(os.path.join(TABLES,
                             f'TABLE_2b_PerCharacteristic_{label}.csv'))

    pd.set_option('display.width', 200)
    print(out.to_string(index=False, float_format=lambda v: f'{v:,.4f}'))
    print()
    print(wide.to_string(float_format=lambda v: f'{v:8.4f}'))

    a, b = out.iloc[0], out.iloc[1]
    do = abs(b.weighted_objective - a.weighted_objective)
    dm = abs(b.mode_tv - a.mode_tv)
    dv = abs(b.visible_tv - a.visible_tv)
    print('\n=== the criterion ===')
    print(f'  objective moved         {do:.4f}   noise sd {NOISE_OBJECTIVE_SD:.4f}'
          f'   {do / NOISE_OBJECTIVE_SD:.2f} sd')
    print(f'  Silverman mode TV moved {dm:.4f}   noise    {NOISE_MODE_TV:.4f}')
    print(f'  visible   mode TV moved {dv:.4f}   noise    {NOISE_MODE_TV:.4f}')
    inside = (do < NOISE_OBJECTIVE_SD and dm < NOISE_MODE_TV
              and dv < NOISE_MODE_TV)
    print(f'  inside seed-to-seed noise on all three: {"YES" if inside else "NO"}')
    print('  -> ' + ('proceed; the calibration survives the ceiling and nothing '
                     'is regenerated'
                     if inside else
                     'STOP AND REPORT. Reopening generation is an author '
                     'decision, not this stage\'s'))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '2026-09-14d')
