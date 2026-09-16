"""Is the gap between two candidate configurations real, or sampling noise?

The tuning loop scores a candidate on 440 generated datasets. Two candidates
separated by one percent of the objective are not distinguishable on one draw,
and picking between them on that basis is picking noise. This re-scores each
candidate at several generator seeds and reports the spread, so a difference can
be compared against the noise it has to beat.

    conda run -n compareuq python p10_config_noise.py
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
sys.path.insert(0, os.path.join(ROOT, 'audits'))

import numpy as np  # noqa: E402
import genconfig as G  # noqa: E402
import tune_configuration as B5  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
os.makedirs(TABLES, exist_ok=True)
SEEDS = (42, 101, 202)

CANDIDATES = {
    '[1e-2.0, 0.9]': dict(overlap_log10_lo=-2.0, overlap_log10_hi=np.log10(0.9)),
    '[1e-2.5, 0.9]': dict(overlap_log10_lo=-2.5, overlap_log10_hi=np.log10(0.9)),
    '[1e-3.0, 0.5]': dict(overlap_log10_lo=-3.0, overlap_log10_hi=np.log10(0.5)),
}


if __name__ == '__main__':
    emp_met, emp_modes = B5.empirical_arm()
    print(f'{len(emp_met)} empirical datasets, '
          f'{(emp_modes == 1).mean()*100:.1f}% unimodal\n')
    rows = []
    for name, kw in CANDIDATES.items():
        cfg = G.DEFAULT.replace(**kw)
        for seed in SEEDS:
            syn_met, syn_modes = B5.sample_config(cfg, seed=seed)
            _, _, s = B5.score(emp_met, emp_modes, syn_met, syn_modes)
            rows.append(dict(config=name, seed=seed,
                             objective=s['weighted_objective'],
                             mean_w1=s['mean_w1_unweighted'],
                             mode_tv=s['mode_tv'],
                             unimodal=s['unimodal_synthetic']))
            print(f'  {name}  seed {seed}: objective {s["weighted_objective"]:.4f}'
                  f'   mean W1 {s["mean_w1_unweighted"]:.4f}'
                  f'   mode TV {s["mode_tv"]:.4f}', flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TABLES, 'TABLE_ConfigNoise.csv'), index=False)
    print('\nacross seeds:')
    g = d.groupby('config').agg(
        obj_mean=('objective', 'mean'), obj_sd=('objective', 'std'),
        obj_min=('objective', 'min'), obj_max=('objective', 'max'),
        tv_mean=('mode_tv', 'mean'), tv_sd=('mode_tv', 'std'),
        uni_mean=('unimodal', 'mean'))
    print(g.to_string(float_format=lambda v: f'{v:,.4f}'))
    sd = d.groupby('config').objective.std().mean()
    spread = g.obj_mean.max() - g.obj_mean.min()
    print(f'\n  typical within-config sd across seeds : {sd:.4f}')
    print(f'  spread between config means           : {spread:.4f}')
    print(f'  ratio                                 : {spread/sd:.2f}'
          f'   ({"separable" if spread > 2*sd else "NOT separable on this evidence"})')
