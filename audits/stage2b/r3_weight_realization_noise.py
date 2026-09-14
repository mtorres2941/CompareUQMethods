"""How much of the objective's movement is the weight draw rather than the data?

Stage 2b, task 1. `r2_calibration_after_ceiling.py` reads the plausibility
ceiling as moving the tuning objective by 1.10 seed-to-seed standard deviations,
marginally outside the 0.0066 gate. That figure needs one thing checked before
it can be read as a real movement.

THE CONFOUND. Each dataset's Dirichlet weights are drawn from a stream keyed by
its NAME, with one weight per surviving value. Six datasets lose a value to the
ceiling, so their weight VECTOR LENGTH changes and they receive an entirely
fresh draw. The other 143 are bit-identical. Discrepancy entry 32 records that
redrawing the weights of the same datasets moves `w_v_uw_wasserstein` by up to
1.02 per dataset, and `w_v_uw_wasserstein` is the single largest contributor to
the movement r2 measures, at +0.0446 of +0.0072 weighted.

So the question is not whether the objective moved. It is whether it moved by
more than the objective moves anyway when nothing changes but the weight
realization. THE 0.0066 GATE DOES NOT COVER THAT: it is generator seed-to-seed
noise from `audits/stage2a2/p10_config_noise.py`, measured with the empirical
arm held fixed.

This script measures the missing term: the same arm, the same values, the same
corpus, `n` independent Dirichlet realizations, and the spread of the objective
across them. It is a MEASUREMENT AND NOT A REMEDY. The gate verdict r2 prints
stands as printed; this only says what the verdict is made of.

The Silverman and visible mode counts depend on the values alone, so they are
computed once and reused across realizations. That is what makes this cheap.

    conda run -n compareuq python audits/stage2b/r3_weight_realization_noise.py [label] [n]
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
from q3_corpus_vs_split_arm import (NOISE_OBJECTIVE_SD, SEED,  # noqa: E402
                                    corpus_modality, objective)

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'stage2b')
N_REALIZATIONS = 12


def modes_for(ceiling):
    """Mode counts, which depend on the values and not on the weights."""
    ds, _ = empirical.prepare(np.random.default_rng(SEED).spawn(1)[0],
                              ceiling=ceiling)
    rng = np.random.default_rng(0)
    modes = np.array([MD.n_modes_silverman(x, rng=rng, nboot=100)
                      for x, _ in ds.values()])
    vis = np.array([MD.n_modes_visible(x) for x, _ in ds.values()])
    return modes, vis


def metrics_for(ceiling, seed):
    ds, _ = empirical.prepare(np.random.default_rng(seed).spawn(1)[0],
                              ceiling=ceiling)
    met = pd.DataFrame({m: empirical_metadata(x, w)
                        for m, (x, w) in ds.items()}).T.astype(float)
    met.index.name = 'material'
    return met.reset_index()


def main(label, n):
    os.makedirs(TABLES, exist_ok=True)
    met, modes_syn, vis_syn = corpus_modality(label)
    print(f'corpus {label}: {len(met):,} datasets, '
          f'{n} weight realizations per arm\n', flush=True)

    rows = []
    for ceiling in [False, True]:
        emodes, evis = modes_for(ceiling)
        for i in range(n):
            seed = SEED if i == 0 else 10_000 + i
            emet = metrics_for(ceiling, seed)
            _, s = objective(emet, emodes, evis, met, modes_syn, vis_syn)
            rows.append(dict(ceiling=ceiling, realization=i, seed=seed,
                             is_stage_seed=(i == 0), **s))
            print(f'  ceiling={str(ceiling):<5} realization {i:>2} '
                  f'seed {seed:<6} objective {s["weighted_objective"]:.4f}',
                  flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLES,
                            f'TABLE_2b_WeightRealizationNoise_{label}.csv'),
               index=False)

    print()
    summary = out.groupby('ceiling').weighted_objective.agg(
        ['mean', 'std', 'min', 'max'])
    print(summary.to_string(float_format=lambda v: f'{v:.4f}'))

    sd_weights = float(out.weighted_objective.std())
    a = out[(~out.ceiling) & out.is_stage_seed].weighted_objective.iloc[0]
    b = out[out.ceiling & out.is_stage_seed].weighted_objective.iloc[0]
    observed = abs(b - a)
    mean_shift = abs(float(summary.loc[True, 'mean'] - summary.loc[False, 'mean']))
    combined = float(np.hypot(NOISE_OBJECTIVE_SD, sd_weights))

    print('\n=== what the r2 movement is made of ===')
    print(f'  objective movement r2 reports, at the stage seed   {observed:.4f}')
    print(f'  movement in the MEAN over {n} weight realizations     {mean_shift:.4f}')
    print(f'  sd of the objective over weight realizations alone {sd_weights:.4f}')
    print(f'  generator seed-to-seed sd (the 0.0066 gate)        {NOISE_OBJECTIVE_SD:.4f}')
    print(f'  the two noise terms combined in quadrature         {combined:.4f}')
    print(f'\n  observed movement is {observed / combined:.2f} combined sd '
          f'and {observed / NOISE_OBJECTIVE_SD:.2f} generator sd')
    print('\n  This does not overturn the r2 verdict. It says what share of the\n'
          '  movement a redrawn weight vector accounts for, so the author can\n'
          '  judge whether the gate has found a change in the DATA or a change\n'
          '  in one Dirichlet draw. Entry 32, owner 2h.')


if __name__ == '__main__':
    lab = sys.argv[1] if len(sys.argv) > 1 else '2026-09-14d'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else N_REALIZATIONS
    main(lab, n)
