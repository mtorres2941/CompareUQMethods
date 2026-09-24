"""Where the design comparison becomes safe, as a claimed saving.

WHY THIS EXISTS. The study runs the design swap at six claimed savings -- 0, 1,
2, 5, 10 and 20 percent -- which brackets the 5 percent risk level inside a
five-point window and leaves the 1 percent level inside a TEN-point window with
no observation between 10 and 20 percent. The manuscript session asked for a
threshold to sit alongside the safe-lead ratio of 2.13, and that ratio came
from a dense sweep with a bootstrap interval. Interpolating across a 2x gap
would not be the same kind of number. This fills the gap.

TWO RISKS, and the second is the one a practitioner cares about:

    methods disagree with each other   at least two of the six land on
                                       opposite sides of a half
    a method names the wrong design    a method lands on the opposite side
                                       from the TRUTH run on the same draws

    conda run -n compareuq python audits/swap_saving_threshold.py [n_pairs]
"""
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import corpus                       # noqa: E402
import fitting as FT                # noqa: E402
import plca as PL                   # noqa: E402

OUT = os.path.join(ROOT, 'outputs', 'tables', 'audits')
#: Dense between 5 and 20 percent, where both crossings sit; the study's own
#: six levels are kept so the two can be checked against each other.
SAVINGS = (0.0, 0.02, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.14, 0.16,
           0.18, 0.20)
SEED = 20260924


def main(argv):
    n_pairs = int(argv[1]) if len(argv) > 1 else 900
    neccs = int(argv[2]) if len(argv) > 2 else 10000
    rng = np.random.default_rng(SEED)
    metrics, values, _meta = corpus.load_corpus()
    nmats = 4
    groups = PL.resample_groups(metrics.dataset.tolist(), nmats + 1, n_pairs,
                                rng)
    needed = sorted({d for g in groups for d in g})
    print(f'{n_pairs} pairs, {len(needed)} distinct datasets, '
          f'{len(SAVINGS)} savings, {neccs} draws', flush=True)

    t0 = time.time()
    by = values[values.dataset_id.isin(needed)].groupby('dataset_id')
    models = {}
    for i, ds in enumerate(needed):
        d = by.get_group(ds)
        models[ds], _p = FT.fit_pewt(d.value.to_numpy(float),
                                     d.weight.to_numpy(float))
        if i and i % 500 == 0:
            print(f'  fitted {i}/{len(needed)}  {time.time()-t0:.0f}s',
                  flush=True)
    print(f'  fitting done in {time.time()-t0:.0f}s', flush=True)

    samplers = PL.LazySamplers(corpus.load_parent_objects(datasets=needed),
                               scheme=PL.TRUTH_SCHEME)
    t1 = time.time()
    df = PL.swap_run(models, groups, rng, savings=SAVINGS, neccs=neccs,
                     samplers=samplers)
    print(f'  swap done in {time.time()-t1:.0f}s', flush=True)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, 'TABLE_SwapSavingSweep.csv.gz'), index=False)
    report(df)


def report(df, level_list=(0.05, 0.01)):
    M = sorted(df.method.unique())
    df = df.assign(wrong=(df.discernibility > 0.5)
                   != (df.discernibility__truth > 0.5))
    w = df.pivot_table(index=['pair', 'saving'], columns='method',
                       values='discernibility')[M]
    obs = pd.DataFrame({
        'truth': df.groupby('saving').discernibility__truth.mean(),
        'disagree': ((w.min(axis=1) < 0.5) & (w.max(axis=1) > 0.5))
        .groupby(level='saving').mean(),
        'wrong': df.groupby('saving').wrong.mean()})
    print('\nOBSERVED, by claimed saving:')
    print((100 * obs).to_string(float_format=lambda v: f'{v:.2f}'))

    print('\nCROSSINGS, by linear interpolation between the two bracketing '
          'observations\n(the sweep is dense enough that no interpolation '
          'spans more than two points):')
    for col, lab in (('disagree', 'methods disagree with each other'),
                     ('wrong', 'a method names the wrong design')):
        y = obs[col].to_numpy()
        x = obs.index.to_numpy()
        for lvl in level_list:
            above = x[y > lvl]
            below = x[y <= lvl]
            if not (len(above) and len(below)):
                print(f'  {lab:34s} {100*lvl:>2.0f} pct: not bracketed')
                continue
            lo, hi = above.max(), below.min()
            ylo = obs.loc[lo, col]
            yhi = obs.loc[hi, col]
            f = (lvl - ylo) / (yhi - ylo)
            print(f'  {lab:34s} {100*lvl:>2.0f} pct: '
                  f'{100*(lo + f*(hi-lo)):5.2f} pct saving   '
                  f'(bracket {100*lo:.0f} to {100*hi:.0f})')
    print(f'\nwrote {OUT}/TABLE_SwapSavingSweep.csv.gz')


if __name__ == '__main__':
    main(sys.argv)
