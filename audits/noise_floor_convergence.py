"""Does the Monte Carlo noise floor go away with more draws? Stage 2d.

The author's challenge, 2026-09-17: comparing two methods under two independent
random streams is the normal thing to do, the study already draws 10,000 samples
so it ought to have converged, and a 5.33 percent disagreement between a method
and ITSELF is shockingly high. "I'd be shocked if 10,000 weren't sufficient."

That deserves a measurement rather than a defence, and the measurement changes
what the number means.

WHAT IS AND IS NOT CONVERGED. Each material's rank-1 frequency IS converged: at
10,000 draws its Monte Carlo standard error is about 0.0043, well under half a
percent. What is not converged -- and cannot be, at any sample size -- is the
IDENTITY OF THE ARGMAX when two materials are nearly tied. The argmax is a
discontinuous function of continuous estimates, so where the true gap between the
top two is small compared with the sampling error, which material comes first is
a coin flip no matter how long the simulation runs.

So the floor should fall like one over the square root of the draw count, not to
zero, and it should be concentrated entirely in the near-tied groups. Both are
testable and this script tests them.

    conda run -n compareuq python audits/noise_floor_convergence.py [n_groups]
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

import corpus  # noqa: E402
import fitting  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
SEED = 20260917
DRAW_COUNTS = (1_000, 10_000, 100_000)
N_GROUPS = 300
METHOD = 'KDE, Uniform'
FMT = lambda v: f'{v:.4f}'  # noqa: E731


def rank1(draws):
    return ((-draws).argsort(axis=1).argsort(axis=1) == 0).mean(axis=0)


def main(n_groups=N_GROUPS):
    os.makedirs(TABLES, exist_ok=True)
    n_groups = int(n_groups)
    rng = np.random.default_rng(SEED)
    metrics, values, meta = corpus.load_corpus()
    DATA = corpus.as_legacy_dict(metrics, values)
    combos = corpus.load_combos()[:n_groups]
    need = sorted({d for g in combos for d in g})
    models = {ds: fitting.fit_pewt(DATA[ds]['data'], DATA[ds]['weights'])[0]
              for ds in need}
    print(f'{n_groups} pLCA groups, method {METHOD}, seed {SEED}')

    rows = []
    for neccs in DRAW_COUNTS:
        t0 = time.time()
        for gi, g in enumerate(combos):
            names = list(g)
            a = np.column_stack([models[d][METHOD].rvs(neccs, random_state=rng)
                                 for d in names])
            b = np.column_stack([models[d][METHOD].rvs(neccs, random_state=rng)
                                 for d in names])
            ra, rb = rank1(a), rank1(b)
            # The TRUE gap is unknown; the best available estimate of it is the
            # pooled one, so use the mean of the two runs.
            pooled = np.sort(0.5 * (ra + rb))[::-1]
            rows.append(dict(neccs=neccs, plca=gi,
                             flip=int(np.argmax(ra)) != int(np.argmax(rb)),
                             gap=float(pooled[0] - pooled[1]),
                             se=float(np.sqrt(0.25 * 0.75 / neccs))))
        print(f'  {neccs:>7,} draws in {time.time() - t0:.0f}s')
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TABLES, 'TABLE_NoiseFloorConvergence.csv'),
             index=False)

    print()
    print('=' * 78)
    print('1. DOES THE FLOOR FALL LIKE 1 / SQRT(DRAWS)?')
    print('=' * 78)
    by = d.groupby('neccs').flip.agg(['mean', 'size'])
    by['vs_10k'] = by['mean'] / by.loc[10_000, 'mean']
    by['sqrt_predicted'] = np.sqrt(10_000 / by.index.to_numpy())
    print(by.to_string(float_format=FMT))
    print()
    print('  If the ratio column tracks the square-root prediction, the floor is')
    print('  ordinary Monte Carlo error in the ARGMAX and shrinks only as the')
    print('  square root of effort: cutting it tenfold costs 100x the compute.')

    print()
    print('=' * 78)
    print('2. IS IT CONCENTRATED IN THE NEAR-TIED GROUPS?')
    print('=' * 78)
    print('The gap between the top two rank-1 frequencies, in units of the')
    print('Monte Carlo standard error of one of them.')
    for neccs, g in d.groupby('neccs'):
        g = g.assign(band=pd.cut(g.gap / g.se, [0, 1, 2, 4, 8, np.inf],
                                 labels=['0-1 se', '1-2 se', '2-4 se', '4-8 se',
                                         '8+ se']))
        t = g.groupby('band', observed=True).flip.agg(['mean', 'size'])
        t.columns = ['flip rate', 'groups']
        print(f'\n  {neccs:,} draws   overall {g.flip.mean():.4f}')
        print(t.to_string(float_format=FMT))

    print()
    print('=' * 78)
    print('WHAT THIS SETTLES')
    print('=' * 78)
    f10 = by.loc[10_000, 'mean']
    f100 = by.loc[100_000, 'mean']
    print(f'  The estimates ARE converged: one rank-1 frequency has a standard')
    print(f'  error of {np.sqrt(0.25*0.75/10_000):.4f} at 10,000 draws. What is not')
    print('  converged is WHICH of two nearly equal materials is larger, and')
    print('  that is a property of the question, not of the sample size.')
    print()
    print(f'  Ten times the draws takes the floor from {f10:.4f} to {f100:.4f}, a')
    print(f'  factor of {f10/f100:.2f} against the {np.sqrt(10):.2f} a square-root law')
    print('  predicts. Reaching 0.5 percent would need roughly a hundred times')
    print('  the compute of the current run and would still not be zero.')
    print()
    print('  Common random numbers take it to EXACTLY zero for nothing, because')
    print('  they remove the comparison noise rather than the estimation noise:')
    print('  the same model on the same variates gives the same draws.')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else N_GROUPS)
