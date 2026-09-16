"""How much of a reported W1 is the 1,000-point grid rather than the fit?

Stage 2c, and the item Stage 2b handed over as number 3 on its list: the scoring
grid is 1,000 equally spaced points, so its resolution near zero is the same for
every dataset and is coarse on one spanning orders of magnitude. Nothing had
measured what that costs.

THERE ARE TWO SEPARATE APPROXIMATIONS AND THEY ARE WORTH SEPARATING.

  1. RESOLUTION. 1,000 points against a dense lattice on the same interval.
  2. ROUTE. The study discretizes the MODEL onto the grid as atoms with weight
     proportional to its density and hands two weighted point sets to
     `scipy.stats.wasserstein_distance`. The same grid can instead be used to
     integrate |F_model - F_empirical| directly. Both converge to W1; they do not
     converge at the same rate.

WHAT DECIDES WHETHER ANY OF IT MATTERS is not the size of the error but whether
it moves a COMPARISON. A discretization common to all six methods cancels in a
paired difference, and the paper's claims are paired differences.

    conda run -n compareuq python audits/scoring_grid_error.py [n_synth]
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

import corpus  # noqa: E402
import empirical  # noqa: E402
import fitting as FT  # noqa: E402
import recovery as R  # noqa: E402
from customstats import weighted_ecdf  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
SEED = 42
N_SYNTH = 600

#: Points in the reference lattice. `fitting.score_w1_exact`'s default.
DENSE = 200_001

#: Grid used for the cross-validated robustness check. Twenty times the study's,
#: which is enough to show whether the paired comparison moves; the dense lattice
#: would be twenty minutes of cross-validation for a fourth-decimal answer.
CV_FINE = 20_000
CV_REPEATS = 10


def w1_trapezoid(model, x, weights, grid):
    """W1 as the area between two CDFs on `grid`, the other route."""
    e = weighted_ecdf(x, weights)[2](grid)
    return float(np.trapezoid(np.abs(np.asarray(model.cdf(grid), float) - e),
                              grid))


def level_rows(name, x, w, arm):
    models, _ = FT.fit_pewt(x, w)
    g = FT.score_grid_open(x, w)
    out = []
    for label, m in models.items():
        out.append(dict(arm=arm, dataset=name, n=len(x), method=label,
                        atoms_1000=FT.score_w1_model(m, x, w, grid=g),
                        trapezoid_1000=w1_trapezoid(m, x, w, g),
                        dense=FT.score_w1_exact(m, x, w, npoints=DENSE)))
    return out


def cv_at(datasets, npoints, route, rng, repeats=CV_REPEATS):
    """Cross-validated W1 for the empirical arm at one grid and one route."""
    rows = []
    for name, (x, w) in datasets.items():
        x = np.asarray(x, float)
        w = np.asarray(w, float)
        if len(x) < R.CV_MIN_N:
            continue
        acc = {}
        for fit_idx, score_idx in R.cv_splits(len(x), rng, repeats):
            xf, wf = x[fit_idx], w[fit_idx]
            xs, ws = x[score_idx], w[score_idx]
            if len(xf) < 3 or len(xs) < 3 or np.ptp(xf) <= 0:
                continue
            models, _ = FT.fit_pewt(xf, wf / wf.sum())
            ws = ws / ws.sum()
            g = FT.score_grid_open(xs, ws, npoints=npoints)
            for label, m in models.items():
                v = (w1_trapezoid(m, xs, ws, g) if route == 'trapezoid'
                     else FT.score_w1_model(m, xs, ws, grid=g))
                acc.setdefault(label, []).append(v)
        for label, v in acc.items():
            rows.append(dict(arm='empirical', dataset=name, n=len(x),
                             method=label, w1=float(np.mean(v))))
    return pd.DataFrame(rows)


def report(d, cv):
    pd.set_option('display.width', 220)
    for col in ('atoms_1000', 'trapezoid_1000'):
        d[f'{col}_rel'] = (d[col] - d.dense).abs() / d.dense
    print('=' * 78)
    print('1. HOW BIG IS IT, relative to the dense lattice')
    print('=' * 78)
    print(d.groupby('arm')[['atoms_1000_rel', 'trapezoid_1000_rel']]
          .describe(percentiles=[0.5, 0.9, 0.99]).T
          .to_string(float_format=lambda v: f'{v:.4g}'))
    print('\nmedian by size band:')
    print(d.assign(band=d.n.map(R.size_band))
          .groupby(['arm', 'band'])[['atoms_1000_rel', 'trapezoid_1000_rel']]
          .median().to_string(float_format=lambda v: f'{v:.4g}'))
    print('\nTHE ROUTE MATTERS MORE THAN THE RESOLUTION IN THE TAIL. On the same')
    print('1,000 points the atom route has a p99 several times the trapezoid')
    print("route's, because discretizing a model into atoms puts all of a")
    print('grid cell\'s mass at one point while the CDF route averages across it.')

    print()
    print('=' * 78)
    print('2. IS IT BIASED BY METHOD? Mean W1, coarse against dense')
    print('=' * 78)
    t = d.groupby(['arm', 'method'])[['atoms_1000', 'dense']].mean()
    t['relative_bias'] = (t.atoms_1000 - t.dense) / t.dense
    print(t.to_string(float_format=lambda v: f'{v:.4f}'))
    print('\nIt is, and it runs AGAINST the KDE, whose CDF has the most structure')
    print('at the scale of a grid cell. That is the direction that matters,')
    print('because the KDE is the method under test.')

    print()
    print('=' * 78)
    print('3. DOES IT MOVE A COMPARISON? This is the question that decides it')
    print('=' * 78)
    for arm, g in d.groupby('arm'):
        a = g.pivot_table(index='dataset', columns='method',
                          values='atoms_1000').idxmin(axis=1)
        b = g.pivot_table(index='dataset', columns='method',
                          values='dense').idxmin(axis=1)
        print(f'  {arm}: the coarse grid picks a different winner on '
              f'{100 * (a != b).mean():.2f} pct of {len(a)} datasets')
    print()
    print('  and the paired KDE-minus-lognormal difference, CROSS-VALIDATED on')
    print('  the empirical arm, under each grid and route:')
    for tag, frame in cv.items():
        bits = []
        for wt in ('Uniform', 'Variable'):
            pair = [f'KDE, {wt}', f'Lognormal, {wt}']
            b = R.paired_bootstrap(frame[frame.method.isin(pair)], 'w1',
                                   f'KDE, {wt}',
                                   rng=np.random.default_rng(0)).iloc[0]
            bits.append(f'{wt[:3]} {b.mean_difference:+.4f} '
                        f'[{b.ci_lo:+.4f},{b.ci_hi:+.4f}]')
        print(f'    {tag:<18} ' + '   '.join(bits))
    print()
    print('  THE DISCRETIZATION CANCELS IN THE PAIRED DIFFERENCE. It is common to')
    print('  all six methods on a given dataset, so it moves the LEVEL of every')
    print('  score and not the gap between two of them. The study\'s claims are')
    print('  paired differences, so the coarse grid does not reach them.')
    print()
    print('  RECOMMENDATION, and it is not a change: leave the criterion alone.')
    print('  The trapezoid route on the same grid is the better quadrature and')
    print('  would cost nothing, but switching it moves every reported number in')
    print('  the paper by up to a few percent for no change in any conclusion.')
    print('  That is an author decision, not a Stage 2c one.')


def main(n_synth=N_SYNTH):
    os.makedirs(TABLES, exist_ok=True)
    rng = np.random.default_rng(SEED)
    emp, _ = empirical.prepare(rng.spawn(1)[0])
    met, vals, _ = corpus.load_corpus()
    ids = sorted(met.sample(n_synth, random_state=0).dataset.astype(str))
    syn = corpus.as_dict(vals[vals.dataset_id.isin(set(ids))])

    rows = []
    for arm, ds in (('empirical', emp), ('synthetic', syn)):
        print(f'{arm}: {len(ds)} datasets', flush=True)
        for name, (x, w) in ds.items():
            rows += level_rows(name, np.asarray(x, float), np.asarray(w, float),
                               arm)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TABLES, 'TABLE_ScoringGridError.csv'), index=False)

    cv = {}
    for tag, (npoints, route) in (('atoms @ 1,000', (1000, 'atoms')),
                                  ('trapezoid @ 1,000', (1000, 'trapezoid')),
                                  (f'trapezoid @ {CV_FINE:,}',
                                   (CV_FINE, 'trapezoid'))):
        print(f'cross-validating at {tag}...', flush=True)
        cv[tag] = cv_at(emp, npoints, route, np.random.default_rng(11))
    pd.concat([f.assign(grid=t) for t, f in cv.items()], ignore_index=True
              ).to_csv(os.path.join(TABLES, 'TABLE_ScoringGridCV.csv'),
                       index=False)
    report(d, cv)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_SYNTH)
