"""Does a market-weighted fit land in the RIGHT PLACE and get the SHAPE wrong?

THE AUTHOR'S OBJECTION, 2026-09-29, and it is the one the earlier explanations
did not answer: "If we get 2 EPDs from a large city and 7 from a small town,
wouldn't distributions that treat all those values equally be a lot worse than
the distributions that take those differences into account? The parent
distribution is probably more clustered around the 2 EPDs, so a distribution
that accounts for that would be better and closer to the parent."

The premise is right and the conclusion needs one more term. W1 is the area
between two CDFs and it is bounded below by the distance between their means:

    W1(model, parent)  =  integral |F_model - F_parent|
    LOCATION           =  |integral (F_model - F_parent)|  =  |mean difference|
    SHAPE              =  W1 - LOCATION,  which is >= 0 on the grid exactly

That is decision 92's split applied to a fit against a parent instead of to
one dataset under two weightings. It separates BEING AIMED AT THE RIGHT
POPULATION, which is what market weights buy, from KNOWING THAT POPULATION'S
SHAPE, which is what a concentrated weight vector spends the sample on.

TWO ESTIMATORS, so the bandwidth cannot be the answer either way:

  1. the kernel estimate at EACH FIT'S OWN BEST bandwidth, swept, which no
     bandwidth rule can beat;
  2. the three-parameter lognormal, which has no bandwidth at all.

Writes `outputs/tables/audits/TABLE_WeightingLocationShape.csv`.

    python audits/weighting_location_shape.py             # 1,200 datasets
    python audits/weighting_location_shape.py --n 2500
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'src'))

import corpus                      # noqa: E402
import customstats as CS           # noqa: E402
import families                    # noqa: E402
import fitting as FT               # noqa: E402
import recovery as RC              # noqa: E402

BANDS = ((3, 9, '3-9'), (10, 80, '10-80'), (81, 99, '81-99'),
         (100, 999, '100-999'), (1000, 10 ** 9, '1000+'))

BW_MULTIPLES = tuple(np.round(np.geomspace(0.2, 5.0, 17), 4))


def band_of(n):
    for lo, hi, lab in BANDS:
        if lo <= n <= hi:
            return lab
    return ''


def split(model, parent, grid):
    """W1 against the market parent, split into location and shape.

    Both terms are taken on the SAME grid as the W1 itself, so the identity
    `location <= w1` holds exactly rather than up to a clipping rule: the
    absolute value of an integral never exceeds the integral of the absolute
    value. `location` is the difference between the two means restricted to
    that grid, which is what a signed area between CDFs is.
    """
    d = np.asarray(model.cdf(grid), float) - parent.cdf(grid, 'market')
    w1 = float(np.trapezoid(np.abs(d), grid))
    loc = abs(float(np.trapezoid(d, grid)))
    return w1, loc, w1 - loc


def kde_at(x, w, bw):
    return families.Truncated(families.WeightedKDE(x, w, bw), label='kde')


def main(n_datasets, seed):
    metrics, values, _ = corpus.load_corpus()
    metrics = metrics[~metrics.is_probe]
    pick = metrics.sample(min(n_datasets, len(metrics)), random_state=seed)
    data = corpus.as_legacy_dict(pick, values[values.dataset_id.isin(
        set(pick.dataset))])
    parents = corpus.load_parent_objects(datasets=sorted(data))

    rows = []
    for name, d in data.items():
        x, w = np.asarray(d['data'], float), np.asarray(d['weights'], float)
        n = len(x)
        eq = np.full(n, 1.0 / n)
        parent = parents[name]
        row = dict(dataset=name, n=n, band=band_of(n),
                   n_eff=float((w.sum() ** 2) / np.sum(w ** 2)))
        for label, weights in (('uniform', eq), ('market', w)):
            grid = RC.recovery_grid(x, weights, parent)
            prod = CS.weighted_bw(x, weights, FT.BW_METHOD)
            swept = [split(kde_at(x, weights, prod * m), parent, grid)
                     for m in BW_MULTIPLES]
            best = min(swept, key=lambda t: t[0])
            row[f'kde_{label}_w1'] = best[0]
            row[f'kde_{label}_loc'] = best[1]
            row[f'kde_{label}_shape'] = best[2]
            model, _ = FT.fit_family('lognormal_3p', x, weights)
            w1, loc, sh = split(model, parent, grid)
            row[f'logn_{label}_w1'] = w1
            row[f'logn_{label}_loc'] = loc
            row[f'logn_{label}_shape'] = sh
        rows.append(row)
    out = pd.DataFrame(rows)
    # PROVENANCE, required of every table this project writes: which
    # corpus, and which empirical weight rule. The rule is stamped even
    # though every row here is SYNTHETIC-arm, so a reader can see that it
    # did not apply -- decision 212 is about exactly that confusion.
    import empirical                                          # noqa: E402
    out['corpus'] = os.path.basename(corpus.active_dir())
    out['weight_rho'] = empirical.WEIGHT_RHO
    out['arm'] = 'synthetic'
    os.makedirs('outputs/tables/audits', exist_ok=True)
    out.to_csv('outputs/tables/audits/TABLE_WeightingLocationShape.csv',
               index=False)

    g = out.groupby('band')
    cols = {'datasets': g.size(), 'median_n_eff': g.n_eff.median()}
    for fam in ('kde', 'logn'):
        for term in ('loc', 'shape', 'w1'):
            cols[f'{fam}_{term}_uniform'] = g[f'{fam}_uniform_{term}'].mean()
            cols[f'{fam}_{term}_market'] = g[f'{fam}_market_{term}'].mean()
    summary = pd.DataFrame(cols).reindex(
        [b[2] for b in BANDS]).dropna(how='all')

    def show(fam, title):
        print(title)
        t = summary[[f'{fam}_{k}_{s}' for k in ('loc', 'shape', 'w1')
                     for s in ('uniform', 'market')]]
        t.columns = ['loc uni', 'loc mkt', 'shape uni', 'shape mkt',
                     'W1 uni', 'W1 mkt']
        print(t.to_string(float_format=lambda v: f'{v:.4f}'))
        print()

    print('MEAN W1 AGAINST THE TRUE MARKET-WEIGHTED PARENT, SPLIT INTO')
    print('LOCATION (how far the mean is off) and SHAPE (everything else).')
    print(f'{len(out)} synthetic datasets.')
    print()
    show('kde', 'KERNEL ESTIMATE, each fit at its OWN BEST bandwidth:')
    show('logn', 'THREE-PARAMETER LOGNORMAL, which has no bandwidth:')
    print('If market weighting aims the fit correctly, `loc mkt` is BELOW')
    print('`loc uni` at every size. If it then spends the sample on the')
    print('group that carries the weight, `shape mkt` is ABOVE `shape uni`')
    print('at small n and below it at large n.')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=1200)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    main(a.n, a.seed)
