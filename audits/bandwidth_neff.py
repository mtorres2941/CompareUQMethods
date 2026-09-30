"""Is the effective sample size in the bandwidth why market weighting loses at
small n? Three tests, and the last one cannot be argued with.

THE AUTHOR'S HYPOTHESIS, 2026-09-29: "Accounting for data weights means you
significantly reduce your n_effective and it way overpunishes methods like KDE,
since its bandwidth is highly dependent on n_effective. I bet if you used a
regular n (nine weighted observations means n=9), KDE variable would dominate."

`customstats.weighted_bw` uses the Kish effective sample size in Silverman's
rule, `0.9 * scale * n_eff ** -0.2`. Market weights are concentrated, so n_eff
is smaller than n, so the bandwidth is WIDER. Three ways to ask whether that is
what makes the market-weighted fit lose below about 81 declarations:

  1. THE PRODUCTION RULE, n_eff, which is what the study uses.
  2. THE PLAIN COUNT, n, which is what the hypothesis proposes.
  3. EACH FIT'S OWN BEST BANDWIDTH, swept. **This is the decisive one**: if
     the market-weighted fit still loses when both fits are given the
     bandwidth that minimizes their own distance to the truth, then no
     bandwidth rule can be the reason, because no rule can beat the best one.

AND A FOURTH ARGUMENT THAT NEEDS NO BANDWIDTH AT ALL. The same crossover
appears in the three-parameter lognormal, which has no bandwidth: market
weights are closer on 32.8 percent of datasets at 3 to 9 declarations and 78.7
percent above a thousand.

The script also reports how far apart the two TRUE populations are -- the
sampling parent a uniform-weighted fit estimates and the market parent it is
scored against -- because that distance is the bias uniform weighting carries,
and it is the other half of the trade.

Writes `outputs/tables/audits/TABLE_BandwidthNeff.csv`.

    python audits/bandwidth_neff.py            # 1,200 datasets, a few minutes
    python audits/bandwidth_neff.py --n 4000
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

#: Multipliers of the production bandwidth the sweep tries. Wide enough that
#: the best value is interior for essentially every dataset.
BW_MULTIPLES = tuple(np.round(np.geomspace(0.2, 5.0, 17), 4))


def band_of(n):
    for lo, hi, lab in BANDS:
        if lo <= n <= hi:
            return lab
    return ''


def bandwidths(x, w):
    """The production bandwidth and the same rule on the plain count."""
    prod = CS.weighted_bw(x, w, FT.BW_METHOD)
    std = CS.weighted_std(x, w)
    iqr = CS.weighted_quantile(x, w, 0.75) - CS.weighted_quantile(x, w, 0.25)
    n = float(len(x))
    scale = min(std, iqr / 1.34) if n >= CS.SILVERMAN_MIN_NEFF else std
    return float(prod), float(0.9 * scale * n ** -0.2)


def score(x, w, bw, parent, grid):
    model = families.Truncated(families.WeightedKDE(x, w, bw), label='kde')
    return float(RC.w1_against_parent(model, parent, 'market', grid))


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
        # How far apart the two TRUE populations are: the bias a
        # uniform-weighted fit carries, before any estimation error.
        g_bias = RC.recovery_grid(x, eq, parent)
        row['parent_separation'] = float(np.trapezoid(
            np.abs(parent.cdf(g_bias, 'market')
                   - parent.cdf(g_bias, 'uniform')), g_bias))
        for label, weights in (('uniform', eq), ('market', w)):
            grid = RC.recovery_grid(x, weights, parent)
            prod, plain = bandwidths(x, weights)
            row[f'{label}_neff'] = score(x, weights, prod, parent, grid)
            row[f'{label}_count'] = score(x, weights, plain, parent, grid)
            swept = [score(x, weights, prod * m, parent, grid)
                     for m in BW_MULTIPLES]
            row[f'{label}_best'] = float(np.min(swept))
            row[f'{label}_best_mult'] = float(
                BW_MULTIPLES[int(np.argmin(swept))])
        rows.append(row)
    out = pd.DataFrame(rows)
    for tag in ('neff', 'count', 'best'):
        out[f'market_helps_{tag}'] = out[f'market_{tag}'] < out[f'uniform_{tag}']
    os.makedirs('outputs/tables/audits', exist_ok=True)
    out.to_csv('outputs/tables/audits/TABLE_BandwidthNeff.csv', index=False)

    g = out.groupby('band')
    summary = pd.DataFrame({
        'datasets': g.size(),
        'median_n': g.n.median(),
        'median_n_eff': g.n_eff.median(),
        'market_closer_neff': 100 * g.market_helps_neff.mean(),
        'market_closer_count': 100 * g.market_helps_count.mean(),
        'market_closer_best_bw': 100 * g.market_helps_best.mean(),
        'median_parent_separation': g.parent_separation.median(),
        'uniform_best_mult': g.uniform_best_mult.median(),
        'market_best_mult': g.market_best_mult.median(),
    }).reindex([b[2] for b in BANDS]).dropna(how='all')
    print('DOES THE BANDWIDTH EXPLAIN THE CROSSOVER? Share of datasets on')
    print('which the market-weighted kernel estimate is closer to the TRUE')
    print('market-weighted parent than its own uniform-weighted twin.')
    print()
    print(summary.to_string(float_format=lambda v: f'{v:.3f}'))
    print()
    print('`best_bw` gives EACH fit the bandwidth that minimizes its own')
    print('distance to the truth, which no rule can beat. If the crossover')
    print('survives there, no bandwidth rule causes it.')
    print()
    print('`parent_separation` is how far the market-weighted population sits')
    print('from the sampling population: the BIAS a uniform-weighted fit')
    print('carries whatever its bandwidth. It is the other half of the trade.')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=1200)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    main(a.n, a.seed)
