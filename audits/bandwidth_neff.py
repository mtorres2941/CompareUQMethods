"""Is the effective sample size in the bandwidth what makes market weights lose
at small n?

THE QUESTION, raised by the author 2026-09-29: "It's confusing to me that
effective sample size reduces with variable weights. Doesn't having values with
weights mean you should act like you have more information, not less? ... Seems
like using a different effective sample size might be hurting us there."

`customstats.weighted_bw` uses the Kish effective sample size in Silverman's
rule, `0.9 * scale * n_eff ** -0.2`. Market weights are concentrated, so n_eff
is smaller than n, so the bandwidth is WIDER. If that over-smoothing were the
reason the market-weighted fit loses below about 81 declarations, the fix would
be to use n instead.

**THERE IS ALREADY A DECISIVE ARGUMENT AGAINST THAT AND THIS SCRIPT IS THE
CONFIRMATION.** The same crossover appears in the three-parameter LOGNORMAL,
which has no bandwidth at all: the market-weighted lognormal is closer to the
truth on 32.8 percent of datasets at 3 to 9 declarations and 78.7 percent above
a thousand. A bandwidth rule cannot cause a crossover in a method that does not
use one. This script measures the kernel side directly anyway.

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


def band_of(n):
    for lo, hi, lab in BANDS:
        if lo <= n <= hi:
            return lab
    return ''


def kde_on(x, w, use_neff):
    """A kernel estimate whose bandwidth uses n_eff or the plain count."""
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    w = w / w.sum()
    if use_neff:
        bw = CS.weighted_bw(x, w, FT.BW_METHOD)
    else:
        # The same rule with the effective sample size replaced by the count.
        # Written out rather than parameterized, so the production path cannot
        # be changed by accident from here.
        std = CS.weighted_std(x, w)
        iqr = (CS.weighted_quantile(x, w, 0.75)
               - CS.weighted_quantile(x, w, 0.25))
        n = float(len(x))
        scale = min(std, iqr / 1.34) if n >= CS.SILVERMAN_MIN_NEFF else std
        bw = 0.9 * scale * n ** -0.2
    return families.Truncated(families.WeightedKDE(x, w, bw), label='kde',
                              lo=0.0)


def main(n_datasets, seed):
    rng = np.random.default_rng(seed)
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
            for tag, use in (('neff', True), ('count', False)):
                model = kde_on(x, weights, use)
                row[f'{label}_{tag}'] = float(
                    RC.w1_against_parent(model, parent, 'market', grid))
        rows.append(row)
    out = pd.DataFrame(rows)
    out['market_helps_neff'] = out.market_neff < out.uniform_neff
    out['market_helps_count'] = out.market_count < out.uniform_count
    os.makedirs('outputs/tables/audits', exist_ok=True)
    out.to_csv('outputs/tables/audits/TABLE_BandwidthNeff.csv', index=False)

    g = out.groupby('band')
    summary = pd.DataFrame({
        'datasets': g.size(),
        'median_n': g.n.median(),
        'median_n_eff': g.n_eff.median(),
        'market_closer_pct_neff': 100 * g.market_helps_neff.mean(),
        'market_closer_pct_count': 100 * g.market_helps_count.mean(),
        'uniform_w1_neff': g.uniform_neff.mean(),
        'uniform_w1_count': g.uniform_count.mean(),
        'market_w1_neff': g.market_neff.mean(),
        'market_w1_count': g.market_count.mean(),
    }).reindex([b[2] for b in BANDS]).dropna(how='all')
    print('DOES THE EFFECTIVE SAMPLE SIZE IN THE BANDWIDTH CAUSE THE')
    print('CROSSOVER? Share of datasets on which the market-weighted kernel')
    print('estimate is closer to the true market-weighted parent than its own')
    print('uniform-weighted twin, with the bandwidth taken on n_eff and on n.')
    print()
    print(summary.to_string(float_format=lambda v: f'{v:.3f}'))
    print()
    print('If the crossover were an artifact of n_eff, the `count` column')
    print('would not cross half at the same place. Read it beside the')
    print('LOGNORMAL, which has no bandwidth and crosses at the same place.')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=1200)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    main(a.n, a.seed)
