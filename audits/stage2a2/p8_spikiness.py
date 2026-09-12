"""How often is a synthetic dataset a needle plus an empty tail?

The dataset-examples figure shows many panels in which the body of the data
occupies a tiny fraction of its own support. That is what driving the component
overlap down buys, and it is the same cause as fit_lognorm_SW degrading. It has
to be measured against the empirical data rather than eyeballed, because real
ECC categories genuinely do span orders of magnitude.

Two measures, both scale free and both computable on either arm:

  concentration  the interdecile range (p90 - p10) divided by the full range.
                 Small means the body is a needle inside a long support.
  tail_gap       the largest multiplicative gap between consecutive sorted
                 values in the upper half, in units of the interquartile range.
                 Large means an isolated far-out clump.

    conda run -n compareuq python p8_spikiness.py [corpus_label ...]
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import empirical  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'stage2a2')
PROCESSED = os.path.join(ROOT, 'data', 'processed')
SEED = 42


def measures(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan, np.nan
    rng_full = x.max() - x.min()
    p10, p90 = np.percentile(x, [10, 90])
    conc = (p90 - p10) / rng_full if rng_full > 0 else np.nan
    s = np.sort(x)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    upper = s[len(s) // 2:]
    gap = np.max(np.diff(upper)) / iqr if len(upper) > 1 and iqr > 0 else np.nan
    return conc, gap


def summarize(name, rows):
    d = pd.DataFrame(rows, columns=['concentration', 'tail_gap']).dropna()
    print(f'  {name:<24} n={len(d):>6}   '
          f'concentration p10/median {d.concentration.quantile(.10):.3f} / '
          f'{d.concentration.median():.3f}   '
          f'share below 0.10 {100*(d.concentration < 0.10).mean():5.1f}%   '
          f'tail_gap median/p90 {d.tail_gap.median():.2f} / '
          f'{d.tail_gap.quantile(.90):.2f}')
    return d


if __name__ == '__main__':
    labels = sys.argv[1:] or ['2026-09-12b', '2026-09-12c']
    out = {}

    ds, _ = empirical.prepare(np.random.default_rng(SEED).spawn(1)[0])
    print('spikiness, empirical vs synthetic')
    out['empirical'] = summarize('empirical (136)',
                                 [measures(x) for x, _ in ds.values()])

    for lab in labels:
        d = os.path.join(PROCESSED, f'corpus_{lab}')
        met = pd.read_parquet(os.path.join(d, 'metrics.parquet'))
        keep = set(met[~met.is_probe].dataset.astype(str))
        vals = pd.read_parquet(os.path.join(d, 'values.parquet'))
        rows, strat = [], []
        stratum_of = dict(zip(met.dataset.astype(str), met.stratum))
        for dsid, g in vals.groupby('dataset_id', observed=True):
            if str(dsid) not in keep:
                continue
            m = measures(g['value'].to_numpy())
            rows.append(m)
            strat.append(stratum_of.get(str(dsid)))
        out[lab] = summarize(f'corpus {lab}', rows)
        df = pd.DataFrame(rows, columns=['concentration', 'tail_gap'])
        df['stratum'] = strat
        print('       by stratum, share with concentration below 0.10:')
        for st, g in df.dropna().groupby('stratum'):
            print(f'         {st:<16} {100*(g.concentration < 0.10).mean():5.1f}%'
                  f'   median concentration {g.concentration.median():.3f}')

    frames = []
    for k, v in out.items():
        frames.append(v.assign(arm=k))
    pd.concat(frames).to_csv(os.path.join(TABLES, 'TABLE_2a2_Spikiness.csv'),
                             index=False)
