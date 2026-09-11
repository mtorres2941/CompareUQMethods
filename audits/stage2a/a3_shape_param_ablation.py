"""Stage 2a, Part 0 item 1d: is correct seeding alone sufficient?

Stage 0 left this open and Stage 1 could not answer it. Three arms, identical
except for where randomness enters the component shape parameters:

  A  legacy            values deterministic given (type, count); a, df, s constant
  B  values_only       values correctly drawn; a, df, s pinned at the legacy
                       constants (3.458732, 3.366328, 1.136962)
  C  current           values correctly drawn; a, df, s drawn per component

B is the arm that answers the question. If B already covers the metric space
as well as C, correct seeding alone was sufficient and the shape parameters
could stay constants. If B falls short of C, they must be drawn.

Writes TABLE_2a_ShapeParamAblation.csv
"""
import numpy as np, pandas as pd, sys, os
import scipy.stats as stats
from _common import write

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datageneration as current
from customstats import empirical_metadata
from a1_collapse_corpus_damage import (gen_legacy, gen_current, effective_dimension,
                                       METRICS_CORE, N_DATASETS, SEED)

A_CONST, DF_CONST, S_CONST = 3.458732, 3.366328, 1.136962


def _pinned_component(t, loc, scale, cnt, rng):
    """generate_random_numbers with the shape parameters pinned at the legacy
    constants but the VALUES drawn correctly from the shared Generator."""
    if t == 'gauss':
        return rng.normal(loc=loc, scale=scale, size=cnt)
    if t == 'skewnorm':
        return stats.skewnorm.rvs(a=A_CONST, loc=loc, scale=scale, random_state=rng, size=cnt)
    if t == 'studentt':
        return stats.t.rvs(DF_CONST, loc=loc, scale=scale, random_state=rng, size=cnt)
    if t == 'lognorm':
        return stats.lognorm.rvs(s=S_CONST, loc=loc, scale=scale, random_state=rng, size=cnt)
    raise ValueError(t)


def gen_values_only(ndatasets=N_DATASETS, seed=SEED):
    orig = current.generate_random_numbers
    current.generate_random_numbers = _pinned_component
    try:
        rng = np.random.default_rng(seed)
        out = {}
        for i in range(ndatasets):
            n = int(current.random_logcount(rng, lo=4, hi=1000, n=1)[0])
            data, w = current.random_irregular_dataset(n=n, rng=rng)
            out[f'dataset{i}'] = dict(data=data, weights=w,
                                      metrics=empirical_metadata(data, w))
        return out
    finally:
        current.generate_random_numbers = orig


def summarize(DATA, label):
    df = pd.DataFrame({k: v['metrics'] for k, v in DATA.items()}).T[METRICS_CORE].astype(float)
    df = df.replace([np.inf, -np.inf], np.nan)
    row = dict(arm=label, effective_metric_dimension=effective_dimension(df.values))
    for m in METRICS_CORE:
        s = df[m].dropna()
        row[f'sd__{m}'] = float(s.std())
        row[f'range99__{m}'] = float(s.quantile(.995) - s.quantile(.005))
    return row


if __name__ == '__main__':
    rows = []
    print('A legacy ...');       rows.append(summarize(gen_legacy(), 'A_legacy'))
    print('B values_only ...');  rows.append(summarize(gen_values_only(), 'B_values_only'))
    print('C current ...');      rows.append(summarize(gen_current(), 'C_current'))
    df = pd.DataFrame(rows)
    write(df, 'TABLE_2a_ShapeParamAblation.csv')
    t = df.set_index('arm').T
    t['B_over_A'] = t['B_values_only'] / t['A_legacy']
    t['C_over_B'] = t['C_current'] / t['B_values_only']
    pd.set_option('display.width', 220)
    print(t.to_string(float_format=lambda v: f'{v:,.4f}'))
