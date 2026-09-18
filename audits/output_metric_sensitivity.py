"""Which pLCA OUTPUT is most sensitive to the choice of UQ method? Stage 2d.

The author's suggestion, 2026-09-17: plot the W1 between each pair of fitted
models against the difference that pair produces in every output metric, and see
which outputs move the most.

THE THING THAT MAKES THIS MORE THAN A CORRELATION. Every output also moves when
nothing changes at all, because the Monte Carlo draw is random. So a metric that
"changes a lot" between two methods may simply be a noisy metric. This measures
both, on the same footing:

    noise   the same method, same fitted models, two independent streams
    signal  two different methods, on COMMON random numbers so the streams
            contribute nothing

and reports the ratio. A metric with a high ratio discriminates between UQ
methods; one near 1.0 is telling you about the random number generator.

Both are expressed in units of the metric's own spread across materials, so
quantities on different scales can be compared.

    conda run -n compareuq python audits/output_metric_sensitivity.py [n_groups]
"""
import os
import sys
import time
import warnings
from itertools import combinations

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import corpus  # noqa: E402
import fitting  # noqa: E402
import flip as FL  # noqa: E402
import weighting as WG  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
SEED = 20260917
NECCS = 10_000
N_GROUPS = 250
FMT = lambda v: f'{v:.3f}'  # noqa: E731


def outputs(draws):
    """Every pLCA output this study reports, from one (neccs, nmat) sample.

    Returns a dict of name -> per-material vector. Each is a quantity notebook 3
    already computes; they are gathered here so they can be compared on equal
    terms rather than one at a time.
    """
    total = draws.sum(axis=1)
    order = (-draws).argsort(axis=1).argsort(axis=1)
    var_total = np.var(total)
    med = np.median(draws, axis=0)
    ui = np.array([1.0 - np.var(total - draws[:, j] + med[j]) / var_total
                   for j in range(draws.shape[1])])
    perc = draws / total[:, None]
    return {
        'eci_mean': draws.mean(axis=0),
        'eci_std': draws.std(axis=0),
        'eci_cov': draws.std(axis=0) / draws.mean(axis=0),
        'eci_perc_mean': perc.mean(axis=0),
        'eci_perc_std': perc.std(axis=0),
        'eci_rank_1': (order == 0).mean(axis=0),
        'eci_rank_4': (order == 3).mean(axis=0),
        'eci_meanrank': order.mean(axis=0) + 1.0,
        'eci_p95': np.quantile(draws, 0.95, axis=0),
        'ui': ui,
    }


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
    scales = {ds: WG.relative_scales(DATA[ds]['data']) for ds in need}
    dist = {ds: FL.pair_distances(models[ds], DATA[ds]['data'], scales[ds])
            for ds in need}
    print(f'{n_groups} pLCA groups, {NECCS:,} draws, seed {SEED}')

    t0 = time.time()
    noise, signal = [], []
    for gi, g in enumerate(combos):
        names = list(g)
        u1 = rng.random((NECCS, len(names)))
        u2 = rng.random((NECCS, len(names)))
        per = {}
        for m in fitting.PEWT:
            per[m] = outputs(np.column_stack([
                np.asarray(models[d][m].rvs_from_uniform(u1[:, j]), float)
                for j, d in enumerate(names)]))
        # NOISE: one method, a second independent set of variates
        ref = fitting.PEWT[0]
        alt = outputs(np.column_stack([
            np.asarray(models[d][ref].rvs_from_uniform(u2[:, j]), float)
            for j, d in enumerate(names)]))
        for k, v in per[ref].items():
            spread = np.ptp(v)
            noise.append(dict(metric=k, absdiff=float(np.abs(v - alt[k]).max()),
                              spread=float(spread)))
        # SIGNAL: every pair of methods, on the SAME variates
        for a, b in combinations(fitting.PEWT, 2):
            sep = float(np.max([dist[n][(a, b)]['rel_mean'] for n in names]))
            for k in per[a]:
                spread = np.ptp(per[a][k])
                signal.append(dict(metric=k, method_a=a, method_b=b, sep=sep,
                                   absdiff=float(np.abs(per[a][k] - per[b][k]).max()),
                                   spread=float(spread)))
    print(f'ran in {time.time() - t0:.0f}s')

    dn = pd.DataFrame(noise)
    ds = pd.DataFrame(signal)
    for d in (dn, ds):
        d['rel'] = d.absdiff / d.spread.replace(0, np.nan)
    dn.to_csv(os.path.join(TABLES, 'TABLE_OutputMetricNoise.csv'), index=False)
    ds.to_csv(os.path.join(TABLES, 'TABLE_OutputMetricSignal.csv.gz'),
              index=False)

    head('WHICH OUTPUT MOVES MOST WHEN THE UQ METHOD CHANGES?')
    print('Both columns are the median absolute change across the four')
    print('materials, divided by that metric\'s own spread across them.')
    print()
    summary = pd.DataFrame({
        'noise (same method)': dn.groupby('metric').rel.median(),
        'signal (diff method)': ds.groupby('metric').rel.median(),
    })
    summary['ratio'] = summary['signal (diff method)'] / summary['noise (same method)']
    summary['raw signal'] = ds.groupby('metric').absdiff.median()
    summary['raw noise'] = dn.groupby('metric').absdiff.median()
    summary = summary.sort_values('ratio', ascending=False)
    summary.to_csv(os.path.join(TABLES, 'TABLE_OutputMetricSensitivity.csv'))
    print(summary.to_string(float_format=FMT))
    print()
    print('A ratio near 1 means the metric moves as much between two runs of the')
    print('SAME method as between two different methods: it is reporting the')
    print('random number generator. A high ratio means the metric genuinely')
    print('discriminates between UQ methods.')

    head('HOW EACH OUTPUT SCALES WITH THE MODEL DISTANCE')
    print('Spearman correlation of the absolute change with the relative W1')
    print('between the two fitted models, and the slope in log-log.')
    rows = []
    for k, g in ds.groupby('metric'):
        ok = (g.sep > 0) & (g.absdiff > 0) & np.isfinite(g.rel)
        gg = g[ok]
        sp = gg[['sep', 'absdiff']].corr('spearman').iloc[0, 1]
        b = np.polyfit(np.log(gg.sep), np.log(gg.absdiff), 1)[0]
        rows.append(dict(metric=k, spearman=sp, loglog_slope=b, n=len(gg)))
    curve = pd.DataFrame(rows).sort_values('spearman', ascending=False)
    curve.to_csv(os.path.join(TABLES, 'TABLE_OutputMetricScaling.csv'),
                 index=False)
    print(curve.to_string(index=False, float_format=FMT))


def head(t):
    print()
    print('=' * 78)
    print(t)
    print('=' * 78)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else N_GROUPS)
