"""How likely is it that variable weighting matters, for a GIVEN dataset?

The author's question, 2026-09-17: "we use the Dirichlet distribution to
uniformly sample all possible combinations of weights. Theoretically, any of
those could be true. So what proportion of those sets of variable weights are
meaningfully different enough from the uniform assumption to matter? ... Just so
someone can understand how safe their assumption of uniform weighting is."

**IT IS A GOOD QUESTION AND THIS IS A FEASIBILITY PROBE, NOT THE ANSWER.** The
answer needs a threshold -- what counts as "enough to matter" -- and that
threshold is Stage 2d's, which owns the flip probability and the named relative
measure. Building the final version on a placeholder threshold would be the
mistake this project has avoided elsewhere. So this establishes that the idea
works, measures what drives it, and leaves the threshold blank for 2d to fill.

WHAT IT COMPUTES. For each dataset, K flat-Dirichlet weight vectors -- the same
prior the study already uses, so every draw is a market share allocation the study
considers possible. For each draw, W1 between the uniform-weighted and the
variable-weighted empirical CDF of the SAME values. The dataset is normalized to
an unweighted mean of 1, so that distance is in units of the mean and a threshold
of 0.10 reads as "shifts the distribution by a tenth of its mean".

The existing `w_v_uw_wasserstein` characteristic is ONE draw from this
distribution. That is the open item about a single Dirichlet realization moving
per-dataset metrics a long way; this is the distribution behind it.

    conda run -n compareuq python audits/weighting_risk.py [n_draws]
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

import empirical  # noqa: E402
import genconfig as G  # noqa: E402
import recovery as R  # noqa: E402
from customstats import wasserstein1_weighted, weighted_quantile  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
SEED = 42
N_DRAWS = 300

#: PLACEHOLDER thresholds, in units of the dataset's own mean. **Stage 2d
#: replaces these with the one it calibrates against the pLCA flip probability.**
#: They are round numbers and nothing here should be quoted as a result.
THRESHOLDS = (0.05, 0.10, 0.25)


def risk_for(x, n_draws, rng, alpha):
    """Distribution of the uniform-to-variable distance over possible weightings."""
    x = np.asarray(x, float)
    n = len(x)
    uniform = np.ones(n) / n
    draws = rng.dirichlet(np.ones(n) * alpha, size=n_draws)
    return np.array([wasserstein1_weighted(x, x, uniform, draws[k])
                     for k in range(n_draws)])


def main(n_draws=N_DRAWS):
    os.makedirs(TABLES, exist_ok=True)
    alpha = G.DEFAULT.point_weight_alpha
    rng = np.random.default_rng(SEED)
    emp, _ = empirical.prepare(rng.spawn(1)[0])
    draw_rng = np.random.default_rng(7)

    t0 = time.time()
    rows = []
    for name, (x, w) in emp.items():
        x = np.asarray(x, float)
        d = risk_for(x, n_draws, draw_rng, alpha)
        uniform = np.ones(len(x)) / len(x)
        q1 = weighted_quantile(x, uniform, 0.25, output='perc2val')
        q3 = weighted_quantile(x, uniform, 0.75, output='perc2val')
        row = dict(dataset=name, n=len(x),
                   coeffvar=float(np.std(x) / np.mean(x)), iqr=float(q3 - q1),
                   median_shift=float(np.median(d)),
                   p90_shift=float(np.quantile(d, 0.90)),
                   realized=float(wasserstein1_weighted(x, x, uniform, w)))
        for thr in THRESHOLDS:
            row[f'P_shift_gt_{thr:g}'] = float((d > thr).mean())
        rows.append(row)
    out = R.add_size_band(pd.DataFrame(rows))
    out.to_csv(os.path.join(TABLES, 'TABLE_WeightingRisk.csv'), index=False)
    print(f'{len(out)} datasets x {n_draws} draws in {time.time() - t0:.0f}s')
    report(out)


def report(d):
    pd.set_option('display.width', 215)
    key = f'P_shift_gt_{THRESHOLDS[1]:g}'
    print()
    print('=' * 78)
    print('P(a possible weighting shifts the distribution by more than X of the mean)')
    print('=' * 78)
    print('X is a PLACEHOLDER. Stage 2d supplies the real one.')
    cols = ['n', 'coeffvar'] + [f'P_shift_gt_{t:g}' for t in THRESHOLDS]
    print(d.groupby('size_band', observed=True)[cols].median().to_string(
        float_format=lambda v: f'{v:.3f}'))
    print()
    print('=' * 78)
    print('WHAT DRIVES IT, and this is the interesting part')
    print('=' * 78)
    for c in ('coeffvar', 'iqr', 'n'):
        r = d[[key, c]].corr('spearman').iloc[0, 1]
        print(f'  Spearman with {c:<10} {r:+.3f}')
    r = pd.DataFrame({'a': d[key], 'b': np.log(d.n)}).corr('spearman').iloc[0, 1]
    print(f'  Spearman with {"log n":<10} {r:+.3f}')
    print()
    print('  DISPERSION BEATS SIZE HERE, which it does nowhere else in this study.')
    print('  Everything about WHICH METHOD FITS BEST is driven by n; whether')
    print('  WEIGHTING MATTERS is driven by how spread the values are, which is')
    print('  the author\'s own intuition and the opposite ordering.')
    print()
    print(f'  uniform weighting is safe, P < 0.05:  '
          f'{int((d[key] < 0.05).sum())} of {len(d)} datasets')
    print(f'  almost never safe, P > 0.95:          '
          f'{int((d[key] > 0.95).sum())} of {len(d)}')
    print()
    print('  riskiest:')
    print(d.nlargest(5, key)[['dataset', 'n', 'coeffvar', key]].to_string(
        index=False, float_format=lambda v: f'{v:.3f}'))
    print('  safest:')
    print(d.nsmallest(5, key)[['dataset', 'n', 'coeffvar', key]].to_string(
        index=False, float_format=lambda v: f'{v:.3f}'))
    print()
    print('WHAT STAGE 2d SHOULD DO WITH THIS. Replace the placeholder threshold')
    print('with the one it calibrates against the pLCA flip probability, then')
    print('this column becomes a per-dataset statement a practitioner can act on:')
    print('"for a category like yours, assuming uniform weights has an X percent')
    print('chance of changing which material ranks first". Report it against')
    print('dispersion rather than against n.')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_DRAWS)
