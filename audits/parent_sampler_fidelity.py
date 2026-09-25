"""Does the truth run's sampler reproduce the parent it stands for?

**WHY THIS EXISTS.** Stage 2h widened the generator's truncation, regenerated
the corpus, and got a truth run reporting 99.98 percent errors. The
configuration was rejected. It was rejected on the wrong evidence: the parents
were sound -- mean 1.012, median 0.773, the 1 - 1e-6 quantile at 33 -- and what
had broken was `plca.ParentSampler`, which tabulated the parent's CDF on a
LINEARLY spaced grid between truncation bounds eight orders of magnitude apart.
Every check that stage ran looked at the generator or at the sample. Nothing
looked at the object the truth run actually draws from.

So this is the check that was missing. It takes a candidate configuration,
draws parents under it, and compares `ParentSampler`'s interpolated inverse CDF
against the parent's own exact bisection at probabilities spanning the range a
10,000-draw Monte Carlo reaches. A candidate that fails here cannot be used
whatever its calibration score says, and -- this is the half that cost a day --
a candidate that PASSES here has not been cleared by its calibration score
either. The two are independent gates.

WHAT IT PRINTS, per configuration:

    sampler error     the relative difference between the sampler's quantile
                      and the parent's own, at the median, the tails, and the
                      worst point of the sweep
    drawn moments     the mean of 10,000 draws through the sampler against the
                      parent's own mean, which is the end-to-end statement
    parent shape      mean, median, their ratio, and the far quantile, so a
                      genuinely tail-dominated parent is distinguishable from
                      a sampler artifact
    bound multiplier  (1 + 1/min_q1_over_iqr) ** trunc_iqr_mult, the width the
                      truncation rule can reach, which is what stresses the
                      grid

    conda run -n compareuq python audits/parent_sampler_fidelity.py
    conda run -n compareuq python audits/parent_sampler_fidelity.py \
        --set min_q1_over_iqr=0.1 --set trunc_iqr_mult=2.0 --label "floor 0.1, mult 2"
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import genconfig as G                # noqa: E402
import generator as GEN              # noqa: E402
import plca as P                     # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')

#: Probabilities the check is made at. The extremes are the reach of the
#: study's own Monte Carlo: at 10,000 draws per material the expected number
#: of values beyond 1e-4 at either end is one, and `fitting.W1_TAIL_QUANTILE`
#: bounds the scoring criterion at 1 - 1e-6 for the same reason (decision 87).
PROBS = np.array([1e-6, 1e-4, 1e-3, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9,
                  0.99, 0.999, 1 - 1e-4, 1 - 1e-6])

#: Sizes to draw parents at. The truncation bound is multiplicative in the
#: data's own log spread, so it is widest where the parent is widest, and
#: dataset size drives how much spread a parent is asked for.
SIZES = (8, 40, 300, 3000)


def one_parent(cfg, n, rng):
    """A parent under `cfg`, or None if every draw was rejected."""
    for _ in range(cfg.max_parent_retries):
        parent, record = GEN.draw_parent(cfg, n, rng)
        if parent is not None:
            return parent, record
        if record.get('status') not in GEN.REDRAWABLE:
            return None, record
    return None, record


def exact_mean(parent, scheme):
    """The truncated mixture's own mean under `scheme`, in RAW units.

    `mixture.truncated_moments` computes this, and only under the SAMPLING
    weights, so it cannot answer for the market-weighted parent the truth run
    uses. The construction here is the same one and is the one decision 42
    settled: integrate the survival function on the COMPONENTS' OWN QUANTILES,
    never on a uniform grid, because a uniform grid over a wide truncation
    lands almost no nodes on the body and silently returns nonsense.

    It is deliberately a DIFFERENT node set from `ParentSampler`'s, which is
    what makes it an independent check rather than a restatement.
    """
    probs = np.linspace(0.0, 1.0, max(4001 // max(len(parent.comps), 1), 64))
    nodes = [np.asarray(d.ppf(np.clip(probs, 1e-12, 1 - 1e-12)), float)
             for d in parent.comps]
    x = np.unique(np.concatenate(nodes + [np.array([parent.lo, parent.hi])]))
    x = x[(x >= parent.lo) & (x <= parent.hi)]
    if len(x) < 8 or not np.all(np.isfinite(x)):
        return float('nan')
    F = np.asarray(parent.cdf(x, scheme), dtype=float)
    return float(parent.lo + np.trapezoid(1.0 - F, x - parent.lo))


def check_parent(parent, scheme, n_draws, rng):
    """Sampler against exact parent, on one parent. Returns a dict of scalars.

    The comparison is RELATIVE and taken in the normalized units the datasets
    live in, because that is the scale everything downstream is read on: the
    parent's `ppf` returns raw units and `ParentSampler` divides by the
    realized unweighted sample mean, so the two must be put on one scale
    before they can be differenced at all. Getting that wrong would report a
    faultless sampler as broken by exactly the normalizer.
    """
    c = float(parent.normalizer)
    sampler = P.ParentSampler(parent, scheme=scheme)
    exact = np.ravel(parent.ppf(PROBS, scheme)).astype(float) / c
    approx = np.asarray(sampler.ppf(PROBS), dtype=float)
    scale = np.maximum(np.abs(exact), 1e-12)
    rel = np.abs(approx - exact) / scale

    u = rng.random(n_draws)
    drawn = np.asarray(sampler.ppf(u), dtype=float)
    # The parent's own mean, in the same normalized units.
    pmean = exact_mean(parent, scheme) / c
    med = float(np.ravel(parent.ppf(0.5, scheme))[0]) / c
    far = float(np.ravel(parent.ppf(1 - 1e-6, scheme))[0]) / c
    return dict(
        rel_median=float(rel[PROBS == 0.5][0]),
        rel_p1e4=float(rel[PROBS == 1e-4][0]),
        rel_p1m1e4=float(rel[np.isclose(PROBS, 1 - 1e-4)][0]),
        rel_max=float(rel.max()),
        rel_max_at=float(PROBS[int(np.argmax(rel))]),
        drawn_mean=float(drawn.mean()),
        drawn_max=float(drawn.max()),
        parent_mean=pmean,
        parent_median=med,
        mean_over_median=(pmean / med) if med > 0 else float('inf'),
        parent_far_q=far,
        lo=float(sampler.lo), hi=float(sampler.hi),
    )


def sweep(cfg, n_parents, n_draws, seed):
    rows = []
    # A FIXED index per scheme, never `hash(scheme)`: Python randomizes string
    # hashes per process, so that would give a different stream on every run
    # and the project requires every draw to come from an explicitly seeded
    # Generator.
    for si, scheme in enumerate((P.TRUTH_SCHEME, P.TRUTH_SCHEME_SAMPLING)):
        for n in SIZES:
            rng = np.random.default_rng([seed, n, si])
            got = 0
            for _ in range(n_parents * 4):
                if got >= n_parents:
                    break
                parent, record = one_parent(cfg, n, rng)
                if parent is None:
                    continue
                got += 1
                row = check_parent(parent, scheme, n_draws, rng)
                row.update(scheme=scheme, n=n)
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', action='append', default=[], metavar='K=V')
    ap.add_argument('--label', default='')
    ap.add_argument('--parents', type=int, default=40,
                    help='parents per (scheme, size) cell')
    ap.add_argument('--draws', type=int, default=10_000,
                    help='draws per parent, matching the study pLCA')
    ap.add_argument('--seed', type=int, default=20260925)
    args = ap.parse_args()

    cfg = G.DEFAULT
    changed = {}
    for item in args.set:
        k, v = item.split('=', 1)
        changed[k] = float(v) if '.' in v or 'e' in v.lower() else int(v)
    if changed:
        cfg = cfg.replace(**changed)
    label = args.label or (', '.join(f'{k}={v}' for k, v in changed.items())
                           or 'current default')

    mult = (1.0 + 1.0 / cfg.min_q1_over_iqr) ** cfg.trunc_iqr_mult
    print('=' * 78)
    print(f'DOES THE TRUTH RUN SAMPLE THE PARENT IT STANDS FOR -- "{label}"')
    print('=' * 78)
    print(f'  min_q1_over_iqr {cfg.min_q1_over_iqr}   '
          f'trunc_iqr_mult {cfg.trunc_iqr_mult}   '
          f'cv_log10_mean {cfg.cv_log10_mean}')
    print(f'  truncation bound multiplier  '
          f'(1 + 1/{cfg.min_q1_over_iqr}) ** {cfg.trunc_iqr_mult}  =  '
          f'{mult:,.0f}')
    print(f'  {args.parents} parents per cell, {len(SIZES)} sizes, both '
          f'weighting schemes, {args.draws:,} draws each', flush=True)

    df = sweep(cfg, args.parents, args.draws, args.seed)
    df['label'] = label

    print()
    print('SAMPLER AGAINST THE PARENT, relative error in the quantile')
    print('  (0.001 is a tenth of a percent; the failed Stage 2h corpus was 1e4)')
    agg = df.groupby(['scheme', 'n']).agg(
        median=('rel_median', 'median'),
        p1e4=('rel_p1e4', 'median'),
        p1m1e4=('rel_p1m1e4', 'median'),
        worst=('rel_max', 'max'),
        worst_at=('rel_max_at', 'median'),
    )
    print(agg.to_string(float_format=lambda v: f'{v:.3g}'))

    print()
    print('THE PARENT ITSELF, and the end-to-end draw through the sampler')
    shape = df.groupby(['scheme', 'n']).agg(
        parent_mean=('parent_mean', 'median'),
        drawn_mean=('drawn_mean', 'median'),
        mean_over_median=('mean_over_median', 'median'),
        worst_mean_over_median=('mean_over_median', 'max'),
        far_q=('parent_far_q', 'median'),
        hi=('hi', 'median'),
    )
    print(shape.to_string(float_format=lambda v: f'{v:.4g}'))

    worst = df.loc[df['rel_max'].idxmax()]
    print()
    print('THE VERDICT, which is a threshold and not a judgment call:')
    print(f'  worst quantile error anywhere   {worst["rel_max"]:.3g} '
          f'at p = {worst["rel_max_at"]:g}, n = {int(worst["n"])}, '
          f'{worst["scheme"]}')
    print(f'  worst drawn mean                {df["drawn_mean"].max():.4g} '
          f'against a parent mean of {df["parent_mean"].max():.4g}')
    print(f'  worst mean over median          '
          f'{df["mean_over_median"].max():.4g} '
          f'(guard rejects above {cfg.max_parent_mean_over_median})')
    ok = df['rel_max'].max() < 1e-2
    print(f'  PASSES: {ok}   (the bar is 1 percent on every quantile of every '
          f'parent)')

    os.makedirs(TABLES, exist_ok=True)
    path = os.path.join(TABLES, 'TABLE_ParentSamplerFidelity.csv')
    header = not os.path.exists(path)
    df.to_csv(path, mode='a', header=header, index=False)
    print(f'\nappended to {os.path.relpath(path, ROOT)}')


if __name__ == '__main__':
    main()
