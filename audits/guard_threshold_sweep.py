"""Can the Silverman guard be set better than SILVERMAN_MIN_NEFF = 30?

The author's question on Stage 2c section 4.8: the parent referee says the
guarded rule loses to pure Silverman, so rather than leaving the guard alone,
adjust its threshold. The guard exists for a real reason -- at small n the
quartiles are interpolated between two order statistics and the robust scale
collapses -- so the question is whether 30 is the right place to switch, not
whether to switch at all.

THE THRESHOLD WAS CHOSEN ON ONE CRITERION AND IS NOW BEING JUDGED ON ANOTHER.
Decision 54 picked 30 on leave-one-out likelihood, a DENSITY criterion. The
synthetic parent gives W1, a CDF criterion, a target that is not the training
data. The two want different bandwidths, because the empirical CDF is already
root-n consistent and smoothing buys a CDF criterion very little. So a threshold
that is good on both is worth more than one that is optimal on either.

This sweeps the threshold on BOTH criteria at once and reports where they agree.

    conda run -n compareuq python audits/guard_threshold_sweep.py [n_synth]
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
import customstats as CS  # noqa: E402
import empirical  # noqa: E402
import families as F  # noqa: E402
import fitting as FT  # noqa: E402
import mixture as M  # noqa: E402
import recovery as R  # noqa: E402
from bandwidth_rules import loo_loglik  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
SEED = 42
N_SYNTH = 1_200

#: Thresholds on the Kish effective sample size. 0 is pure Silverman, a very
#: large value is pure Scott.
THRESHOLDS = (0, 3, 5, 8, 10, 12, 15, 18, 20, 22, 25, 30, 40, 50,
              75, 100, 200, 10 ** 9)


def kish(w):
    w = np.asarray(w, float)
    w = w / w.sum()
    return float(1.0 / np.sum(w ** 2))


def one(name, x, w_var, parent, rng):
    """Both bandwidths and both criteria, once per (dataset, weighting)."""
    out = []
    for wt, ww in (('Uniform', FT.uniform_weights(x)), ('Variable', w_var)):
        ww = np.asarray(ww, float) / np.sum(ww)
        n_eff = kish(ww)
        row = dict(dataset=name, n=len(x), weighting=wt, n_eff=n_eff)
        for rule in ('scott', 'silverman'):
            h = CS.weighted_bw(x, ww, bw_method=rule)
            m = F.Truncated(F.WeightedKDE(x, ww, h), label='kde')
            row[f'h_{rule}'] = h
            row[f'loo_{rule}'] = loo_loglik(x, ww, h, rng)
            if parent is not None:
                grid = R.recovery_grid(x, w_var, parent)
                row[f'parent_{rule}'] = R.w1_against_parent(
                    m, parent, R.PARENT_SCHEME[wt], grid)
        out.append(row)
    return out


def sweep(d, arm):
    """Apply each threshold to the per-dataset table and score the result."""
    rows = []
    for thr in THRESHOLDS:
        use_silverman = d.n_eff >= thr
        label = {0: 'pure Silverman',
                 10 ** 9: 'pure Scott'}.get(thr, f'guard at n_eff >= {thr}')
        r = dict(arm=arm, threshold=thr, rule=label,
                 pct_silverman=float(use_silverman.mean() * 100))
        loo = np.where(use_silverman, d.loo_silverman, d.loo_scott)
        r['mean_loo'] = float(np.nanmean(loo))
        r['p05_loo'] = float(np.nanquantile(loo, 0.05))
        if 'parent_silverman' in d:
            par = np.where(use_silverman, d.parent_silverman, d.parent_scott)
            r['mean_parent'] = float(np.nanmean(par))
            r['median_parent'] = float(np.nanmedian(par))
            r['p90_parent'] = float(np.nanquantile(par, 0.90))
        rows.append(r)
    return pd.DataFrame(rows)


def knee(out):
    """Where does buying parent accuracy start costing held-out likelihood?

    The two criteria disagree, so there is no optimum, only a trade, and the
    question "why this threshold and not a smaller one" has to be answered by
    the shape of the trade rather than by picking a round number.

    The diagnostic is the MARGINAL rate between consecutive thresholds: stepping
    down from one to the next, how much W1-against-parent does it buy per unit of
    held-out likelihood it gives up. A marginal ratio far above 1 means the step
    is nearly free. The first step below 1 is where each further reduction costs
    more than it buys, and that is the stopping point.

    The mean held-out likelihood is the cost used, not the p05: the p05 on 147
    datasets is an order statistic that steps between discrete values and is too
    lumpy to differentiate. It is printed beside it.
    """
    e = out[out.arm == 'empirical'].set_index('threshold')
    s = out[out.arm == 'synthetic'].set_index('threshold')
    thrs = sorted(t for t in e.index if t < 10 ** 9)[::-1]   # high to low
    rows = []
    for hi, lo in zip(thrs[:-1], thrs[1:]):
        gain = 100 * (s.loc[hi, 'mean_parent'] - s.loc[lo, 'mean_parent']) \
            / s.loc[hi, 'mean_parent']
        cost = 100 * (e.loc[hi, 'mean_loo'] - e.loc[lo, 'mean_loo']) \
            / abs(e.loc[hi, 'mean_loo'])
        rows.append(dict(step=f'{hi} -> {lo}', parent_gain_pct=gain,
                         loo_mean_cost_pct=cost,
                         marginal_ratio=(gain / cost) if cost > 1e-6
                         else float('inf'),
                         loo_p05=e.loc[lo, 'p05_loo']))
    k = pd.DataFrame(rows)
    print()
    print('=' * 78)
    print('WHY THIS THRESHOLD AND NOT A SMALLER ONE: the MARGINAL trade')
    print('=' * 78)
    print(k.to_string(index=False, float_format=lambda v: f'{v:9.3f}'))
    print()
    good = k[k.marginal_ratio > 1]
    if len(good):
        print(f'Steps that buy more than they cost: {list(good.step)}')
    print('Every step below that costs more held-out likelihood than it buys in')
    print('parent accuracy, so it is where the reduction stops being free.')


def main(n_synth=N_SYNTH):
    os.makedirs(TABLES, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(0)
    emp, _ = empirical.prepare(np.random.default_rng(SEED).spawn(1)[0])
    met, vals, _ = corpus.load_corpus()
    ids = sorted(met.sample(n_synth, random_state=0).dataset.astype(str))
    syn = corpus.as_dict(vals[vals.dataset_id.isin(set(ids))])
    specs = corpus.load_parent_specs()

    rows = []
    for i, (name, (x, w)) in enumerate(emp.items()):
        rows += one(name, np.asarray(x, float), np.asarray(w, float), None, rng)
        if (i + 1) % 50 == 0:
            print(f'  empirical {i+1}/{len(emp)}  {time.time()-t0:.0f}s',
                  flush=True)
    emp_d = pd.DataFrame(rows)

    rows = []
    for i, name in enumerate(ids):
        x, w = syn[name]
        rows += one(name, np.asarray(x, float), np.asarray(w, float),
                    M.parent_from_spec(specs[name]), rng)
        if (i + 1) % 200 == 0:
            print(f'  synthetic {i+1}/{len(ids)}  {time.time()-t0:.0f}s',
                  flush=True)
    syn_d = pd.DataFrame(rows)

    out = pd.concat([sweep(emp_d, 'empirical'), sweep(syn_d, 'synthetic')],
                    ignore_index=True)
    out.to_csv(os.path.join(TABLES, 'TABLE_GuardThresholdSweep.csv'),
               index=False)
    pd.concat([emp_d.assign(arm='empirical'), syn_d.assign(arm='synthetic')],
              ignore_index=True).to_csv(
        os.path.join(TABLES, 'TABLE_GuardThresholdPerDataset.csv'), index=False)

    knee(out)
    pd.set_option('display.width', 220)
    print()
    print('=' * 78)
    print('THE GUARD THRESHOLD ON BOTH CRITERIA')
    print('=' * 78)
    print('`mean_loo` and `p05_loo`: held-out log density, HIGHER is better, and')
    print('p05 is the worst cases the guard exists to repair.')
    print('`mean_parent`: W1 against the known parent, LOWER is better.')
    for arm in ('empirical', 'synthetic'):
        print(f'--- {arm} ---')
        cols = ['rule', 'pct_silverman', 'mean_loo', 'p05_loo']
        if arm == 'synthetic':
            cols += ['mean_parent', 'median_parent', 'p90_parent']
        print(out[out.arm == arm][cols].to_string(
            index=False, float_format=lambda v: f'{v:.4f}'))
        print()
    print('WHERE DOES THE EFFECTIVE SAMPLE SIZE ACTUALLY SIT? The guard can only')
    print('bind where n_eff is below the threshold.')
    both = pd.concat([emp_d.assign(arm='empirical'),
                      syn_d.assign(arm='synthetic')], ignore_index=True)
    both['band'] = both.n.map(R.size_band)
    print(both.groupby(['arm', 'weighting']).n_eff.describe(
        percentiles=[0.05, 0.25, 0.5, 0.75]).to_string(
        float_format=lambda v: f'{v:.1f}'))
    print()
    print('n_eff median by size band:')
    print(both.groupby(['arm', 'band', 'weighting']).n_eff.median().unstack()
          .to_string(float_format=lambda v: f'{v:.1f}'))


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_SYNTH)
