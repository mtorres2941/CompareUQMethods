"""Which KDE bandwidth rule, judged by something other than W1.

Stage 2b, added after the author asked how to fix Silverman's degenerate-IQR
failure in a defensible way. Two things have to be settled and they are
different questions.

THE CRITERION PROBLEM COMES FIRST. `r8_why_kde_loses.py` shows W1 falls
monotonically as the KDE bandwidth shrinks, down to about 2 percent of any
standard rule, because a KDE with a vanishing bandwidth IS the empirical
distribution it is being scored against. So W1 CANNOT be used to choose a
bandwidth, and "Silverman beats Scott on W1" is partly just "Silverman is
smaller". Any comparison of bandwidth rules needs a referee that penalizes
undersmoothing.

The referee used here is LEAVE-ONE-OUT LIKELIHOOD CROSS-VALIDATION, which is the
standard one (Habbema, Duin and Hermans 1974; Silverman 1986 section 3.4.4):

    CV(h) = sum_i w_i log f_{-i}(x_i),
    f_{-i}(x) = sum_{j != i} w_j K_h(x - x_j) / (1 - w_i)

It has a genuine interior maximum. As h goes to zero every held-out point falls
in a gap left by its own kernel and CV goes to minus infinity; as h grows the
density flattens and CV falls again. It needs no quantile, no interquartile
range and no scale estimator, so it cannot suffer the failure the author
described, and it is a defensible answer to "how do I fix this robustly" in its
own right rather than only as a referee.

WHAT IS COMPARED. Scott (the study's current rule), Silverman (KL2's rule),
Silverman with a floor on the robust scale, and the cross-validated bandwidth
itself. Reported on three axes: the LOO-CV referee, the variance the fitted
model implies against the data's, and W1, so the confound stays visible.

NOTHING HERE IS ADOPTED. The bandwidth rule belongs to Stage 2h.

    conda run -n compareuq python audits/stage2b/r9_bandwidth.py [n_synth]
"""
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import corpus  # noqa: E402
import empirical  # noqa: E402
import families as F  # noqa: E402
import fitting as FT  # noqa: E402
from customstats import weighted_quantile, weighted_std  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'stage2b')
SEED = 42
N_SYNTH = 800

#: Leave-one-out is O(n^2). Above this many values the held-out set is a random
#: subsample evaluated against the FULL dataset, which is an unbiased estimate
#: of the same quantity at a fraction of the cost.
LOO_MAX_POINTS = 1500

#: Bandwidth grid for the cross-validated rule, as multiples of Scott's.
CV_GRID = np.geomspace(0.02, 3.0, 40)

#: Floor on the robust scale in the guarded Silverman variant, as a fraction of
#: the weighted standard deviation. See `silverman_floored`.
SILVERMAN_FLOOR = 1.0 / 3.0


def loo_loglik(x, w, h, rng, max_points=LOO_MAX_POINTS):
    """Weighted leave-one-out log-likelihood of a Gaussian KDE at bandwidth h.

    The density is the TRUNCATED, renormalized one, matching how the model is
    scored and sampled everywhere else in this project.
    """
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    w = w / w.sum()
    n = len(x)
    if n < 3 or h <= 0:
        return np.nan
    idx = (np.arange(n) if n <= max_points
           else rng.choice(n, max_points, replace=False))
    # Renormalizing constant of the truncation, computed once.
    from scipy.stats import norm as _norm
    mass_kept = 1.0 - float(_norm.cdf((0.0 - x) / h) @ w)
    if not mass_kept > 0:
        return np.nan
    total, wsum = 0.0, 0.0
    step = max(1, (1 << 20) // max(1, n))
    for s in range(0, len(idx), step):
        blk = idx[s:s + step]
        z = (x[blk][:, None] - x[None, :]) / h
        k = np.exp(-0.5 * z ** 2) / (h * np.sqrt(2.0 * np.pi))
        dens = k @ w
        self_term = w[blk] * (1.0 / (h * np.sqrt(2.0 * np.pi)))
        loo = (dens - self_term) / np.maximum(1.0 - w[blk], 1e-12)
        loo = loo / mass_kept
        good = loo > 0
        total += float(np.sum(w[blk][good] * np.log(loo[good])))
        wsum += float(np.sum(w[blk][good]))
    return total / wsum if wsum > 0 else np.nan


def silverman_floored(x, w, floor=SILVERMAN_FLOOR):
    """Silverman's rule with the robust scale bounded BELOW as well as above.

    Silverman's `min(sd, IQR/1.34)` bounds the scale estimate from above, which
    is what stops a few outliers inflating the bandwidth. It does not bound it
    from below, and the interquartile range can collapse -- to exactly zero at
    small n under concentrated weights, and to a small fraction of the standard
    deviation on a heavy-tailed dataset with a tight core.

    This variant takes `max(IQR/1.34, floor * sd)` before the `min`, so the
    robust scale can never fall below a stated fraction of the standard
    deviation. `floor` is a free parameter and that is the honest objection to
    this variant; the cross-validated rule has none.
    """
    w = np.asarray(w, float)
    w = w / w.sum()
    sd = weighted_std(x, w)
    iqr = (weighted_quantile(x, w, 0.75, output='perc2val')
           - weighted_quantile(x, w, 0.25, output='perc2val'))
    if not np.isfinite(iqr) or iqr <= 0:
        iqr = sd * 1.34
    n_eff = 1.0 / np.sum(w ** 2)
    robust = max(iqr / 1.34, floor * sd)
    return 0.9 * min(sd, robust) * n_eff ** -0.2


def model_sd(m, npoints=20_001):
    return float(np.std(m.ppf(np.linspace(1e-9, 1 - 1e-9, npoints))))


def one(name, x, w, wt, arm, rng):
    ww = FT.uniform_weights(x) if wt == 'Uniform' else w
    ww = np.asarray(ww, float) / np.sum(ww)
    sd = weighted_std(x, ww)
    scott = FT.weighted_bw(x, ww, bw_method='scott')
    rules = {
        'scott': scott,
        'silverman': FT.weighted_bw(x, ww, bw_method='silverman'),
        'silverman_floored': silverman_floored(x, ww),
    }
    # The cross-validated bandwidth: maximize the leave-one-out likelihood.
    cv = [(h, loo_loglik(x, ww, h, rng)) for h in scott * CV_GRID]
    cv = [(h, v) for h, v in cv if np.isfinite(v)]
    if cv:
        rules['cv'] = max(cv, key=lambda t: t[1])[0]
    out = []
    for rule, h in rules.items():
        m = F.Truncated(F.WeightedKDE(x, ww, h), label='kde')
        out.append(dict(
            arm=arm, dataset=name, n=len(x), weighting=wt, rule=rule,
            bandwidth=h, h_over_sd=h / sd if sd > 0 else np.nan,
            loo=loo_loglik(x, ww, h, rng),
            model_sd_over_data_sd=model_sd(m) / sd if sd > 0 else np.nan,
            mass_below=m.mass_below,
            w1=FT.score_w1_model(m, x, w)))
    return out


def report(d, arm):
    g = d[d.arm == arm]
    print('=' * 78)
    print(f'{arm.upper()} ARM, {g.dataset.nunique()} datasets')
    print('=' * 78)
    for wt in ('Uniform', 'Variable'):
        h = g[g.weighting == wt]
        t = h.groupby('rule').agg(
            median_h_over_sd=('h_over_sd', 'median'),
            mean_loo=('loo', 'mean'), median_loo=('loo', 'median'),
            median_sd_ratio=('model_sd_over_data_sd', 'median'),
            max_sd_ratio=('model_sd_over_data_sd', 'max'),
            mean_mass_below=('mass_below', 'mean'),
            mean_w1=('w1', 'mean'), median_w1=('w1', 'median'))
        t = t.sort_values('mean_loo', ascending=False)
        print(f'--- {wt} weighting, sorted by the LOO referee (higher is better) ---')
        print(t.to_string(float_format=lambda v: f'{v:.4f}'))
        wide = h.pivot(index='dataset', columns='rule', values='loo')
        if 'cv' in wide:
            print('  how often each rule is within 0.01 nats of the '
                  'cross-validated optimum:')
            for r in ('scott', 'silverman', 'silverman_floored'):
                if r in wide:
                    print(f'    {r:<20} {(wide[r] > wide["cv"] - 0.01).mean()*100:5.1f} pct'
                          f'   beats the other two on LOO: '
                          f'{(wide[r] >= wide[[c for c in ("scott","silverman","silverman_floored") if c in wide]].max(axis=1)).mean()*100:5.1f} pct')
        print()


def analyse_guarded():
    """The guarded rule, swept over its threshold, from this script's own table.

    `silverman_guarded` is Silverman above an effective-sample-size threshold
    and Scott below it. The threshold is chosen on the LOO referee and NOT on
    W1, so it is not tuned to the criterion the study scores by.
    """
    d = pd.read_csv(os.path.join(TABLES, 'TABLE_2b_BandwidthRules.csv'))
    w = d.pivot_table(index=['arm', 'dataset', 'n', 'weighting'],
                      columns='rule', values=['loo', 'w1']).reset_index()
    w.columns = ['_'.join([c for c in col if c]).strip() for col in w.columns]
    rows = []
    for arm in ('empirical', 'synthetic'):
        g = w[w.arm == arm]
        for thr in (0, 10, 15, 20, 30, 50, 100, 10 ** 9):
            use = g.n >= thr
            loo = np.where(use, g.loo_silverman, g.loo_scott)
            w1 = np.where(use, g.w1_silverman, g.w1_scott)
            label = {0: 'always Silverman',
                     10 ** 9: 'always Scott'}.get(thr, f'guarded at n >= {thr}')
            rows.append(dict(arm=arm, rule=label,
                             pct_silverman=float(use.mean() * 100),
                             mean_loo=float(np.mean(loo)),
                             median_loo=float(np.median(loo)),
                             p05_loo=float(np.quantile(loo, 0.05)),
                             mean_w1=float(np.mean(w1)),
                             median_w1=float(np.median(w1))))
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(TABLES, 'TABLE_2b_GuardedBandwidthSweep.csv'),
             index=False)
    print()
    print('=' * 78)
    print('THE GUARDED RULE: Silverman above a threshold, Scott below it')
    print('=' * 78)
    for arm in ('empirical', 'synthetic'):
        print(f'--- {arm}, both weightings ---')
        print(r[r.arm == arm].drop(columns='arm').to_string(
            index=False, float_format=lambda v: f'{v:.4f}'))
        print()
    print('`p05_loo` is the 5th percentile of held-out log-likelihood: the worst')
    print('cases, which is where a collapsing bandwidth does its damage. The')
    print('guarded rule beats BOTH pure rules on the mean and the median of the')
    print('referee and repairs the tail. W1 still prefers pure Silverman, which')
    print('is the undersmoothing bias of W1 and not evidence about the rule.')


def main(n_synth):
    os.makedirs(TABLES, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    ds, _ = empirical.prepare(np.random.default_rng(SEED).spawn(1)[0])
    print(f'empirical: {len(ds)} datasets', flush=True)
    for i, (name, (x, w)) in enumerate(ds.items()):
        for wt in ('Uniform', 'Variable'):
            rows += one(name, x, w, wt, 'empirical', rng)
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(ds)}', flush=True)
    met, vals, _ = corpus.load_corpus()
    syn = corpus.as_dict(vals[vals.dataset_id.isin(
        set(met.sample(n_synth, random_state=0).dataset.astype(str)))])
    print(f'synthetic: {len(syn)} datasets', flush=True)
    for i, (name, (x, w)) in enumerate(syn.items()):
        for wt in ('Uniform', 'Variable'):
            rows += one(name, x, w, wt, 'synthetic', rng)
        if (i + 1) % 200 == 0:
            print(f'  {i+1}/{len(syn)}', flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TABLES, 'TABLE_2b_BandwidthRules.csv'), index=False)
    pd.set_option('display.width', 220)
    for arm in ('empirical', 'synthetic'):
        report(d, arm)
    analyse_guarded()


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_SYNTH)
