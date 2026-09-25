"""Did the weight rule unlock the widening, or is the improvement a coincidence?

**THE CLAIM THIS EXISTS TO TEST.** Three stages found the same structural trade
-- decision 39 in Stage 2a-2, decision 138 in Stage 2f, decision 170 in the
Stage 2g review -- that anything widening the corpus's dispersion also widens
its uniform-to-variable Wasserstein distance past the real arm's, which is the
paper's headline quantity. 36 configurations on three different levers, and the
sign never flipped.

After Stage 2h ported the synthetic arm's mode-coupled weight rule to the
empirical arm, bounded widening candidates IMPROVE both at once. Decision 170's
closing paragraph predicted exactly that -- "breaking the trade means changing
the WEIGHT model at the same time as the shape model" -- but a prediction being
confirmed is not the same as a mechanism being shown.

**THE EXPERIMENT.** Score the SAME synthetic samples against the empirical arm
built two ways: under the OLD rule, a flat Dirichlet over every declaration,
and under the NEW one, contiguous groups at a coherence of 0.5. If the weight
rule is the mechanism, the trade reappears against the old arm and vanishes
against the new one, with the synthetic side held fixed. If the improvement is
a property of the candidates themselves, both arms show it.

The synthetic draw is shared between the two arms at each (config, seed), so
the two columns differ ONLY in how the real categories were weighted.

    conda run -n compareuq python audits/widening_and_weights.py
"""
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, HERE)

import coverage                      # noqa: E402
import empirical                     # noqa: E402
import genconfig as G                # noqa: E402
import modality as MD                # noqa: E402
import tune_configuration as TC      # noqa: E402
import weighting                     # noqa: E402
from customstats import empirical_metadata   # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')

#: The candidates. `base` is the shipped configuration; the other two are the
#: bounded widening candidates whose truncation bound multiplier,
#: `(1 + 1/floor) ** mult`, is 121 and 64 rather than the 345 million of the
#: configuration Stage 2h rejected.
CONFIGS = {
    'base': {},
    'f0.2 m3.0 c0.329': dict(min_q1_over_iqr=0.2, trunc_iqr_mult=3.0,
                             cv_log10_mean=0.329),
    'f0.1 m2.0 c0.329': dict(min_q1_over_iqr=0.1, trunc_iqr_mult=2.0,
                             cv_log10_mean=0.329),
}
SEEDS = (42, 43, 44)


def arm(flat, rho=None):
    """The empirical arm's characteristics under one weight rule.

    `flat=True` forces the OLD rule. `coherent_weights` with `k` equal to the
    number of declarations is a flat Dirichlet over every declaration -- one
    group per point, so the contiguous cut has nothing to cut and the coherence
    parameter cannot act. That equivalence is asserted by
    `tests/test_weighting.py`, which is why this is a faithful reconstruction
    of the old rule rather than an approximation of it.

    Otherwise the ported rule is used at coherence `rho`. Sweeping rho here as
    well as flipping the rule answers the question a reviewer asks next: how
    much of the result depends on the ONE free parameter of the weight model,
    as against the change of rule itself. rho = 0 is the ported rule's own
    null -- contiguous groups whose membership is random -- so the gap between
    it and the flat draw is what the GROUPING costs, and the gap between it and
    0.5 is what the COHERENCE costs.

    The substitution is confined to this function and restored on the way out,
    so nothing else in the process sees a patched module.
    """
    real = weighting.coherent_weights
    if flat:
        def forced(values, rng, k=None, **kw):
            return real(values, rng, k=len(np.asarray(values)), **kw)
        weighting.coherent_weights = forced
    try:
        kw = {} if flat else dict(rho=rho)
        ds, _ = empirical.prepare(np.random.default_rng(TC.SEED).spawn(1)[0],
                                  **kw)
    finally:
        weighting.coherent_weights = real
    met = pd.DataFrame({m: empirical_metadata(x, w)
                        for m, (x, w) in ds.items()}).T.astype(float)
    met.index.name = 'material'
    rng = np.random.default_rng(0)
    modes = np.array([MD.n_modes_silverman(x, rng=rng, nboot=100)
                      for x, _ in ds.values()])
    vis = np.array([MD.n_modes_visible(x) for x, _ in ds.values()])
    return met.reset_index(), modes, vis


def main():
    os.makedirs(TABLES, exist_ok=True)
    pd.set_option('display.width', 200)

    print('building the empirical arm under BOTH weight rules ...', flush=True)
    arms = {'old: flat Dirichlet': arm(flat=True),
            'ported, rho=0.00': arm(flat=False, rho=0.0),
            'ported, rho=0.25': arm(flat=False, rho=0.25),
            'ported, rho=0.50': arm(flat=False, rho=0.5)}
    print()
    print('THE TWO ARMS, so the patch can be checked before anything rests '
          'on it')
    for name, (met, _, _) in arms.items():
        wv, cv = met['w_v_uw_wasserstein'], met['coeffvar']
        print(f'  {name:28s}  w_v_uw median {wv.median():.4f}  '
              f'mean {wv.mean():.4f}   coeffvar median {cv.median():.4f}  '
              f'({len(met)} categories)')
    print('  the UNWEIGHTED twin is the control and must not move:')
    for name, (met, _, _) in arms.items():
        print(f'  {name:28s}  coeffvar_uw median '
              f'{met["coeffvar_uw"].median():.6f}')

    rows = []
    for label, changes in CONFIGS.items():
        cfg = G.DEFAULT.replace(**changes) if changes else G.DEFAULT
        for seed in SEEDS:
            print(f'sampling "{label}" at seed {seed} ...', flush=True)
            syn_met, syn_modes, syn_vis = TC.sample_config(
                cfg, seed=seed, per_stratum=TC.PER_STRATUM)
            # ONE synthetic draw, scored against both arms: the two rows below
            # differ only in the empirical weighting.
            for arm_name, (emp_met, emp_modes, emp_vis) in arms.items():
                _, _, summary = TC.score(emp_met, emp_modes, syn_met,
                                         syn_modes, emp_vis, syn_vis)
                d = coverage.distribution_comparison(emp_met, syn_met)
                d = d.set_index('metric')['w1_standardized']
                rows.append(dict(
                    config=label, seed=seed, arm=arm_name,
                    objective=float(summary['weighted_objective']),
                    coeffvar=float(d['coeffvar']),
                    w_v_uw=float(d['w_v_uw_wasserstein']),
                    crit_bw_1=float(d['crit_bw_1'])))

    df = pd.DataFrame(rows)
    path = os.path.join(TABLES, 'TABLE_WideningAndWeights.csv')
    df.to_csv(path, index=False)

    print()
    print('=' * 78)
    print('THE SAME SYNTHETIC DRAWS, SCORED AGAINST TWO EMPIRICAL WEIGHTINGS')
    print('=' * 78)
    g = df.groupby(['arm', 'config'])[
        ['objective', 'coeffvar', 'w_v_uw', 'crit_bw_1']].mean()
    print(g.to_string(float_format=lambda v: f'{v:.4f}'))

    print()
    print('CHANGE FROM THE SHIPPED CONFIGURATION, in seed standard deviations')
    print('(the objective\'s own seed noise is 0.0066; NEGATIVE is better)')
    for arm_name in arms:
        sub = g.loc[arm_name]
        b = sub.loc['base']
        print(f'  {arm_name}')
        for label in CONFIGS:
            if label == 'base':
                continue
            r = sub.loc[label]
            print(f'    {label:20s} objective {(r.objective - b.objective) / 0.0066:+6.1f} sd'
                  f'   coeffvar {r.coeffvar - b.coeffvar:+.4f}'
                  f'   w_v_uw {r.w_v_uw - b.w_v_uw:+.4f}')
    print()
    print(f'written to {os.path.relpath(path, ROOT)}')


if __name__ == '__main__':
    main()
