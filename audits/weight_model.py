"""ONE market-share rule for both arms, and what the coherence parameter costs.

THE DEFECT. Until Stage 2h the two halves of this study drew market-share
weights by DIFFERENT rules, on the exact dimension the paper is built on.

    empirical   `rng.dirichlet(ones(n))` over every individual declaration, so
                share is INDEPENDENT of the carbon coefficient
    synthetic   a share per mixture component, split inside the component, so
                share is CORRELATED with the coefficient

Independent weights are exchangeable, so the weighted CDF converges to the
unweighted one and the measured weighting effect MUST decay like n^-1/2
whatever markets do. Real market share does not become more uniform as more
manufacturers publish. So the real arm's decay is an artifact of its weight
model, and the two arms disagree by a factor of ten above a thousand
declarations.

THE FIX, and its direction is settled: port the SYNTHETIC rule to the
EMPIRICAL arm, not the reverse. Real data carries no mode labels, so the
proxy is a contiguous cut of the SORTED values -- no fitting, and no failure
mode at three declarations. `weighting.coherent_weights` is the rule and
`rho` is its coherence axis.

WHAT THIS SCRIPT MEASURES, in four parts.

1.  THE PROXY, VALIDATED WHERE THE TRUTH IS KNOWN. The synthetic arm has BOTH
    the true mode labels and the values, so block-derived weights can be
    scored against the weights the true labels give. This is a direct
    measurement of what the proxy costs and it is available nowhere else.

2.  THE SWEEP over rho and the block count k, on both arms, reporting the
    median separation by size band and the DECAY SLOPE on log(n), which is the
    quantity that exposed the defect.

3.  THE ANCHOR. Published production volumes say share tracks technology:
    63.75 percent of world steel on the higher-carbon Rest-of-World BOF route
    against 0.03 percent on Austrian EAF (Marsh, Hattam and Allen 2025), and
    54 percent of global production in China alone (KL2). The sweep reports
    the top-share concentration each (rho, k, block_alpha) produces so a
    setting can be anchored on those rather than on agnosticism.

4.  THE CONSEQUENCES the stage prompt names: what happens to the finding that
    equal weighting beats market-share weighting below about a hundred
    declarations, and how far the generator's calibration objective moves
    against its own seed-to-seed noise.

    conda run -n compareuq python audits/weight_model.py [n_synth]
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

import corpus                      # noqa: E402
import coverage                    # noqa: E402
import empirical                   # noqa: E402
import weighting as WG             # noqa: E402
from customstats import empirical_metadata   # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')

#: Coherence levels. 0 is the old empirical rule's assumption (share unrelated
#: to carbon intensity) and 1 is maximal clustering by coefficient.
RHOS = (0.0, 0.25, 0.5, 0.75, 1.0)

#: Block counts. `None` is the generator's own rule -- uniform on 1 to 5,
#: independent of n -- and is the faithful port. `'size'` is a size-growing
#: alternative, kept in the sweep because a first version of this used it and
#: measuring what it costs is more useful than deleting it. Fixing k at a
#: constant separates CONCENTRATION from COHERENCE, which decision 97 requires.
KS = (None, 2, 3, 5, 'size')

#: Dirichlet concentration BETWEEN market groups. Smaller is more
#: concentrated. This is the axis the published production volumes anchor:
#: a flat draw over four groups puts about half the market in the largest,
#: and the real figures are 54 to 64 percent in ONE route.
BLOCK_ALPHAS = (0.3, 1.0)

#: Weight realizations per dataset per cell. The separation is a random
#: variable and a single draw of it moves a per-dataset statistic a long way
#: (up to 1.02 in absolute terms, Stage 2a-3), so every cell is a MEDIAN over
#: draws and the draw-to-draw spread is reported beside it.
N_DRAWS = 60

SIZE_BANDS = (('3-9', 3, 9), ('10-99', 10, 99), ('100-999', 100, 999),
              ('1000+', 1000, None))

#: Seed-to-seed standard deviation of the generator's calibration objective,
#: measured in Stage 2a-3 and quoted in decisions 47, 49, 61, 63 and 138. A
#: move smaller than this is not a move.
OBJECTIVE_SEED_SD = 0.0066

#: Published production volumes, for the anchor. Each is a TOP SHARE: the
#: largest single producer's fraction of the market.
PUBLISHED_TOP_SHARES = {
    'Marsh, Hattam and Allen (2025), world steel, RoW BOF': 0.6375,
    'KL2 steel example, China': 0.54,
}


def band_of(n):
    for label, lo, hi in SIZE_BANDS:
        if n >= lo and (hi is None or n <= hi):
            return label
    return None


def decay_slope(frame, value='separation', size='n'):
    """Slope of log(separation) on log(n): the statistic that exposed the gap.

    Independent weights force this toward -0.5 by exchangeability whatever the
    market does, so a rule whose slope is near -0.5 is reporting its own
    assumption.
    """
    f = frame[(frame[size] > 0) & (frame[value] > 0)]
    if len(f) < 8:
        return np.nan
    x = np.log(f[size].to_numpy(float))
    y = np.log(f[value].to_numpy(float))
    return float(np.polyfit(x, y, 1)[0])


# ---------------------------------------------------------------------------
# 1. the proxy, validated against the true mode labels
# ---------------------------------------------------------------------------
def validate_proxy(n_synth, rng, draws=N_DRAWS):
    """How closely block-derived weights reproduce what the TRUE labels give.

    Only the synthetic arm can answer this, because only it has both. For each
    dataset the reference is the weighting effect under the generator's own
    rule -- a share per mixture component, split inside the component -- and
    the candidate is the same effect under a contiguous cut of the sorted
    values at each rho.
    """
    met, values, _ = corpus.load_corpus(with_values=True)
    labels = corpus.load_mode_labels()
    met = met[met.dataset.astype(str).str.startswith('dataset')]
    pick = met.sample(min(n_synth, len(met)), random_state=0)
    vals = corpus.as_dict(values)
    lab = {str(d): np.asarray(v) for d, v in labels.items()}

    rows = []
    t0 = time.time()
    for i, (_, r) in enumerate(pick.iterrows()):
        ds = str(r.dataset)
        if ds not in vals or ds not in lab:
            continue
        x = np.asarray(vals[ds][0], dtype=float)
        modes = np.asarray(lab[ds])
        if len(x) != len(modes) or len(x) < 3:
            continue
        k_true = int(modes.max()) + 1
        # THE REFERENCE: the generator's own rule, redrawn, so that the
        # comparison is between two rules and not between a rule and one
        # stored realization.
        ref = []
        for _ in range(draws):
            share = rng.dirichlet(np.ones(k_true))
            w = np.zeros(len(x))
            for j in range(k_true):
                idx = np.flatnonzero(modes == j)
                if len(idx):
                    w[idx] = share[j] * rng.dirichlet(np.ones(len(idx)))
            ref.append(WG.weight_effect(x, w / w.sum()))
        row = dict(dataset=ds, n=len(x), k_true=k_true,
                   band=band_of(len(x)),
                   true_labels=float(np.median(ref)))
        for rho in RHOS:
            # THE PROXY AT THE TRUE BLOCK COUNT, which is the fair comparison:
            # it isolates whether a contiguous cut finds the same structure,
            # rather than confounding that with choosing k.
            got = [WG.weight_effect(
                x, WG.coherent_weights(x, rng, k=k_true, rho=rho))
                for _ in range(draws)]
            row[f'rho_{rho:g}'] = float(np.median(got))
        rows.append(row)
        if len(rows) % 200 == 0:
            print(f'  proxy {len(rows)}/{len(pick)}  {time.time()-t0:.0f}s',
                  flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. the sweep, on both arms
# ---------------------------------------------------------------------------
def sweep_arm(datasets, arm, rng, draws=N_DRAWS, rhos=RHOS, ks=KS,
              block_alphas=BLOCK_ALPHAS):
    """Median separation, effective sample size and top shares, per cell.

    TWO TOP SHARES, because the published anchors are about GROUPS and not
    about single declarations. Marsh, Hattam and Allen (2025) put 63.75 percent
    of world steel on the Rest-of-World BOF ROUTE, which is a group of
    products, so `top_block_share` is the column that anchors; `top_share`,
    the largest single declaration's weight, is reported beside it because it
    is what the effective sample size responds to.
    """
    rows = []
    for ds, x in datasets.items():
        x = np.asarray(x, dtype=float)
        if len(x) < 3:
            continue
        base = dict(arm=arm, dataset=str(ds), n=len(x), band=band_of(len(x)),
                    coeffvar=float(np.std(x) / np.mean(x)))
        # The rule the arm uses TODAY, as the reference column.
        flat = [WG.weight_effect(x, rng.dirichlet(np.ones(len(x))))
                for _ in range(draws)]
        for rho in rhos:
            for k in ks:
                kk = (int(np.mean([WG.draw_blocks(len(x), rng)
                                   for _ in range(20)])) if k is None
                      else WG.blocks_for_n(len(x)) if k == 'size'
                      else min(k, len(x)))
                for ba in block_alphas:
                    sep, neff, top, topb = [], [], [], []
                    for _ in range(draws):
                        w, blk = WG.coherent_weights(
                            x, rng, k=k, rho=rho, block_alpha=ba,
                            return_blocks=True)
                        sep.append(WG.weight_effect(x, w))
                        neff.append(WG.effective_n(w))
                        top.append(float(np.max(w)))
                        # The largest GROUP's realized share, which is what a
                        # published production volume reports: Marsh, Hattam
                        # and Allen's 63.75 percent is a steel ROUTE's share of
                        # world output, not one declaration's.
                        topb.append(float(np.bincount(blk, weights=w).max()))
                    rows.append(dict(
                        base, rho=rho,
                        k=('generator' if k is None else str(k)),
                        block_alpha=ba, k_used=kk,
                        separation=float(np.median(sep)),
                        separation_p90=float(np.percentile(sep, 90)),
                        flat_separation=float(np.median(flat)),
                        neff_frac=float(np.median(neff)) / len(x),
                        top_share=float(np.median(top)),
                        top_block_share=float(np.median(topb))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2b. the status quo, so the sweep has the right baseline
# ---------------------------------------------------------------------------
def status_quo(emp_datasets, syn_values, rng, draws=N_DRAWS):
    """Each arm under the rule it uses TODAY, which is not the same rule.

    THE SYNTHETIC ARM'S BASELINE IS ITS STORED WEIGHTS, not a flat draw over
    its values. Those stored weights are the mode-coupled ones the generator
    produced, and they are the thing the empirical rule is being brought to.
    A flat draw over the same values is a THIRD quantity -- the counterfactual
    that proves the gap is the rule and not the data -- and all three are
    reported here so none is mistaken for another.
    """
    rows = []
    for mat, (x, w) in emp_datasets.items():
        x = np.asarray(x, float)
        if len(x) < 3:
            continue
        rows.append(dict(arm='empirical', rule='stored (flat Dirichlet)',
                         dataset=str(mat), n=len(x), band=band_of(len(x)),
                         separation=WG.weight_effect(x, np.asarray(w, float))))
        flat = [WG.weight_effect(x, rng.dirichlet(np.ones(len(x))))
                for _ in range(draws)]
        rows.append(dict(arm='empirical', rule='flat draw, redrawn',
                         dataset=str(mat), n=len(x), band=band_of(len(x)),
                         separation=float(np.median(flat))))
    for ds, (x, w) in syn_values.items():
        x = np.asarray(x, float)
        if len(x) < 3:
            continue
        rows.append(dict(arm='synthetic', rule='stored (mode coupled)',
                         dataset=str(ds), n=len(x), band=band_of(len(x)),
                         separation=WG.weight_effect(x, np.asarray(w, float))))
        flat = [WG.weight_effect(x, rng.dirichlet(np.ones(len(x))))
                for _ in range(draws)]
        rows.append(dict(arm='synthetic', rule='flat draw, redrawn',
                         dataset=str(ds), n=len(x), band=band_of(len(x)),
                         separation=float(np.median(flat))))
    return pd.DataFrame(rows)


def report_status_quo(sq):
    print()
    print('THE STATUS QUO. Median separation by size band, each arm under the')
    print('rule it uses today, with the flat-draw counterfactual beside it.')
    print('The counterfactual is what proves the arm-to-arm gap is the RULE')
    print('and not the data: reweighting the synthetic values flat reproduces')
    print("the empirical arm's behavior.")
    print()
    piv = sq.pivot_table(index=['arm', 'rule'], columns='band',
                         values='separation', aggfunc='median')
    piv = piv.reindex(columns=[b[0] for b in SIZE_BANDS])
    slope = (sq.groupby(['arm', 'rule']).apply(decay_slope, include_groups=False)
             .rename('decay_slope'))
    print(piv.join(slope).to_string(float_format=lambda v: f'{v:.4f}'))


# ---------------------------------------------------------------------------
# 3. what it does to the generator's calibration
# ---------------------------------------------------------------------------
#: The calibration check is the expensive part: each cell recomputes every
#: empirical dataset's characteristics, and the critical-bandwidth bootstrap
#: on a 31,000-value category is not cheap. It is a SECONDARY check -- the
#: question is whether the objective moves more than its own seed noise -- so
#: it runs on a representative slice rather than the whole grid.
CAL_CELLS = ((0.0, None, 1.0), (0.25, None, 1.0), (0.4, None, 1.0),
             (0.5, None, 1.0), (0.75, None, 1.0), (1.0, None, 1.0),
             (0.5, 3, 1.0), (0.5, None, 0.3), (0.5, 'size', 1.0))


def calibration_shift(emp_datasets, syn_metrics, rng, cells=CAL_CELLS):
    """How far the arm-to-arm calibration moves when the EMPIRICAL rule changes.

    ONLY THE EMPIRICAL ARM IS REWEIGHTED HERE, and that is the direction the
    stage settled rather than a shortcut. The synthetic arm already attaches
    share to the mixture components -- it IS the rule being ported -- so
    reweighting it as well would replace the real thing with its own proxy and
    would break the market-weighted parent every truth run is scored against.

    The generator's tuning objective reads market-weighted characteristics, so
    changing the empirical rule moves it whether or not anything is
    regenerated. This measures that move against the objective's own
    seed-to-seed noise, which is what decides whether it is a move at all.
    """
    rows = []
    for rho, k, ba in cells:
        met = {}
        for mat, (x, _) in emp_datasets.items():
            w = WG.coherent_weights(x, rng, k=k, rho=rho, block_alpha=ba)
            met[mat] = empirical_metadata(np.asarray(x), w)
        emp = pd.DataFrame(met).T.astype(float)
        d = coverage.distribution_comparison(emp, syn_metrics)
        rows.append(dict(
            rho=rho, k=('generator' if k is None else str(k)), block_alpha=ba,
            objective=float(d.w1_standardized.mean()),
            coeffvar=float(_metric(d, 'coeffvar')),
            crit_bw_1=float(_metric(d, 'crit_bw_1')),
            w_v_uw=float(_metric(d, 'w_v_uw_wasserstein')),
            worst=d.iloc[0].metric,
            worst_w1=float(d.iloc[0].w1_standardized)))
        print(f'  calibration rho={rho} k={k} alpha={ba}: '
              f'objective {rows[-1]["objective"]:.4f}', flush=True)
    return pd.DataFrame(rows)


def _metric(d, name):
    hit = d.loc[d.metric == name, 'w1_standardized']
    return float(hit.iloc[0]) if len(hit) else np.nan


def main(n_synth=400):
    os.makedirs(TABLES, exist_ok=True)
    rng = np.random.default_rng(20260924)
    pd.set_option('display.width', 220)

    print('=' * 78)
    print('1. THE PROXY, VALIDATED WHERE THE TRUE MODE LABELS EXIST')
    print('=' * 78)
    prox = validate_proxy(n_synth, rng)
    prox.to_csv(os.path.join(TABLES, 'TABLE_WeightProxyValidation.csv'),
                index=False)
    report_proxy(prox)

    print()
    print('=' * 78)
    print('2. THE SWEEP, ON BOTH ARMS')
    print('=' * 78)
    emp, _ = empirical.prepare(np.random.default_rng(20260912))
    emp_x = {m: v for m, (v, _) in emp.items()}
    met, values, _ = corpus.load_corpus(with_values=True)
    met = met[met.dataset.astype(str).str.startswith('dataset')]
    keep = set(met.sample(min(n_synth, len(met)), random_state=1)
               .dataset.astype(str))
    vals = corpus.as_dict(values)
    syn_x = {d: v[0] for d, v in vals.items() if d in keep}

    syn_pairs = {d: v for d, v in vals.items() if d in keep}
    sq = status_quo(emp, syn_pairs, rng)
    sq.to_csv(os.path.join(TABLES, 'TABLE_WeightStatusQuo.csv'), index=False)
    report_status_quo(sq)

    sweep = pd.concat([sweep_arm(emp_x, 'empirical', rng),
                       sweep_arm(syn_x, 'synthetic', rng)],
                      ignore_index=True)
    sweep.to_csv(os.path.join(TABLES, 'TABLE_WeightModelSweep.csv.gz'),
                 index=False)
    report_sweep(sweep)

    print()
    print('=' * 78)
    print("3. WHAT IT DOES TO THE GENERATOR'S CALIBRATION")
    print('=' * 78)
    cal = calibration_shift(emp, met.set_index('dataset'), rng)
    cal.to_csv(os.path.join(TABLES, 'TABLE_WeightModelCalibration.csv'),
               index=False)
    report_calibration(cal)
    return prox, sweep, cal


def report_proxy(prox):
    if prox.empty:
        print('no rows')
        return
    print()
    print('Median uniform-to-variable separation per dataset, under the TRUE')
    print('mode labels and under a contiguous cut of the sorted values at the')
    print('SAME block count. Closer to the `true_labels` column is a better')
    print('proxy. This is what the proxy COSTS, measured rather than assumed.')
    print()
    cols = ['true_labels'] + [f'rho_{r:g}' for r in RHOS]
    print(prox.groupby('band')[cols].median()
          .reindex([b[0] for b in SIZE_BANDS])
          .to_string(float_format=lambda v: f'{v:.4f}'))
    print()
    print('RATIO to the true-label effect, and the per-dataset rank')
    print('correlation with it, which says whether the proxy orders the')
    print('categories the same way even where it misses the level:')
    print()
    out = []
    for r in RHOS:
        c = f'rho_{r:g}'
        sub = prox[(prox.true_labels > 0) & (prox[c] > 0)]
        out.append(dict(
            rho=r,
            ratio_of_medians=float(sub[c].median() / sub.true_labels.median()),
            median_ratio=float((sub[c] / sub.true_labels).median()),
            spearman=float(sub[c].rank().corr(sub.true_labels.rank())),
            n=len(sub)))
    print(pd.DataFrame(out).to_string(index=False,
                                      float_format=lambda v: f'{v:.4f}'))


def report_sweep(sweep):
    print()
    print('MEDIAN SEPARATION BY SIZE BAND, and the DECAY SLOPE on log(n).')
    print('A slope near -0.5 is exchangeability asserting itself, which is the')
    print('artifact this stage is removing; the real arm reads -0.397 today.')
    print()
    for arm in ('empirical', 'synthetic'):
        a = sweep[sweep.arm == arm]
        print(f'--- {arm} ---')
        key = ['k', 'block_alpha', 'rho']
        piv = a.pivot_table(index=key, columns='band', values='separation')
        piv = piv.reindex(columns=[b[0] for b in SIZE_BANDS])
        slope = (a.groupby(key).apply(decay_slope, include_groups=False)
                 .rename('decay_slope'))
        top = a.groupby(key).top_block_share.median().rename('top_block')
        neff = a.groupby(key).neff_frac.median().rename('neff_frac')
        print(piv.join(slope).join(top).join(neff)
              .to_string(float_format=lambda v: f'{v:.4f}'))
        print()
    print('TODAY, for reference: the flat draw each arm uses now.')
    for arm in ('empirical', 'synthetic'):
        a = sweep[(sweep.arm == arm) & (sweep.k == 'generator')
                  & (sweep.rho == 0.0) & (sweep.block_alpha == 1.0)]
        f = a.drop_duplicates('dataset')
        print(f'  {arm:<10s} median flat separation '
              f'{f.flat_separation.median():.4f}, decay slope '
              f'{decay_slope(f, "flat_separation"):+.3f}')
    print()
    print('THE ANCHOR. Published top shares, against what each cell produces:')
    for name, share in PUBLISHED_TOP_SHARES.items():
        print(f'  {share:.4f}   {name}')


def report_calibration(cal):
    print()
    print('Mean standardized arm-to-arm distance over the characteristics,')
    print('with the EMPIRICAL arm reweighted and the synthetic arm untouched.')
    print(f'The objective\'s own seed-to-seed standard deviation is '
          f'{OBJECTIVE_SEED_SD:.4f}, so a move smaller than that is not a move.')
    print()
    base = cal[(cal.rho == 0.0) & (cal.k == 'generator')
               & (cal.block_alpha == 1.0)]
    ref = float(base.objective.iloc[0]) if len(base) else np.nan
    out = cal.copy()
    out['vs_rho0_in_seed_sd'] = (out.objective - ref) / OBJECTIVE_SEED_SD
    print(out[['k', 'block_alpha', 'rho', 'objective', 'vs_rho0_in_seed_sd',
               'coeffvar', 'crit_bw_1', 'w_v_uw', 'worst']]
          .to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print()
    print('`w_v_uw` is the arm-to-arm distance on the uniform-to-variable')
    print("Wasserstein distance itself -- the paper's central quantity -- so it")
    print('is the column that says whether the two arms now AGREE about how')
    print('much weighting matters. Lower is closer.')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
