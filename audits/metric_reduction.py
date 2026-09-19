"""The full multivariate reduction of the characteristic set. Stage 2f.

NOTEBOOK 3 IS WHERE THE PAPER'S NUMBERS COME FROM, not this script. Its last
section calls the same functions from `src/metricreduction.py` and writes the
`TABLE_Reduction*` tables; this script exists so the reduction can be re-run
and argued with on its own without a two-hour notebook execution.

**It runs on its own random stream, so its importances differ from the
notebook's within fold-to-fold noise.** Quote the notebook's tables. A number
in the decision log that came from a different seed than the table the paper
prints is the mistake Stage 2c's tenth habit records.

THE REDUCTION IS RUN TWICE, AGAINST TWO DIFFERENT KINDS OF TARGET, and that is
the design rather than a robustness check:

  FIT     how far the fitted curve sits from its target. `w1` in sample,
          `w1_cv` cross-validated on the empirical arm, `w1_market` against
          the known parent on the synthetic arm.
  ANSWER  how wrong the probabilistic LCA's answer is, from Stage 2e's run
          against the true distributions. Synthetic arm only, because only
          there does a true parent exist.

A characteristic that survives one and not the other is the interesting case
and is reported as such.

    conda run -n compareuq python audits/metric_reduction.py [--quick]
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
sys.path.insert(0, HERE)

import corpus  # noqa: E402
import coverage  # noqa: E402
import metricreduction as RED  # noqa: E402
from _common import write  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables')
SEED = 20260919

FIT_TARGETS = ('w1', 'w1_cv', 'w1_market')
ANSWER_TARGETS = ('err_eci_mean', 'err_eci_rank_1', 'err_ui', 'err_eci_p95')


# --------------------------------------------------------------------------

def load_inputs():
    """Characteristics, scores, the truth run and the visible-mode counts.

    The characteristics come from the tables the notebooks write, so this
    script and the notebook are reading the same numbers. The synthetic
    characteristics come from the corpus itself rather than from the Excel
    table, because the corpus is the source and the table is a copy of it.
    """
    emp = pd.read_excel(os.path.join(TABLES, 'TABLE_EmpiricalECCMetrics.xlsx'),
                        index_col=0)
    emp = emp.reset_index().rename(columns={emp.index.name or 'index': 'dataset'})
    emp.columns = ['dataset'] + list(emp.columns[1:])
    emp['arm'] = 'empirical'

    metrics, _values, _meta = corpus.load_corpus(with_values=False)
    syn = metrics.rename(columns={'dataset': 'dataset'}).copy()
    syn['arm'] = 'synthetic'

    shared = [c for c in emp.columns if c in syn.columns]
    chars = pd.concat([emp[shared], syn[shared]], ignore_index=True)

    scores = pd.read_csv(os.path.join(TABLES, 'TABLE_TargetComparison.csv'))
    truth = pd.read_csv(os.path.join(TABLES, 'TABLE_PLCATruth.csv.gz'))
    modes_path = os.path.join(TABLES, 'TABLE_VisibleModes.csv')
    modes = pd.read_csv(modes_path) if os.path.exists(modes_path) else None
    return chars, scores, truth, modes


def run(quick=False):
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    chars, scores, truth, modes = load_inputs()
    frame, metrics = RED.assemble(chars, scores, truth=truth, modes=modes)
    metrics = list(metrics) + [m for m in RED.MODE_METRICS if m in frame.columns]
    print(f'{len(frame):,} rows, {len(metrics)} candidate metrics, '
          f'{frame.dataset.nunique():,} datasets')

    # ---- what the models are allowed to forget -------------------------
    write(RED.missingness_by_band(frame, metrics),
          'AUDIT_ReductionMissingness.csv')
    cost = RED.complete_case_cost(frame, metrics)
    write(cost, 'AUDIT_ReductionCompleteCaseCost.csv')
    print('\nWHAT A COMPLETE-CASE MODEL WOULD DROP')
    print(cost.to_string(index=False))

    used = RED.rows_used_by_band(frame, FIT_TARGETS + ANSWER_TARGETS)
    write(used, 'AUDIT_ReductionRowsUsed.csv')
    print('\nROWS EACH MODEL ACTUALLY USES, per size band')
    print(used[used.method == 'KDE, Uniform'].to_string(index=False))

    red = RED.redundancy_table(frame, metrics)
    write(red, 'AUDIT_ReductionRedundancy.csv')
    print('\nEFFECTIVE DIMENSION of the candidate set')
    print(red.groupby('arm')[['effective_dimension', 'n_metrics']].first()
          .to_string())
    print('\nMOST REDUNDANT PAIRS')
    print(red.head(10)[['arm', 'metric_a', 'metric_b', 'correlation']]
          .to_string(index=False))

    defn = RED.definitional_check(frame, scores, metrics=metrics)
    write(defn, 'AUDIT_ReductionDefinitional.csv')
    print('\nIS A CANDIDATE PREDICTING A TARGET, OR IS IT PART OF ONE?')
    ident = defn[defn.is_an_identity]
    if len(ident):
        print(ident[['arm', 'method', 'metric', 'spearman_definitional',
                     'spearman_w1']].to_string(index=False))
    else:
        print('  no candidate reproduces the definitional term exactly')

    conf = RED.size_confounding(frame, metrics)
    write(conf, 'AUDIT_ReductionSizeConfounding.csv')
    print('\nHOW MUCH OF EACH METRIC IS DATASET SIZE (spline R2 on log n)')
    print(conf[conf.arm == 'empirical'].head(8)
          [['metric', 'r2_on_log_n', 'spearman_with_log_n']].to_string(index=False))

    # ---- the reduction, twice ------------------------------------------
    methods = sorted(frame.method.unique())
    n_rep, n_spl = (3, 3) if quick else (10, 5)

    imps, incs = [], []
    plan = []
    for target in FIT_TARGETS:
        for arm in ('empirical', 'synthetic'):
            if target == 'w1_market' and arm == 'empirical':
                continue  # no parent exists on the empirical arm
            plan += [(arm, m, target) for m in methods]
    for target in ANSWER_TARGETS:
        plan += [('synthetic', m, target) for m in methods]

    for i, (arm, method, target) in enumerate(plan, 1):
        if target not in frame.columns:
            continue
        sub = frame[(frame.arm == arm) & (frame.method == method)]
        if sub[target].notna().sum() < 60:
            continue
        print(f'  [{i:>3}/{len(plan)}] {arm:9s} {method:20s} {target:16s} '
              f'{time.time()-t0:6.0f}s', flush=True)
        imps.append(RED.importance(frame, metrics, target, arm=arm,
                                   method=method, rng=rng,
                                   n_repeats=n_rep, n_splits=n_spl))
        incs.append(RED.incremental_over_size(frame, metrics, target,
                                              arm=arm, method=method))

    importances = pd.concat([d for d in imps if len(d)], ignore_index=True)
    increments = pd.concat([d for d in incs if len(d)], ignore_index=True)
    importances['target_family'] = importances.target.map(
        lambda t: RED.TARGETS[t]['family'])
    increments['target_family'] = increments.target.map(
        lambda t: RED.TARGETS[t]['family'])
    write(importances, 'AUDIT_ReductionImportance.csv')
    write(increments, 'AUDIT_ReductionIncremental.csv')

    # ---- the survivors, per target family ------------------------------
    out = []
    for fam in ('fit', 'answer'):
        sub = importances[importances.target_family == fam]
        s = RED.rank_survivors(sub)
        if len(s):
            s.insert(0, 'target_family', fam)
            out.append(s)
    pooled = RED.rank_survivors(importances)
    pooled.insert(0, 'target_family', 'both')
    out.append(pooled)
    # The same ranking with the definitional candidate removed, because on a
    # fit target it is an identity rather than a predictor. On the downstream
    # error no identity exists, so both rankings are reported.
    trimmed = RED.rank_survivors(
        importances[~importances.metric.isin(RED.DEFINITIONAL_CANDIDATES)])
    trimmed.insert(0, 'target_family', 'both, definitional removed')
    out.append(trimmed)
    for fam in ('fit', 'answer'):
        t = RED.rank_survivors(
            importances[(importances.target_family == fam)
                        & ~importances.metric.isin(RED.DEFINITIONAL_CANDIDATES)])
        if len(t):
            t.insert(0, 'target_family', f'{fam}, definitional removed')
            out.append(t)
    survivors = pd.concat(out, ignore_index=True)
    write(survivors, 'AUDIT_ReductionSurvivors.csv')

    print('\nSURVIVORS, by target family')
    for fam in sorted(survivors.target_family.unique()):
        s = survivors[survivors.target_family == fam]
        if len(s):
            print(f'\n  {fam.upper()}')
            print(s.head(8)[['metric', 'mean_rank', 'mean_importance',
                             'share_top5', 'n_models']].to_string(index=False))

    # ---- which survives one and not the other --------------------------
    fit = survivors[survivors.target_family == 'fit'].set_index('metric')
    ans = survivors[survivors.target_family == 'answer'].set_index('metric')
    both = fit[['mean_rank']].join(ans[['mean_rank']], lsuffix='_fit',
                                   rsuffix='_answer', how='inner')
    both['rank_shift'] = both.mean_rank_answer - both.mean_rank_fit
    both = both.sort_values('rank_shift').reset_index()
    write(both, 'AUDIT_ReductionFitVersusAnswer.csv')
    print('\nSURVIVES ONE TARGET AND NOT THE OTHER (negative = matters more '
          'for the ANSWER than for the fit)')
    print(pd.concat([both.head(5), both.tail(5)]).to_string(index=False))

    # ---- the three modality measures head to head -----------------------
    mrows = []
    for arm, tg in (('empirical', ('w1', 'w1_cv')),
                    ('synthetic', ('w1', 'w1_market', 'err_eci_mean'))):
        d = RED.modality_head_to_head(frame, tg, methods, arm, rng=rng)
        if len(d):
            mrows.append(d)
    if mrows:
        modality = pd.concat(mrows, ignore_index=True)
        write(modality, 'AUDIT_ReductionModality.csv')
        print('\nTHE THREE MODALITY MEASURES, over a spline in log(n)')
        print(modality.to_string(index=False))
    agree = RED.modality_agreement(frame)
    write(agree, 'AUDIT_ReductionModalityAgreement.csv')
    print('\nHOW FAR APART THE MODALITY MEASURES ARE')
    print(agree.to_string(index=False))

    # ---- post-stratified -------------------------------------------------
    shares = RED.empirical_size_shares(frame)
    ps_frame = RED.size_mix_resample(frame, shares, rng=rng)
    print('\nempirical size mix: '
          + ', '.join(f'{k} {v:.3f}' for k, v in shares.items())
          + f'  -> {ps_frame.dataset.nunique():,} synthetic datasets')
    ps = []
    for target in ('w1', 'w1_market', 'err_eci_mean'):
        for method in methods:
            d = RED.importance(ps_frame, metrics, target, arm='synthetic',
                               method=method, rng=rng,
                               n_repeats=n_rep, n_splits=n_spl)
            if len(d):
                ps.append(d)
    if ps:
        imp_ps = pd.concat(ps, ignore_index=True)
        imp_ps['allocation'] = 'empirical size mix'
        eq = importances[(importances.arm == 'synthetic')
                         & importances.target.isin(('w1', 'w1_market',
                                                    'err_eci_mean'))].copy()
        eq['allocation'] = 'equal'
        both_alloc = pd.concat([eq, imp_ps], ignore_index=True)
        write(both_alloc, 'AUDIT_ReductionPostStratified.csv')
        print('\nMEAN IMPORTANCE AT BOTH ALLOCATIONS, synthetic arm')
        print(both_alloc.groupby(['allocation', 'metric']).importance.mean()
              .unstack(0).sort_values('equal', ascending=False)
              .head(12).round(4).to_string())

    # ---- which method wins ----------------------------------------------
    wrows = []
    for value, tag in (('w1', 'in sample'), ('w1_cv', 'cross-validated'),
                       ('w1_market', 'against the market parent')):
        if value not in frame.columns:
            continue
        w = RED.winner_frame(frame, value, within_weighting=True)
        for arm in ('empirical', 'synthetic'):
            for weighting in ('Uniform', 'Variable'):
                wi = RED.winner_importance(w, metrics, arm=arm,
                                           weighting=weighting, rng=rng,
                                           n_repeats=n_rep, n_splits=n_spl)
                if len(wi):
                    wi['value'] = value
                    wi['value_label'] = tag
                    wrows.append(wi)
    winners = pd.concat(wrows, ignore_index=True) if wrows else pd.DataFrame()
    if len(winners):
        write(winners, 'AUDIT_ReductionWinner.csv')
        print('\nWHICH METHOD WINS: can it be predicted at all?')
        print(winners.groupby(['value', 'arm', 'weighting', 'model'])
              [['accuracy', 'majority_baseline', 'lift_over_baseline', 'n_rows']]
              .first().to_string())
        print('\nTOP PREDICTORS OF THE WINNER')
        top = (winners.groupby('metric').importance.mean()
               .sort_values(ascending=False).head(8))
        print(top.to_string())

    # ---- partial dependence: the marginal view against the multivariate --
    survivor_names = list(trimmed[trimmed.survivor].metric)
    pdrows = []
    for arm, value in (('empirical', 'w1_cv'), ('synthetic', 'w1_market'),
                       ('synthetic', 'err_eci_mean')):
        if value not in frame.columns:
            continue
        for method in methods:
            d = RED.partial_dependence_table(frame, metrics, value,
                                             which=survivor_names, arm=arm,
                                             method=method, rng=rng)
            if len(d):
                pdrows.append(d)
    if pdrows:
        partial = pd.concat(pdrows, ignore_index=True)
        write(partial, 'AUDIT_ReductionPartialDependence.csv')
        curves = pd.concat(
            [RED.curves_for(frame, survivor_names, methods, value='w1_market',
                            arm='synthetic', rng=rng)], ignore_index=True)
        mvp = RED.marginal_versus_partial(
            curves, partial[(partial.arm == 'synthetic')
                            & (partial.target == 'w1_market')])
        write(mvp, 'AUDIT_ReductionMarginalVersusPartial.csv')
        print('\nHOW MUCH OF EACH MARGINAL SLOPE SURVIVES HOLDING THE REST')
        print(mvp.to_string(index=False))

    # ---- the generalization question -------------------------------------
    emp_chars = chars[chars.arm == 'empirical']
    syn_chars = chars[chars.arm == 'synthetic']
    cov = coverage.coverage_table(emp_chars, syn_chars,
                                  metrics=tuple(m for m in metrics
                                                if m in emp_chars.columns))
    cvi = RED.coverage_versus_importance(cov, trimmed)
    write(cvi, 'AUDIT_ReductionCoverageVsImportance.csv')
    print('\nDO THE METRICS THAT MATTER SIT WHERE THE CORPUS IS DENSEST?')
    print(cvi.to_string(index=False))
    for k, v in cvi.attrs.items():
        print(f'  {k}: {v:+.3f}')

    print(f'\ndone in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    run(quick='--quick' in sys.argv)
