"""Stage 2c: the old evaluation target against the two new ones, side by side.

This is the measurement behind the stage. It computes, for both arms:

  the OLD score      in-sample W1 against the variable-weighted empirical CDF of
                     the same data the model was fitted to;
  the SYNTHETIC fix  W1 against the KNOWN PARENT, under the weighting scheme the
                     method is estimating -- the sampling mixture for a
                     uniform-weighted method, the market-weighted mixture for a
                     variable-weighted one;
  the EMPIRICAL fix  cross-validated W1, fitted on half the values and scored
                     against the weighted empirical CDF of the other half, over
                     ten random splits in both directions.

and then the things that only make sense once the target is fixed: regret,
post-stratification to the empirical size mix, the fit-versus-definitional
decomposition, the overlap area as a second criterion, and the conditioning on
visible modality.

Everything here calls `src/recovery.py`, which is under test. The script exists
so the numbers can be looked at without re-running a notebook; notebook 2 writes
the paper-facing versions of the same tables.

    conda run -n compareuq python audits/evaluation_target.py [n_synth]

`n_synth` limits the corpus sample for a quick look. The default is the whole
corpus, which takes about six minutes.
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

import comparison as CMP  # noqa: E402
import corpus  # noqa: E402
import empirical  # noqa: E402
import mixture as M  # noqa: E402
import recovery as R  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
SEED = 42

#: Corpus datasets that also get a cross-validated score, so the two arms can be
#: compared on the SAME criterion as well as on their own. Cross-validating the
#: whole corpus would take about half an hour for no visible change in a mean.
N_CV_SYNTHETIC = 2_000


def load_arms(n_synth=None):
    rng = np.random.default_rng(SEED)
    emp, _ = empirical.prepare(rng.spawn(1)[0])
    met, vals, _ = corpus.load_corpus()
    ids = sorted(met.dataset.astype(str))
    if n_synth:
        ids = sorted(pd.Series(ids).sample(n_synth, random_state=0))
    syn = corpus.as_dict(vals[vals.dataset_id.isin(set(ids))])
    specs = corpus.load_parent_specs()
    parents = {k: M.parent_from_spec(specs[k]) for k in ids}
    # `n_modes_visible` is not one of the stored characteristics -- the corpus
    # stores `modality_index` and Silverman's critical bandwidth -- and it is
    # the one the generator is steered by (decision 38), so Stage 2c computes it
    # here for both arms rather than substituting a different modality measure.
    import modality as _md
    chars = met.set_index('dataset')
    chars['n_modes_visible'] = pd.Series(
        {k: _md.n_modes_visible(syn[k][0]) for k in ids})
    emp_modes = pd.Series({k: _md.n_modes_visible(v[0]) for k, v in emp.items()})
    return emp, {k: syn[k] for k in ids}, parents, chars, emp_modes


def fmt(df, cols=None):
    d = df[cols] if cols else df
    return d.to_string(index=False, float_format=lambda v: f'{v:.4f}')


def section(title):
    print()
    print('=' * 78)
    print(title)
    print('=' * 78)


def main(n_synth=None):
    os.makedirs(TABLES, exist_ok=True)
    pd.set_option('display.width', 240)
    t0 = time.time()
    emp, syn, parents, chars, emp_modes = load_arms(n_synth)
    print(f'empirical {len(emp)} datasets, synthetic {len(syn)}, '
          f'{time.time() - t0:.0f}s')

    # ---------------------------------------------------------------- recovery
    print('scoring the synthetic arm against its parents...', flush=True)
    rec = R.score_recovery_arm(syn, parents, progress=2000)
    rec.to_csv(os.path.join(TABLES, 'TABLE_RecoveryScores.csv'), index=False)

    # -------------------------------------------------------- cross-validation
    print('cross-validating the empirical arm...', flush=True)
    rng = np.random.default_rng(SEED)
    cv_emp = R.cv_arm(emp, rng.spawn(1)[0], 'empirical', progress=50)
    cv_pick = sorted(pd.Series(sorted(syn)).sample(
        min(N_CV_SYNTHETIC, len(syn)), random_state=0))
    print('cross-validating a corpus sample...', flush=True)
    cv_syn = R.cv_arm({k: syn[k] for k in cv_pick}, rng.spawn(1)[0],
                      'synthetic', progress=500)
    cv = pd.concat([cv_emp, cv_syn], ignore_index=True)
    cv.to_csv(os.path.join(TABLES, 'TABLE_CrossValidatedScores.csv.gz'),
              index=False)
    cvs = R.cv_summary(cv)

    # ------------------------------------------------------------- in sample
    print('in-sample scores and the decomposition...', flush=True)
    dec = pd.concat([R.decompose_arm(emp, 'empirical'),
                     R.decompose_arm(syn, 'synthetic', progress=2000)],
                    ignore_index=True)
    dec.to_csv(os.path.join(TABLES, 'TABLE_WeightingDecomposition.csv'),
               index=False)

    scores = pd.concat([
        dec[dec.arm == 'empirical'][['arm', 'dataset', 'n', 'method',
                                     'w1_total']].rename(
            columns={'w1_total': 'w1'}),
        rec[['arm', 'dataset', 'n', 'method', 'w1']],
    ], ignore_index=True)
    scores = scores.merge(
        cvs[['arm', 'dataset', 'method', 'w1_cv', 'w1_cv_sd_across_splits',
             'w1_cv_se', 'w1_cv_n_splits']],
        on=['arm', 'dataset', 'method'], how='left')
    scores = scores.merge(
        rec[['arm', 'dataset', 'method', 'w1_parent', 'w1_parent_tail',
             'w1_parent_total', 'overlap', 'w1_parent_location',
             'w1_parent_shape', 'w1_market', 'w1_sampling', 'overlap_market',
             'parent_separation']],
        on=['arm', 'dataset', 'method'], how='left')
    scores['ovl_loss'] = 1.0 - scores.overlap
    scores = R.add_size_band(scores)
    scores.to_csv(os.path.join(TABLES, 'TABLE_TargetComparison.csv'),
                  index=False)

    report(scores, rec, cv, cvs, dec, chars, emp_modes)
    print(f'\ntotal {time.time() - t0:.0f}s')


def report(scores, rec, cv, cvs, dec, chars, emp_modes):
    shares = R.empirical_size_shares(scores[scores.arm == 'empirical'])
    syn = scores[scores.arm == 'synthetic']
    emp_s = scores[scores.arm == 'empirical']

    section('1. WHAT CHANGING THE TARGET DOES, synthetic arm')
    t = syn.groupby('method').agg(
        w1_in_sample=('w1', 'mean'), w1_parent=('w1_parent', 'mean'),
        w1_parent_median=('w1_parent', 'median'),
        w1_parent_p90=('w1_parent', lambda s: s.quantile(0.90)),
        tail=('w1_parent_tail', 'mean'),
        ovl_loss=('ovl_loss', 'mean')).reset_index()
    t['rank_in_sample'] = t.w1_in_sample.rank()
    t['rank_parent'] = t.w1_parent.rank()
    print(fmt(t.sort_values('w1_parent')))
    print('\nper-dataset ranks (1 = best), mean:')
    a = CMP.add_ranks(syn, 'w1').groupby('method').w1_rank.mean()
    b = CMP.add_ranks(syn, 'w1_parent').groupby('method').w1_parent_rank.mean()
    c = CMP.add_ranks(syn, 'ovl_loss').groupby('method').ovl_loss_rank.mean()
    print(fmt(pd.DataFrame(dict(method=a.index, in_sample=a.values,
                                parent=b.values, overlap=c.values))
              .sort_values('parent')))
    print('\nwin share:')
    for val in ('w1', 'w1_parent'):
        w = R.win_share(syn, val)
        print(f'  by {val}:  ' + '  '.join(
            f'{r.method} {r.win_share:.3f}' for _, r in w.iterrows()))

    section('1b. THE COMMON TARGET: every method against the MARKET parent')
    print('`w1_parent` above scores each method against the parent IT estimates,')
    print('which is the only fair way to judge an ESTIMATION method and CANNOT')
    print('compare the two weighting schemes, because they are estimating')
    print('different things. Against one common target they can. The')
    print('market-weighted parent is the decision-relevant one: a pLCA of what')
    print('gets built is a statement about the market-weighted population.')
    t = syn.groupby('method').agg(
        w1_market=('w1_market', 'mean'), median=('w1_market', 'median'),
        p90=('w1_market', lambda s: s.quantile(0.90)),
        w1_sampling=('w1_sampling', 'mean')).reset_index()
    a = CMP.add_ranks(syn, 'w1_market').groupby('method').w1_market_rank.mean()
    wm = R.win_share(syn, 'w1_market').set_index('method').win_share
    t['mean_rank'] = t.method.map(a)
    t['win_share'] = t.method.map(wm).fillna(0.0)
    print(fmt(t.sort_values('w1_market')))
    sep = syn.drop_duplicates('dataset').parent_separation
    print(f'\n  distance between the two parents, before any fitting: '
          f'mean {sep.mean():.4f}, median {sep.median():.4f}, '
          f'p90 {sep.quantile(0.90):.4f}')
    print('  That is the floor a uniform-weighted method cannot beat against the')
    print('  market parent, and the scale the numbers above have to be read at.')
    print('\n  head to head, within each estimation family, against the market '
          'parent:')
    wide = syn.pivot_table(index='dataset', columns='method', values='w1_market')
    for fam in ('Normal', 'Lognormal', 'KDE'):
        u, v = f'{fam}, Uniform', f'{fam}, Variable'
        if u in wide and v in wide:
            print(f'    {fam:<10} variable beats uniform on '
                  f'{100 * (wide[v] < wide[u]).mean():5.1f} pct of datasets; '
                  f'mean {wide[v].mean():.4f} against {wide[u].mean():.4f}')
    print('\n  by size band, share of datasets where variable beats uniform:')
    nb = syn.drop_duplicates('dataset').set_index('dataset').size_band
    for band in [b[0] for b in R.SIZE_BANDS]:
        k = nb[nb == band].index.intersection(wide.index)
        if not len(k):
            continue
        bits = []
        for fam in ('Normal', 'Lognormal', 'KDE'):
            u, v = f'{fam}, Uniform', f'{fam}, Variable'
            bits.append(f'{fam} {100 * (wide.loc[k, v] < wide.loc[k, u]).mean():5.1f}')
        print(f'    {band:16s} n={len(k):5d}   ' + '   '.join(bits))

    section('2. CROSS-VALIDATION, empirical arm')
    print('READ THIS BEFORE THE TABLE. A cross-validated score may be compared')
    print('ACROSS ESTIMATION METHODS WITHIN ONE WEIGHTING SCHEME and NOT across')
    print('weighting schemes. The empirical weights are an exchangeable flat')
    print('Dirichlet draw with no market information in them, so they carry')
    print('nothing that generalizes from one half of a dataset to the other: the')
    print('expected variable-weighted CDF of a random half IS the unweighted one,')
    print('and a uniform-weighted fit is the better predictor by construction.')
    print('That is a property of the SYNTHETIC weights, not a finding about')
    print('weighting, and it is why the weighting claim rests on the synthetic')
    print("arm's market parent (section 1b) and not on this table.")
    g = cvs[cvs.arm == 'empirical']
    t = g.groupby('method').agg(
        w1_cv=('w1_cv', 'mean'), median=('w1_cv', 'median'),
        p90=('w1_cv', lambda s: s.quantile(0.90)),
        sd_across_splits=('w1_cv_sd_across_splits', 'median'),
        n_datasets=('dataset', 'nunique')).reset_index()
    ins = emp_s.groupby('method').w1.mean().rename('w1_in_sample')
    print()
    print(fmt(t.merge(ins.reset_index(), on='method').sort_values('w1_cv')))
    print('\nTHE COMPARISON THAT IS VALID: the three estimation methods, inside '
          'each weighting scheme.')
    for wt in ('Uniform', 'Variable'):
        h = g[g.method.str.endswith(wt)]
        wide = h.pivot_table(index='dataset', columns='method', values='w1_cv')
        ins_w = emp_s[emp_s.method.str.endswith(wt)].groupby('method').w1.mean()
        tt = pd.DataFrame(dict(
            method=wide.columns,
            cv_mean=[wide[c].mean() for c in wide.columns],
            cv_median=[wide[c].median() for c in wide.columns],
            cv_rank=[wide.rank(axis=1)[c].mean() for c in wide.columns],
            cv_win_share=[(wide.idxmin(axis=1) == c).mean()
                          for c in wide.columns],
            in_sample=[ins_w.get(c, np.nan) for c in wide.columns]))
        print(f'  --- {wt} weighting, {len(wide)} datasets ---')
        print(fmt(tt.sort_values('cv_mean')))
    print('\nHOW BIG IS THE SPLIT NOISE against the gaps between methods?')
    wide = g.pivot_table(index='dataset', columns='method', values='w1_cv')
    sd = g.pivot_table(index='dataset', columns='method',
                       values='w1_cv_sd_across_splits')
    spread = (wide.max(axis=1) - wide.min(axis=1))
    print(f'  median spread between best and worst method   {spread.median():.4f}')
    print(f'  median split-to-split sd of one method        '
          f'{sd.median(axis=1).median():.4f}')
    print('  by size band:')
    nb = emp_s.drop_duplicates('dataset').set_index('dataset').size_band
    for band in [b[0] for b in R.SIZE_BANDS]:
        k = nb[nb == band].index.intersection(wide.index)
        if len(k):
            print(f'    {band:16s} n={len(k):3d}  spread {spread[k].median():.4f}'
                  f'   split sd {sd.loc[k].median(axis=1).median():.4f}')
    print('\nwithin-weighting ranks (the only honest way to read a held-out score):')
    cvr = CMP.add_ranks(
        cvs.rename(columns={'w1_cv': 'value'}), 'value',
        within_weighting=True, suffix='cv_rank_within')
    print(fmt(cvr.groupby(['arm', 'method']).cv_rank_within.mean()
              .reset_index().sort_values(['arm', 'cv_rank_within'])))

    section('3. THE TWO LINES OF EVIDENCE, side by side')
    print('Within each weighting scheme, so the two arms are read the same way')
    print('and the empirical confound of section 2 cannot enter. Ranks are 1 to 3')
    print('among the estimation methods.')
    rows = []
    for wt in ('Uniform', 'Variable'):
        sw = syn[syn.method.str.endswith(wt)]
        ew = emp_s[emp_s.method.str.endswith(wt) & emp_s.w1_cv.notna()]
        sp = sw.pivot_table(index='dataset', columns='method',
                            values='w1_parent')
        si = sw.pivot_table(index='dataset', columns='method', values='w1')
        ec = ew.pivot_table(index='dataset', columns='method', values='w1_cv')
        ei = ew.pivot_table(index='dataset', columns='method', values='w1')
        for method in sp.columns:
            rows.append(dict(
                method=method,
                syn_parent=sp[method].mean(),
                syn_parent_rank=sp.rank(axis=1)[method].mean(),
                syn_parent_win=(sp.idxmin(axis=1) == method).mean(),
                syn_in_rank=si.rank(axis=1)[method].mean(),
                emp_cv=ec[method].mean() if method in ec else np.nan,
                emp_cv_rank=(ec.rank(axis=1)[method].mean()
                             if method in ec else np.nan),
                emp_cv_win=((ec.idxmin(axis=1) == method).mean()
                            if method in ec else np.nan),
                emp_in_rank=ei.rank(axis=1)[method].mean()))
    print(fmt(pd.DataFrame(rows).sort_values(['method'])))
    print('\nDO THE TWO ARMS AGREE? The ordering of the three estimation methods,')
    print('by each arm and criterion:')
    for wt in ('Uniform', 'Variable'):
        sw = syn[syn.method.str.endswith(wt)]
        ew = emp_s[emp_s.method.str.endswith(wt) & emp_s.w1_cv.notna()]
        def order(frame, col):
            return ' < '.join(frame.groupby('method')[col].mean()
                              .sort_values().index.str.split(',').str[0])
        print(f'  {wt:<9} synthetic, parent : {order(sw, "w1_parent")}')
        print(f'  {wt:<9} synthetic, in samp: {order(sw, "w1")}')
        print(f'  {wt:<9} empirical, CV     : {order(ew, "w1_cv")}')
        print(f'  {wt:<9} empirical, in samp: {order(ew, "w1")}')

    section('3b. IS THE DISAGREEMENT DISTINGUISHABLE? Paired bootstrap')
    print('Resampled over DATASETS, paired within dataset. Positive means the')
    print('reference method scored LOWER, so better. This says whether a gap')
    print('survives resampling the datasets; it does NOT cover the empirical')
    print("arm's other noise source, the single Dirichlet weight realization,")
    print('which Stage 2b measured at a median W1 of 0.1344 and which Stage 2h')
    print('averages over.')
    rngb = np.random.default_rng(0)
    for wt in ('Uniform', 'Variable'):
        sw = syn[syn.method.str.endswith(wt)]
        ew = emp_s[emp_s.method.str.endswith(wt) & emp_s.w1_cv.notna()]
        ref = f'KDE, {wt}'
        print(f'--- {wt} weighting, reference {ref} ---')
        a = R.paired_bootstrap(sw, 'w1_parent', ref, rng=rngb)
        a.insert(0, 'arm_criterion', 'synthetic, parent')
        b = R.paired_bootstrap(ew, 'w1_cv', ref, rng=rngb)
        b.insert(0, 'arm_criterion', 'empirical, CV')
        print(fmt(pd.concat([a, b])[['arm_criterion', 'method', 'n_datasets',
                                     'mean_difference', 'ci_lo', 'ci_hi',
                                     'reference_wins', 'distinguishable']]))
        print()
    print('And the weighting question, on the COMMON market target:')
    for fam in ('Normal', 'Lognormal', 'KDE'):
        g = syn[syn.method.isin([f'{fam}, Uniform', f'{fam}, Variable'])]
        b = R.paired_bootstrap(g, 'w1_market', f'{fam}, Variable', rng=rngb)
        b.insert(0, 'family', fam)
        print(fmt(b[['family', 'method', 'n_datasets', 'mean_difference',
                     'ci_lo', 'ci_hi', 'reference_wins', 'distinguishable']]))
    print('  (positive = VARIABLE weighting scored lower, so better)')
    print('  by size band, KDE:')
    g = syn[syn.method.isin(['KDE, Uniform', 'KDE, Variable'])]
    b = R.paired_bootstrap(g, 'w1_market', 'KDE, Variable', by=['size_band'],
                           rng=rngb)
    print(fmt(b[['size_band', 'n_datasets', 'mean_difference', 'ci_lo',
                 'ci_hi', 'reference_wins', 'distinguishable']]))

    section('3c. THE TWO ARMS DISAGREE. WHY, in three steps')
    print('Against the parent the KDE beats the lognormal; cross-validated on the')
    print('empirical arm the lognormal beats the KDE, and both gaps survive the')
    print('bootstrap. They are not contradicting each other -- they are being read')
    print('on different criteria and different size mixes. Removing one at a time:')
    rngb = np.random.default_rng(0)
    for wt in ('Uniform', 'Variable'):
        ref, alt = f'KDE, {wt}', f'Lognormal, {wt}'
        pair = [ref, alt]
        sp = syn[syn.method.isin(pair)]
        sc = sp[sp.w1_cv.notna()]
        ec = emp_s[emp_s.method.isin(pair) & emp_s.w1_cv.notna()]

        def band_reweighted(frame, col):
            b = R.paired_bootstrap(frame, col, ref, by=['size_band'],
                                   rng=rngb).set_index('size_band')
            w = np.array([shares.get(k, 0.0) for k in b.index], float)
            v = b.mean_difference.to_numpy(float)
            return (float(v.mean()),
                    float((v * w).sum() / w.sum()) if w.sum() else np.nan)

        step1 = R.paired_bootstrap(sp, 'w1_parent', ref, rng=rngb)
        step2 = R.paired_bootstrap(sc, 'w1_cv', ref, rng=rngb)
        _, step3 = band_reweighted(sc, 'w1_cv')
        step4 = R.paired_bootstrap(ec, 'w1_cv', ref, rng=rngb)
        _, step4r = band_reweighted(ec, 'w1_cv')
        print(f'--- {wt}: {ref} minus {alt}, positive = KDE better ---')
        print(f'  synthetic, against the parent, equal allocation  '
              f'{step1.mean_difference.iloc[0]:+.4f}')
        print(f'  synthetic, CROSS-VALIDATED instead               '
              f'{step2.mean_difference.iloc[0]:+.4f}'
              f'   (a CV half measures the KDE at n/2, and its advantage is a '
              f'large-n advantage)')
        print(f'  synthetic, CV and REWEIGHTED to the empirical mix '
              f'{step3:+.4f}')
        print(f'  empirical, CV, reweighted                        '
              f'{step4r:+.4f}   (equal allocation '
              f'{step4.mean_difference.iloc[0]:+.4f})')
    print()
    print('The criterion and the size mix account for the SIGN. A factor of about')
    print('two in the magnitude does not, and that is a genuine difference between')
    print('the corpus and the arm rather than an artifact of how either is read.')
    print('What both arms agree on, on every criterion, is the SHAPE: the KDE')
    print('loses at n = 10-99 and wins at n >= 1000.')
    print('  KDE minus lognormal, per band, all criteria, positive = KDE better:')
    for wt in ('Uniform', 'Variable'):
        pair = [f'KDE, {wt}', f'Lognormal, {wt}']
        for arm, frame, col in (('synthetic', syn, 'w1_parent'),
                                ('synthetic', syn, 'w1_cv'),
                                ('empirical', emp_s, 'w1_cv'),
                                ('empirical', emp_s, 'w1')):
            g = frame[frame.method.isin(pair) & frame[col].notna()]
            if not len(g):
                continue
            b = R.paired_bootstrap(g, col, f'KDE, {wt}', by=['size_band'],
                                   rng=rngb)
            bits = ' | '.join(
                f'{r.size_band[:4]} {r.mean_difference:+.4f}'
                f'{"*" if r.distinguishable else " "}' for _, r in b.iterrows())
            print(f'    {wt:<9}{arm:<10}{col:<10} {bits}')

    section('4. POST-STRATIFICATION to the empirical size mix')
    print('empirical size shares, measured: '
          + '  '.join(f'{k} {v:.4f}' for k, v in shares.items())
          + f'   (sum {sum(shares.values()):.4f}; the remainder is datasets '
            f'above the corpus maximum of n = 9,999)')
    for val, arm in (('w1_parent', 'synthetic'), ('w1', 'synthetic'),
                     ('w1_cv', 'empirical'), ('w1', 'empirical')):
        sub = scores[scores.arm == arm]
        if sub[val].notna().sum() == 0:
            continue
        ps = R.post_stratify(sub, val, shares)
        ps = ps[['method', 'mean_equal_allocation', 'mean_post_stratified']
                + [c for c in ps.columns if c.startswith('mean__')]]
        print(f'\n--- {arm}, {val} ---')
        print(fmt(ps.sort_values('mean_post_stratified')))
    print('\nMEAN RANK, the number Stage 2b section 4.11 said the two arms '
          'disagreed on:')
    sr = CMP.add_ranks(syn, 'w1_parent')
    er = CMP.add_ranks(emp_s[emp_s.w1_cv.notna()], 'w1_cv')
    a = R.post_stratify(sr, 'w1_parent_rank', shares)
    b = R.post_stratify(er, 'w1_cv_rank', shares)
    print(fmt(a[['method', 'mean_equal_allocation', 'mean_post_stratified']]
              .rename(columns={'mean_equal_allocation': 'syn_equal',
                               'mean_post_stratified': 'syn_reweighted'})
              .merge(b[['method', 'mean_equal_allocation',
                        'mean_post_stratified']]
                     .rename(columns={'mean_equal_allocation': 'emp_equal',
                                      'mean_post_stratified': 'emp_reweighted'}),
                     on='method')))

    section('5. REGRET: what it costs to use one method everywhere')
    for val, arm in (('w1_parent', 'synthetic'), ('w1_cv', 'empirical')):
        sub = scores[(scores.arm == arm) & scores[val].notna()]
        t = R.regret_table(sub, val)
        print(f'\n--- {arm}, {val} ---')
        print(fmt(t[['method', 'n_datasets', 'regret_mean', 'regret_p50',
                     'regret_p90', 'regret_p95', 'regret_max',
                     'regret_relative_mean', 'regret_relative_p90',
                     'regret_zero_share']].sort_values('regret_mean')))

    section('6. THE DECOMPOSITION: fit error against the definitional gap')
    t = dec.groupby(['arm', 'method']).agg(
        w1_total=('w1_total', 'mean'), w1_own=('w1_own', 'mean'),
        w1_definitional=('w1_definitional', 'mean'),
        w1_slack=('w1_slack', 'mean')).reset_index()
    t['definitional_share'] = t.w1_definitional / t.w1_total
    print(fmt(t.sort_values(['arm', 'w1_total'])))
    print('\nRanking the three estimation methods against their OWN weighting '
          'scheme, which is what removes the definitional term:')
    for arm in ('empirical', 'synthetic'):
        g = dec[dec.arm == arm]
        a = g.groupby('method').w1_total.mean().rank()
        b = g.groupby('method').w1_own.mean().rank()
        print(f'  {arm}: ')
        print(fmt(pd.DataFrame(dict(method=a.index, rank_vs_variable=a.values,
                                    rank_vs_own=b.values)).sort_values(
            'rank_vs_own')))

    section('7. OVERLAP AREA against W1, synthetic arm')
    w = syn.pivot_table(index='dataset', columns='method',
                        values=['w1_parent', 'ovl_loss'])
    agree = (w['w1_parent'].idxmin(axis=1) == w['ovl_loss'].idxmin(axis=1))
    print(f'  the two criteria pick the same winner on {100 * agree.mean():.1f} '
          f'percent of datasets')
    print(f'  Spearman correlation of the two per-dataset scores: '
          f'{syn[["w1_parent", "ovl_loss"]].corr("spearman").iloc[0, 1]:.4f}')
    print('  mean rank under each:')
    a = CMP.add_ranks(syn, 'w1_parent').groupby('method').w1_parent_rank.mean()
    b = CMP.add_ranks(syn, 'ovl_loss').groupby('method').ovl_loss_rank.mean()
    print(fmt(pd.DataFrame(dict(method=a.index, by_w1=a.values,
                                by_overlap=b.values)).sort_values('by_w1')))

    section('8. IS THE KDE ADVANTAGE ABOUT MULTIMODALITY?')
    print('Stage 2a-2 found 94.9 percent of empirical datasets have ONE visible')
    print('mode, so "the KDE wins because it can represent several modes" is an')
    print('assumption the data may not support. If the advantage is there in the')
    print('visibly unimodal majority, it is coming from skewness or tail shape,')
    print('and that is a different claim for the paper to make.')
    print()
    print('TWO THINGS HAVE TO BE CONTROLLED OR THE ANSWER IS AN ARTIFACT.')
    print('`modality.n_modes_visible` returns 1 below n = 8 without measuring')
    print('anything, so the whole of stratum 1 would be counted unimodal by fiat;')
    print('and multimodality is only RESOLVABLE at large n, where every method')
    print('does better, so an unconditional split confounds modality with size.')
    print('Datasets below n = 8 are therefore dropped and the split is reported')
    print('WITHIN each size band.')
    if 'n_modes_visible' not in chars.columns:
        print('  n_modes_visible is not available')
        return
    v = chars['n_modes_visible']
    s8 = syn.assign(modes=syn.dataset.map(v))
    s8 = s8[s8.n >= 8]
    s8['visibly_unimodal'] = s8.modes <= 1
    one = s8.drop_duplicates('dataset')
    print(f'\n  corpus at n >= 8: {len(one)} datasets, '
          f'{100 * one.visibly_unimodal.mean():.1f} pct visibly unimodal')
    print(f'  empirical arm:    {len(emp_modes)} datasets, '
          f'{100 * float((emp_modes <= 1).mean()):.1f} pct visibly unimodal '
          f'(all n, and {100 * float((emp_modes[[k for k in emp_modes.index]] <= 1).mean()):.1f} '
          f'pct is the figure Stage 2a-2 quotes)')
    for band in [b[0] for b in R.SIZE_BANDS]:
        g = s8[s8.size_band == band]
        if not len(g):
            continue
        print(f'\n  --- {band} ---')
        for flag, label in ((True, 'unimodal  '), (False, 'multimodal')):
            h = g[g.visibly_unimodal == flag]
            if h.dataset.nunique() < 20:
                print(f'    {label}  only {h.dataset.nunique()} datasets, '
                      f'not reported')
                continue
            wide = h.pivot_table(index='dataset', columns='method',
                                 values='w1_parent')
            wsh = (wide.idxmin(axis=1).value_counts(normalize=True)
                   .reindex(wide.columns).fillna(0.0))
            rk = wide.rank(axis=1).mean()
            print(f'    {label}  {len(wide):5d} datasets   '
                  + '  '.join(f'{c.split(",")[0][:4]}.{c.split(", ")[1][:3]} '
                              f'r{rk[c]:.2f}/w{wsh[c]:.2f}'
                              for c in wide.columns))
    print('\n  KDE win share against the lognormal, same weighting, at n >= 8:')
    for band in [b[0] for b in R.SIZE_BANDS]:
        g = s8[s8.size_band == band]
        if not len(g):
            continue
        wide = g.pivot_table(index='dataset', columns='method',
                             values='w1_parent')
        uni = g.drop_duplicates('dataset').set_index('dataset').visibly_unimodal
        bits = []
        for flag, label in ((True, 'uni'), (False, 'multi')):
            k = uni[uni == flag].index
            if len(k) < 20:
                bits.append(f'{label} n/a')
                continue
            for wt in ('Uniform', 'Variable'):
                a, b = f'KDE, {wt}', f'Lognormal, {wt}'
                bits.append(f'{label}/{wt[:3]} '
                            f'{100 * (wide.loc[k, a] < wide.loc[k, b]).mean():.0f}')
        print(f'    {band:16s} ' + '   '.join(bits))
    print('\n  LOCATION against SHAPE, within band, variable weighting:')
    t = (s8[s8.method.str.endswith('Variable')]
         .groupby(['size_band', 'visibly_unimodal', 'method'])
         .agg(location=('w1_parent_location', 'mean'),
              shape=('w1_parent_shape', 'mean'),
              total=('w1_parent', 'mean'),
              n=('dataset', 'nunique')).reset_index())
    print(fmt(t[t.n >= 20]))

    section('9. THE TAIL W1 DOES NOT SEE')
    t = syn.groupby('method').agg(
        body=('w1_parent', 'mean'), tail_mean=('w1_parent_tail', 'mean'),
        tail_max=('w1_parent_tail', 'max'),
        tail_p99=('w1_parent_tail', lambda s: s.quantile(0.99))).reset_index()
    t['tail_share_of_total'] = t.tail_mean / (t.body + t.tail_mean)
    print(fmt(t.sort_values('body')))
    bad = syn[syn.w1_parent_tail > syn.w1_parent]
    print(f'\n  fits whose unseen tail exceeds their whole body score: '
          f'{len(bad)} of {len(syn)} ({100 * len(bad) / len(syn):.2f} pct)')
    if len(bad):
        print(fmt(bad.method.value_counts().reset_index()))
    if 'model_sd_ratio' in syn.columns:
        print('  fitted model spread over the data\'s, max by method:')
        print(fmt(syn.groupby('method').model_sd_ratio.max().reset_index()))
    print('  rank correlation between the body score and the total: '
          f'{syn[["w1_parent", "w1_parent_total"]].corr("spearman").iloc[0, 1]:.4f}')
    a = CMP.add_ranks(syn, 'w1_parent').groupby('method').w1_parent_rank.mean()
    b = CMP.add_ranks(syn, 'w1_parent_total').groupby(
        'method').w1_parent_total_rank.mean()
    print(fmt(pd.DataFrame(dict(method=a.index, rank_body=a.values,
                                rank_with_tail=b.values)).sort_values(
        'rank_body')))


def report_only():
    """Redraw the report from the tables already on disk.

    The computation is the expensive part and the reading of it is the part that
    gets revised, so they are separable. Nothing here recomputes a score.
    """
    pd.set_option('display.width', 240)
    scores = pd.read_csv(os.path.join(TABLES, 'TABLE_TargetComparison.csv'))
    rec = pd.read_csv(os.path.join(TABLES, 'TABLE_RecoveryScores.csv'))
    cv = pd.read_csv(os.path.join(TABLES,
                                  'TABLE_CrossValidatedScores.csv.gz'))
    dec = pd.read_csv(os.path.join(TABLES,
                                   'TABLE_WeightingDecomposition.csv'))
    cvs = R.cv_summary(cv)
    met, vals, _ = corpus.load_corpus()
    chars = met.set_index('dataset')
    import modality as _md
    ids = set(scores[scores.arm == 'synthetic'].dataset)
    syn = corpus.as_dict(vals[vals.dataset_id.isin(ids)])
    chars['n_modes_visible'] = pd.Series(
        {k: _md.n_modes_visible(v[0]) for k, v in syn.items()})
    rng = np.random.default_rng(SEED)
    emp, _ = empirical.prepare(rng.spawn(1)[0])
    emp_modes = pd.Series({k: _md.n_modes_visible(v[0]) for k, v in emp.items()})
    report(scores, rec, cv, cvs, dec, chars, emp_modes)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        report_only()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
