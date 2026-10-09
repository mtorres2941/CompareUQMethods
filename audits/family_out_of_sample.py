"""Does the three-parameter lognormal earn its place? Gamma, out of sample.

Stage 2c task 0d, and the question Stage 2b's own assessment said a reviewer
would ask. The study's "Lognormal" is a three-parameter lognormal whose
threshold is chosen by profile likelihood, because the likelihood is unbounded
as the threshold approaches min(x) and the global maximum does not exist
(decision 51). The guard `PROFILE_DELTA_LO_FRAC = 0.25` therefore SETS the
threshold for about half the empirical arm, which means for half the arm the
number is a scale-aware heuristic rather than an estimate.

Gamma has no threshold, no pathology and no guard. Stage 2b measured that it
beats the three-parameter lognormal on the datasets where the guard binds. If
that survives an out-of-sample test, the lognormal's third parameter is buying
in-sample flexibility and nothing else, and the paper should say so.

`audits/family_comparison.py` already compares the families IN SAMPLE, under
both maximum likelihood and direct W1 minimization. The missing piece is the
one Stage 2b named: W1 has no complexity penalty and these families run from two
parameters to three, so an in-sample comparison cannot settle it. Two
out-of-sample criteria are used here and they are the stage's two targets:

  SYNTHETIC   W1 against the known parent. Exact, no training data in it.
  EMPIRICAL   cross-validated W1, ten random half-splits in both directions.

Reported overall, by size band, and split on whether the profile guard bound the
threshold, because that is where Stage 2b found the difference.

    conda run -n compareuq python audits/family_out_of_sample.py [n_synth]
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
import empirical  # noqa: E402
import fitting as FT  # noqa: E402
import mixture as M  # noqa: E402
import recovery as R  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
SEED = 42
N_SYNTH = 2_000
CV_REPEATS = 10

#: The families compared. `lognormal_offset` is the Stage 1 method, kept so the
#: comparison includes what the manuscript currently describes.
FAMILIES = ('normal', 'lognormal_2p', 'lognormal_3p', 'lognormal_offset',
            'gamma')

#: Number of free parameters, for the complexity argument the comparison makes.
N_PARAMS = {'normal': 2, 'lognormal_2p': 2, 'lognormal_3p': 3,
            'lognormal_offset': 2, 'gamma': 2}


def fit_families(x, w):
    """One model per family, plus the lognormal's threshold status."""
    models, status = {}, None
    for fam in FAMILIES:
        try:
            m, p = FT.fit_family(fam, x, w)
        except Exception:
            continue
        models[fam] = m
        if fam == 'lognormal_3p':
            status = p.get('status')
    return models, status


def synthetic_rows(name, x, w, parent):
    grid = R.recovery_grid(x, w, parent)
    rows = []
    for wt, ww in (('Uniform', FT.uniform_weights(x)), ('Variable', w)):
        scheme = R.OWN_TARGET_SCHEME[wt]
        models, status = fit_families(x, ww)
        for fam, m in models.items():
            tail, _ = R.tail_charge(m, parent, grid)
            rows.append(dict(
                arm='synthetic', dataset=name, n=len(x), weighting=wt,
                family=fam, n_params=N_PARAMS[fam], guard_status=status,
                w1_own_target=R.w1_against_parent(m, parent, scheme, grid),
                w1_own_target_tail=tail,
                w1_in_sample=FT.score_w1_model(m, x, w)))
    return rows


def empirical_rows(name, x, w, rng, repeats=CV_REPEATS):
    """Cross-validated W1 per family, plus the in-sample score for contrast."""
    rows = []
    _, status = fit_families(x, FT.uniform_weights(x))
    acc = {}
    if len(x) >= R.CV_MIN_N:
        for fit_idx, score_idx in R.cv_splits(len(x), rng, repeats):
            xf, xs = x[fit_idx], x[score_idx]
            wf, ws = w[fit_idx], w[score_idx]
            if len(xf) < 3 or len(xs) < 3 or np.ptp(xf) <= 0:
                continue
            for wt, (wwf, wws) in (
                    ('Uniform', (FT.uniform_weights(xf), FT.uniform_weights(xs))),
                    ('Variable', (wf / wf.sum(), ws / ws.sum()))):
                models, _ = fit_families(xf, wwf)
                grid = FT.score_grid_open(xs, wws)
                for fam, m in models.items():
                    try:
                        v = FT.score_w1_model(m, xs, wws, grid=grid)
                    except Exception:
                        continue
                    acc.setdefault((wt, fam), []).append(v)
    for wt, ww in (('Uniform', FT.uniform_weights(x)), ('Variable', w)):
        models, _ = fit_families(x, ww)
        for fam, m in models.items():
            v = acc.get((wt, fam), [])
            rows.append(dict(
                arm='empirical', dataset=name, n=len(x), weighting=wt,
                family=fam, n_params=N_PARAMS[fam], guard_status=status,
                w1_cv=float(np.mean(v)) if v else np.nan,
                w1_cv_sd=float(np.std(v)) if len(v) > 1 else np.nan,
                w1_in_sample=FT.score_w1_model(m, x, ww if wt == 'Uniform' else w)))
    return rows


def fmt(d):
    return d.to_string(float_format=lambda v: f'{v:.4f}')


def report(d):
    pd.set_option('display.width', 220)
    e = d[d.arm == 'empirical']
    s = d[d.arm == 'synthetic']

    print('=' * 78)
    print('OUT OF SAMPLE, empirical arm: cross-validated W1 by family')
    print('=' * 78)
    for wt in ('Uniform', 'Variable'):
        g = e[(e.weighting == wt) & e.w1_cv.notna()]
        t = g.groupby('family').agg(
            n_params=('n_params', 'first'), cv_mean=('w1_cv', 'mean'),
            cv_median=('w1_cv', 'median'),
            cv_p90=('w1_cv', lambda v: v.quantile(0.90)),
            in_sample_mean=('w1_in_sample', 'mean'),
            n=('dataset', 'nunique')).sort_values('cv_mean')
        print(f'--- {wt} weighting ---')
        print(fmt(t))
        wide = g.pivot_table(index='dataset', columns='family', values='w1_cv')
        if {'gamma', 'lognormal_3p'} <= set(wide.columns):
            print(f'  gamma beats the 3-parameter lognormal on '
                  f'{100 * (wide.gamma < wide.lognormal_3p).mean():.1f} pct '
                  f'of datasets')
        print()

    print('=' * 78)
    print('OUT OF SAMPLE, synthetic arm: W1 against the known parent')
    print('=' * 78)
    for wt in ('Uniform', 'Variable'):
        g = s[s.weighting == wt]
        t = g.groupby('family').agg(
            n_params=('n_params', 'first'), parent_mean=('w1_own_target', 'mean'),
            parent_median=('w1_own_target', 'median'),
            parent_p90=('w1_own_target', lambda v: v.quantile(0.90)),
            tail_mean=('w1_own_target_tail', 'mean'),
            in_sample_mean=('w1_in_sample', 'mean'),
            n=('dataset', 'nunique')).sort_values('parent_mean')
        print(f'--- {wt} weighting ---')
        print(fmt(t))
        wide = g.pivot_table(index='dataset', columns='family',
                             values='w1_own_target')
        if {'gamma', 'lognormal_3p'} <= set(wide.columns):
            print(f'  gamma beats the 3-parameter lognormal on '
                  f'{100 * (wide.gamma < wide.lognormal_3p).mean():.1f} pct '
                  f'of datasets')
        print()

    print('=' * 78)
    print('WHERE THE GUARD BINDS, which is where Stage 2b saw the difference')
    print('=' * 78)
    for arm, col in (('empirical', 'w1_cv'), ('synthetic', 'w1_own_target')):
        g = d[(d.arm == arm) & (d.weighting == 'Variable') & d[col].notna()]
        if not len(g):
            continue
        counts = g.drop_duplicates('dataset').guard_status.value_counts(
            normalize=True)
        print(f'--- {arm}, by `status` of the profile fit ---')
        print('  share of datasets: '
              + '  '.join(f'{k} {100 * v:.1f} pct' for k, v in counts.items()))
        t = g.pivot_table(index=['guard_status', 'dataset'], columns='family',
                          values=col)
        for status, h in t.groupby('guard_status'):
            if {'gamma', 'lognormal_3p'} <= set(h.columns):
                print(f'  {status:<22} n={len(h):5d}  '
                      f'gamma {h.gamma.mean():.4f}  '
                      f'lognormal_3p {h.lognormal_3p.mean():.4f}  '
                      f'gamma wins {100 * (h.gamma < h.lognormal_3p).mean():5.1f} pct')
        print()

    print('=' * 78)
    print('IS ANY OF IT DISTINGUISHABLE? Paired bootstrap against lognormal_3p')
    print('=' * 78)
    print('Positive = the three-parameter lognormal scored LOWER, so better.')
    for arm, col in (('empirical', 'w1_cv'), ('synthetic', 'w1_own_target')):
        for wt in ('Uniform', 'Variable'):
            g = d[(d.arm == arm) & (d.weighting == wt) & d[col].notna()]
            if not len(g):
                continue
            b = R.paired_bootstrap(g.rename(columns={'family': 'method'}), col,
                                   'lognormal_3p',
                                   rng=np.random.default_rng(0))
            print(f'--- {arm}, {wt}, {int(b.n_datasets.max())} datasets ---')
            print(fmt(b[['method', 'mean_difference', 'ci_lo', 'ci_hi',
                         'reference_wins', 'distinguishable']]
                      .set_index('method')))
            print()

    print('=' * 78)
    print('BY SIZE BAND, variable weighting')
    print('=' * 78)
    for arm, col in (('empirical', 'w1_cv'), ('synthetic', 'w1_own_target')):
        g = d[(d.arm == arm) & (d.weighting == 'Variable') & d[col].notna()]
        if not len(g):
            continue
        g = g.assign(band=g.n.map(R.size_band))
        t = g.pivot_table(index='family', columns='band', values=col,
                          aggfunc='mean')
        print(f'--- {arm}, mean {col} ---')
        print(fmt(t))
        print()


def main(n_synth=N_SYNTH):
    os.makedirs(TABLES, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    emp, _ = empirical.prepare(rng.spawn(1)[0])
    rows = []
    cvrng = np.random.default_rng(7)
    print(f'empirical: {len(emp)} datasets', flush=True)
    for i, (name, (x, w)) in enumerate(emp.items()):
        rows += empirical_rows(name, np.asarray(x, float),
                               np.asarray(w, float), cvrng)
        if (i + 1) % 25 == 0:
            print(f'  {i+1}/{len(emp)}  {time.time()-t0:.0f}s', flush=True)

    met, vals, _ = corpus.load_corpus()
    ids = sorted(met.sample(n_synth, random_state=0).dataset.astype(str))
    syn = corpus.as_dict(vals[vals.dataset_id.isin(set(ids))])
    specs = corpus.load_parent_specs()
    print(f'synthetic: {len(ids)} datasets', flush=True)
    for i, name in enumerate(ids):
        x, w = syn[name]
        rows += synthetic_rows(name, np.asarray(x, float), np.asarray(w, float),
                               M.parent_from_spec(specs[name]))
        if (i + 1) % 250 == 0:
            print(f'  {i+1}/{len(ids)}  {time.time()-t0:.0f}s', flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TABLES, 'TABLE_FamilyOutOfSample.csv'), index=False)
    print(f'wrote TABLE_FamilyOutOfSample.csv ({len(d)} rows), '
          f'{time.time()-t0:.0f}s\n')
    report(d)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_SYNTH)
