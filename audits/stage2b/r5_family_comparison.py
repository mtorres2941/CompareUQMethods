"""Every family, both weightings, fitted by MLE and by W1. And the truncation.

Stage 2b. Four questions that share one loop, so they share one script.

Q1. TRUNCATION AND RENORMALIZATION. Decision 13, confirmed by the author: every
    method lives on (0, inf), open at zero. The normal puts mass below zero
    directly and the KDE puts it there through Gaussian kernels on small values.
    How much does it cost the normal to be charged for mass it is never allowed
    to occupy? The comparison is the untruncated normal against the truncated,
    renormalized one, both scored on the same grid.

Q2. WHAT THE STAGE 1 SCORING WAS ACTUALLY DOING. The old grid started at
    exactly zero, so the model was already implicitly truncated and
    renormalized -- the manuscript's numbers are truncated-normal numbers even
    though the text describes an untruncated normal. Q1's answer is therefore
    the size of a claim the paper makes about itself, and this is how much the
    reported numbers move when the truncation is made explicit.

Q3. THE FAMILIES. Two-parameter lognormal with no offset, the +0.5 offset,
    profile-likelihood three-parameter lognormal, gamma, normal, KDE.

Q4. FITTING BY THE CRITERION WE SCORE BY. Every parametric family under both
    maximum likelihood and direct W1 minimization. If the KDE still wins against
    W1-optimally fitted parametric families the finding is far stronger; if it
    does not, the author needs to know before a reviewer says so.

    conda run -n compareuq python audits/stage2b/r5_family_comparison.py [n_synth]
"""
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import corpus  # noqa: E402
import empirical  # noqa: E402
import families as F  # noqa: E402
import fitting as FT  # noqa: E402
from customstats import weighted_ecdf  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'stage2b')
SEED = 42
N_SYNTH = 2000


def one_dataset(name, x, w_var, arm):
    """Every family, both weightings, both estimators, for one dataset."""
    rows = []
    grid = FT.score_grid_open(x, w_var)
    for wt, w in (('Uniform', FT.uniform_weights(x)), ('Variable', w_var)):
        for fam in FT.PARAMETRIC_FAMILIES:
            base = dict(arm=arm, dataset=name, n=len(x), weighting=wt,
                        family=fam)
            for method in ('mle', 'w1'):
                t0 = time.perf_counter()
                try:
                    m, p = FT.fit_family(fam, x, w_var if wt == 'Variable'
                                         else FT.uniform_weights(x), method)
                    # Scored against the VARIABLE-weighted empirical CDF, for
                    # every fit including the uniform-weighted ones. That is the
                    # study's scoring rule and it is not changed here.
                    d = FT.score_w1_model(m, x, w_var, grid=grid)
                except Exception as exc:                      # pragma: no cover
                    d, p = np.nan, {'error': repr(exc)}
                rows.append(dict(base, method=method, w1=d,
                                 seconds=time.perf_counter() - t0,
                                 status=p.get('status', ''),
                                 mass_below_zero=(m.mass_below
                                                  if np.isfinite(d) else np.nan),
                                 threshold=p.get('loc', np.nan)))
        # KDE has no parametric estimator to optimize; its one choice is the
        # bandwidth, which is Stage 2h's.
        mk, pk = FT.fit_kde(x, w if wt == 'Uniform' else w_var)
        rows.append(dict(arm=arm, dataset=name, n=len(x), weighting=wt,
                         family='kde', method='mle',
                         w1=FT.score_w1_model(mk, x, w_var, grid=grid),
                         seconds=np.nan, status='',
                         mass_below_zero=mk.mass_below, threshold=np.nan))
    return rows


#: Points in the dense lattice used for the truncation comparison only. The
#: study's own criterion stays at SCORE_GRID_POINTS; this is finer so that the
#: two models can be compared on ONE lattice spanning both supports, which a
#: 1,000-point grid whose span depends on the model cannot do.
#:
#: 20,001 and not more because the KDE's CDF costs `npoints * n` and the arm
#: reaches n = 31,025. The number matters only through the resolution of the
#: integral, and BOTH models in each comparison are integrated on the same
#: lattice, so it cancels out of the difference that is being measured.
EXACT_POINTS = 20_001


def w1_exact(cdf, x, w, lo, hi, npoints=EXACT_POINTS):
    """W1 = integral of |F_model - F_empirical|, both evaluated on one lattice.

    The empirical CDF is a step function and the model CDF is closed form for
    every family here, so this needs no discretization of the MODEL onto grid
    points and no renormalization by the grid. That is what makes it a fair
    comparison between a model on (0, inf) and one on the whole real line: the
    untruncated model is charged for the mass it puts below zero, because the
    lattice reaches down to where that mass is.
    """
    g = np.linspace(lo, hi, npoints)
    e = weighted_ecdf(x, w)[2](g)
    return float(np.trapezoid(np.abs(np.asarray(cdf(g), float) - e), g))


def truncation_rows(name, x, w_var, arm):
    """The normal and the KDE, truncated against untruncated, on ONE lattice.

    The question the author asked: how much does it cost a model to be charged
    for mass in a region it is never allowed to occupy? The comparison is only
    meaningful if both models are scored the same way, so both are integrated
    exactly against the empirical CDF on a lattice that spans both supports.

    `w1_stage1` is the Stage 1 number, kept in the same row, because the second
    question is what the manuscript's reported numbers actually were.
    """
    out = []
    spread = max(np.std(x), float(np.sqrt(
        (w_var / w_var.sum()) @ (x - x @ (w_var / w_var.sum())) ** 2)))
    hi = float(x.max() + FT.SCORE_GRID_STD_MULTIPLE * spread)
    for wt, w in (('Uniform', FT.uniform_weights(x)), ('Variable', w_var)):
        pn = F.fit_normal_mle(x, w)
        parent = norm(loc=pn['loc'], scale=pn['scale'])
        trunc = F.make_normal(pn)
        lo = float(min(0.0, parent.ppf(1e-9)))
        hi_n = float(max(hi, parent.ppf(1 - 1e-9)))
        out.append(dict(
            arm=arm, dataset=name, n=len(x), weighting=wt, family='normal',
            mass_below_zero=trunc.mass_below,
            w1_untruncated=w1_exact(parent.cdf, x, w_var, lo, hi_n),
            w1_truncated=w1_exact(trunc.cdf, x, w_var, lo, hi_n),
            w1_stage1=FT.score_w1(parent, x, w_var)))

        bw = FT.weighted_bw(x, w, bw_method=FT.BW_METHOD)
        kparent = F.WeightedKDE(x, w, bw)
        ktrunc = F.Truncated(kparent, label='kde')
        lo_k = float(min(0.0, x.min() - 10.0 * bw))
        hi_k = float(max(hi, x.max() + 10.0 * bw))
        out.append(dict(
            arm=arm, dataset=name, n=len(x), weighting=wt, family='kde',
            mass_below_zero=ktrunc.mass_below,
            w1_untruncated=w1_exact(kparent.cdf, x, w_var, lo_k, hi_k),
            w1_truncated=w1_exact(ktrunc.cdf, x, w_var, lo_k, hi_k),
            w1_stage1=np.nan))
    return out


def report(d, label):
    print()
    print('=' * 78)
    print(label)
    print('=' * 78)
    for agg in ('mean', 'median'):
        piv = (d.pivot_table(index=['family', 'method'], columns='weighting',
                             values='w1', aggfunc=agg).sort_values('Variable'))
        print(f'{agg.upper()} W1 (lower is better)')
        print(piv.to_string(float_format=lambda v: f'{v:.5f}'))
        print()
    print()
    # Mean rank across the methods a paper would actually put side by side:
    # each family at its best estimator, plus the KDE.
    best = (d.sort_values('w1').groupby(['dataset', 'weighting', 'family'])
            .first().reset_index())
    wide = best.pivot_table(index=['dataset'], columns=['family', 'weighting'],
                            values='w1')
    ranks = wide.rank(axis=1).mean().sort_values()
    print('MEAN RANK over the 12 (family, weighting) pairs, best estimator each')
    print(ranks.to_string(float_format=lambda v: f'{v:.3f}'))
    print()
    mle = d[d.method == 'mle'].set_index(['dataset', 'weighting', 'family']).w1
    w1o = d[d.method == 'w1'].set_index(['dataset', 'weighting', 'family']).w1
    gain = ((mle - w1o) / mle * 100).dropna()
    print('W1-OPTIMAL FITTING AGAINST MLE, percent reduction in W1')
    g = gain.groupby('family')
    print(pd.DataFrame(dict(median_pct=g.median(), mean_pct=g.mean(),
                            improved_pct=g.apply(lambda s: (s > 1e-9).mean() * 100)
                            )).to_string(float_format=lambda v: f'{v:.2f}'))


def main(n_synth):
    os.makedirs(TABLES, exist_ok=True)
    rows, trows = [], []

    ds, _ = empirical.prepare(np.random.default_rng(SEED).spawn(1)[0])
    print(f'empirical arm: {len(ds)} datasets', flush=True)
    for i, (name, (x, w)) in enumerate(ds.items()):
        rows += one_dataset(name, x, w, 'empirical')
        trows += truncation_rows(name, x, w, 'empirical')
        if (i + 1) % 25 == 0:
            print(f'  {i+1}/{len(ds)}', flush=True)

    met, vals, _ = corpus.load_corpus()
    pick = met.sample(n_synth, random_state=0).dataset.astype(str)
    syn = corpus.as_dict(vals[vals.dataset_id.isin(set(pick))])
    print(f'synthetic: {len(syn)} datasets sampled from the corpus', flush=True)
    for i, (name, (x, w)) in enumerate(syn.items()):
        rows += one_dataset(name, x, w, 'synthetic')
        trows += truncation_rows(name, x, w, 'synthetic')
        if (i + 1) % 250 == 0:
            print(f'  {i+1}/{len(syn)}', flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TABLES, 'TABLE_2b_FamilyComparison.csv'), index=False)
    t = pd.DataFrame(trows)
    t.to_csv(os.path.join(TABLES, 'TABLE_2b_TruncationEffect.csv'), index=False)

    for arm in ('empirical', 'synthetic'):
        report(d[d.arm == arm], f'{arm.upper()} ARM')

    print()
    print('=' * 78)
    print('TRUNCATION AND RENORMALIZATION ONTO (0, inf)')
    print('=' * 78)
    t['gain_pct'] = (t.w1_untruncated - t.w1_truncated) / t.w1_untruncated * 100
    for arm in ('empirical', 'synthetic'):
        g = t[t.arm == arm]
        print(f'--- {arm} ---')
        s = g.groupby(['family', 'weighting']).agg(
            mean_mass_below=('mass_below_zero', 'mean'),
            max_mass_below=('mass_below_zero', 'max'),
            mean_w1_untrunc=('w1_untruncated', 'mean'),
            mean_w1_trunc=('w1_truncated', 'mean'),
            median_w1_untrunc=('w1_untruncated', 'median'),
            median_w1_trunc=('w1_truncated', 'median'),
            median_gain_pct=('gain_pct', 'median'),
            improved_pct=('gain_pct', lambda s: (s > 0).mean() * 100))
        print(s.to_string(float_format=lambda v: f'{v:.5f}'))
        n = g[(g.family == 'normal')]
        print(f'  datasets where the normal puts more than 1 pct of its mass '
              f'below zero: {int((n.mass_below_zero > 0.01).sum())} of {len(n)} '
              f'({(n.mass_below_zero > 0.01).mean()*100:.1f} pct)')
        print(f'  more than 10 pct: {int((n.mass_below_zero > 0.10).sum())} '
              f'({(n.mass_below_zero > 0.10).mean()*100:.1f} pct)')
        print()

    print('WHAT THE STAGE 1 SCORING WAS DOING, normal only')
    n = t[t.family == 'normal'].dropna(subset=['w1_stage1'])
    n = n.assign(vs_trunc=(n.w1_stage1 - n.w1_truncated).abs() / n.w1_truncated,
                 vs_untrunc=(n.w1_stage1 - n.w1_untruncated).abs()
                 / n.w1_untruncated)
    print(f'  Stage 1 W1 against the TRUNCATED score:   median relative '
          f'difference {n.vs_trunc.median()*100:.3f} pct')
    print(f'  Stage 1 W1 against the UNTRUNCATED score: median relative '
          f'difference {n.vs_untrunc.median()*100:.3f} pct')
    print('  The old grid started at exactly zero, so Stage 1 was already')
    print('  scoring a truncated, renormalized normal without saying so. The')
    print('  manuscript describes an untruncated one. Making it explicit moves')
    print('  the reported numbers by the first figure, not the second.')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_SYNTH)
