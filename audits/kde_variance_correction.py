"""If the KDE loses at small n because it is OVER-DISPERSED, does fixing that win?

A PROBE, NOT A PROPOSAL. Stage 2c must not change the fitting families
(decision 52) and does not. This measures whether a candidate Stage 2h should
test would change the paper's conclusion, so that 2h knows whether it is worth
its time and the author knows what is on the table.

THE MECHANISM IT ATTACKS. A Gaussian KDE is the data convolved with a kernel, so
its variance is the data's PLUS h^2. With a rule-of-thumb bandwidth
h = c * s * n_eff ** -0.2 that inflation is large exactly where the KDE loses:
the fitted model's standard deviation over the data's is 1.63 at n = 3-9 and
1.19 at n = 10-99, against 1.04 and 1.02 for the normal. The lognormal and the
normal match the data's spread by construction; the KDE cannot.

THE CORRECTION is standard and has one line of algebra. Shrink the points toward
their weighted mean by a = s / sqrt(s^2 - h^2) before placing the kernels, and
the resulting density has variance exactly s^2 again. It needs h < s, which every
rule here satisfies, and it changes nothing at large n where h << s.

    conda run -n compareuq python audits/kde_variance_correction.py [n_synth]
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
import families as F  # noqa: E402
import fitting as FT  # noqa: E402
import mixture as M  # noqa: E402
import recovery as R  # noqa: E402
from customstats import weighted_bw, weighted_std  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
SEED = 42
N_SYNTH = 2_000


def variance_corrected_kde(x, w, bw_method=FT.BW_METHOD):
    """A KDE whose density has the data's weighted variance, not more.

    Returns (model, bandwidth, shrink). `shrink` is 1.0 when the correction is
    not applicable, which happens only if the bandwidth reaches the data's own
    standard deviation.
    """
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    w = w / w.sum()
    h = weighted_bw(x, w, bw_method=bw_method)
    s = weighted_std(x, w)
    if not (s > 0) or h >= s:
        return F.Truncated(F.WeightedKDE(x, w, h), label='kde'), h, 1.0
    mu = float(np.sum(w * x))
    a = s / np.sqrt(s * s - h * h)
    y = mu + (x - mu) / a
    return F.Truncated(F.WeightedKDE(y, w, h), label='kde'), h, float(a)


def main(n_synth=N_SYNTH):
    os.makedirs(TABLES, exist_ok=True)
    t0 = time.time()
    met, vals, _ = corpus.load_corpus()
    ids = sorted(met.sample(n_synth, random_state=0).dataset.astype(str))
    syn = corpus.as_dict(vals[vals.dataset_id.isin(set(ids))])
    specs = corpus.load_parent_specs()

    rows = []
    for i, name in enumerate(ids):
        x, w = syn[name]
        x = np.asarray(x, float)
        w = np.asarray(w, float)
        parent = M.parent_from_spec(specs[name])
        grid = R.recovery_grid(x, w, parent)
        models, _ = FT.fit_pewt(x, w)
        for wt, ww in (('Uniform', FT.uniform_weights(x)), ('Variable', w)):
            scheme = R.PARENT_SCHEME[wt]
            base = models[f'KDE, {wt}']
            corrected, h, a = variance_corrected_kde(x, ww)
            logn = models[f'Lognormal, {wt}']
            # A pure-Silverman KDE too, because the guard SWITCHES TO SCOTT
            # below n_eff = 30 and the median effective sample size in the
            # n = 10-99 band is 31 under uniform weights and 11 under variable.
            # So the band where the KDE loses is largely the band where it is
            # running Scott, and the guard could be causing the loss it was not
            # chosen to cause.
            h_sil = weighted_bw(x, ww, bw_method='silverman')
            sil = F.Truncated(F.WeightedKDE(x, ww, h_sil), label='kde')
            vc_sil, _, a_sil = variance_corrected_kde(x, ww,
                                                      bw_method='silverman')
            rows.append(dict(
                dataset=name, n=len(x), weighting=wt, shrink=a,
                h_over_sd=h / weighted_std(x, ww),
                guard_bound=bool(h == weighted_bw(x, ww, bw_method='scott')),
                kde=R.w1_against_parent(base, parent, scheme, grid),
                kde_vc=R.w1_against_parent(corrected, parent, scheme, grid),
                kde_silverman=R.w1_against_parent(sil, parent, scheme, grid),
                kde_silverman_vc=R.w1_against_parent(vc_sil, parent, scheme,
                                                     grid),
                lognormal=R.w1_against_parent(logn, parent, scheme, grid)))
        if (i + 1) % 250 == 0:
            print(f'  {i+1}/{len(ids)}  {time.time()-t0:.0f}s', flush=True)

    d = R.add_size_band(pd.DataFrame(rows))
    d.to_csv(os.path.join(TABLES, 'TABLE_KDEVarianceCorrection.csv'),
             index=False)
    pd.set_option('display.width', 220)
    print()
    print('=' * 78)
    print('W1 AGAINST THE PARENT: plain KDE, variance-corrected KDE, lognormal')
    print('=' * 78)
    t = d.groupby(['weighting', 'size_band'])[
        ['kde', 'kde_vc', 'kde_silverman', 'kde_silverman_vc', 'lognormal',
         'shrink', 'guard_bound']].mean()
    t['vc_beats_kde'] = d.groupby(['weighting', 'size_band']).apply(
        lambda g: float((g.kde_vc < g.kde).mean()), include_groups=False)
    t['vc_beats_lognormal'] = d.groupby(['weighting', 'size_band']).apply(
        lambda g: float((g.kde_vc < g.lognormal).mean()), include_groups=False)
    t['kde_beats_lognormal'] = d.groupby(['weighting', 'size_band']).apply(
        lambda g: float((g.kde < g.lognormal).mean()), include_groups=False)
    print(t.to_string(float_format=lambda v: f'{v:.4f}'))
    print()
    print('THE LOGNORMAL MINUS EACH KDE VARIANT, paired against the parent.')
    print('POSITIVE MEANS THE KDE VARIANT IS BETTER, because these are')
    print('distances and the lower one wins. `*` = interval excludes zero.')
    for wt in ('Uniform', 'Variable'):
        for band in [b[0] for b in R.SIZE_BANDS]:
            g = d[(d.weighting == wt) & (d.size_band == band)]
            if len(g) < 20:
                continue
            long = pd.concat([
                g[['dataset']].assign(method=k, arm='s', w1=g[k])
                for k in ('kde', 'kde_vc', 'kde_silverman',
                          'kde_silverman_vc', 'lognormal')])
            b = R.paired_bootstrap(long, 'w1', 'lognormal',
                                   rng=np.random.default_rng(0)).set_index(
                'method')
            bits = '  '.join(
                f'{k:<17}{-b.loc[k, "mean_difference"]:+.4f}'
                f'{"*" if b.loc[k, "distinguishable"] else " "}'
                for k in ('kde', 'kde_silverman', 'kde_vc',
                          'kde_silverman_vc'))
            print(f'  {wt:<9} {band:<14} {bits}')
    print()
    print('The shrink factor is 1.0 where the correction does nothing and rises')
    print('as the bandwidth approaches the data\'s own spread. It is a PROBE:')
    print('nothing here is adopted and Stage 2h owns the decision.')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_SYNTH)
