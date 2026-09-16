"""What is the +0.5 offset actually solving, and what does the code actually do?

an earlier revision. Three questions, in order.

QUESTION 1, WHAT THE CODE DOES. The Methods text is self-contradictory: it says
shape, location and scale are all estimated, and then says location is held at
zero. Neither is what runs. This script prints the answer from the code itself
rather than from a reading of it.

QUESTION 2, WHICH PROBLEM THE OFFSET WAS SOLVING. There are two candidates and
they are different failures with different fixes.

  A. NEAR-ZERO VALUES. A handful of values orders of magnitude below the mean
     drag the log-space mean down and inflate sigma, so the fitted mode
     collapses toward zero. This afflicts the TWO-parameter lognormal and needs
     no threshold to appear.
  B. THE UNBOUNDED LIKELIHOOD. For a three-parameter lognormal the likelihood
     diverges as the threshold approaches the smallest observation from below,
     so the global MLE does not exist and a naive optimizer returns whatever it
     stopped at. This afflicts the THREE-parameter fit only.

The test is direct: run the no-offset two-parameter fit on the empirical
datasets with and without their surviving near-zero values, and see whether the
offset's advantage survives their removal. If it does not, the offset was
solving A.

QUESTION 3, HOW MUCH THE OFFSET WAS WORTH. Whether this was a real problem or a
cosmetic one, in W1 and in head-to-head wins.

Near-zero values are far rarer than they were. the original code measured 28.3 percent of
138 empirical datasets with a minimum below 1 percent of their mean; the
symmetric log-space cleaning of an earlier revision and the plausibility ceiling of
an earlier revision bring that to the figure this script prints. They have not gone away.

    conda run -n compareuq python audits/lognormal_offset.py
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

import empirical  # noqa: E402
import families as F  # noqa: E402
import fitting as FT  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
SEED = 42

#: A value below this fraction of the dataset mean is "near zero". Every
#: dataset is normalized to an unweighted mean of exactly 1, so this is a
#: fraction of the mean by construction. 1 percent is the threshold the original code
#: used to size the problem, kept so the two measurements are comparable.
NEAR_ZERO_FRAC = 0.01


def fit_lognorm3_naive(x, w, maxiter=2000):
    """A three-parameter lognormal fit the way a naive optimizer would do it.

    Joint Nelder-Mead over (log(min(x) - gamma), mu, log sigma), started at the
    two-parameter fit, with NO lower guard on how close gamma may come to
    min(x). This is not a method anyone should use. It is here to show what the
    unbounded likelihood does when nothing stops it, which is the thing the
    profile-likelihood treatment exists to prevent.
    """
    from scipy.optimize import minimize
    x, w = np.asarray(x, float), np.asarray(w, float) / np.sum(w)
    xmin = float(x.min())
    p0 = F.fit_lognorm2_mle(x, w)
    sd = float(np.sqrt(w @ (x - (x @ w)) ** 2))
    z0 = np.array([np.log(max(xmin * 0.5, 1e-12)), np.log(p0['scale']),
                   np.log(p0['s'])])

    def nll(z):
        gamma = xmin - np.exp(z[0])
        d = x - gamma
        if np.any(d <= 0) or not np.isfinite(z).all():
            return np.inf
        sigma = float(np.exp(z[2]))
        if not sigma > 0:
            return np.inf
        ll = -np.log(d) - np.log(sigma) - 0.5 * ((np.log(d) - z[1]) / sigma) ** 2
        v = -float(w @ ll)
        return v if np.isfinite(v) else np.inf

    res = minimize(nll, z0, method='Nelder-Mead',
                   options=dict(maxiter=maxiter, xatol=1e-12, fatol=1e-14))
    gamma = float(xmin - np.exp(res.x[0]))
    return dict(s=float(np.exp(res.x[2])), loc=gamma,
                scale=float(np.exp(res.x[1])),
                gap_over_sd=float((xmin - gamma) / sd),
                neg_loglik=float(res.fun))


def w1(model, x, w):
    return FT.score_w1_model(model, x, w)


def main():
    os.makedirs(TABLES, exist_ok=True)
    ds, _ = empirical.prepare(np.random.default_rng(SEED).spawn(1)[0])

    print('=' * 74)
    print('1. WHAT THE CODE DOES WITH THE LOGNORMAL THRESHOLD')
    print('=' * 74)
    x, w = ds['Cement']
    shape, loc, scale = FT.weighted_lognorm_fit(x + FT.LOGFIT_OFFSET,
                                                weights=w, method='MLE')
    print(f'customstats.weighted_lognorm_fit returns loc = {loc!r}')
    print('  Its docstring says so outright: "loc : float -- Location parameter')
    print('  (always 0 in this fit)". It optimizes over (sigma, mu) ONLY, and')
    print('  there is no threshold in its likelihood at all.')
    print()
    print(f'fitting.fit_pewt_models then builds  lognorm(s={shape:.4f}, '
          f'loc={loc - FT.LOGFIT_OFFSET:g}, scale={scale:.4f})')
    print(f'  so the fitted THRESHOLD is a CONSTANT, -{FT.LOGFIT_OFFSET:g}, '
          f'set by hand and never estimated.')
    print()
    print('  VERDICT. The Methods text is wrong in BOTH of its statements. The')
    print('  location is not estimated, and it is not held at zero either: it')
    print(f'  is held at -{FT.LOGFIT_OFFSET:g}. The fitted family is a')
    print('  three-parameter lognormal with two free parameters and a threshold')
    print('  fixed at a value chosen by hand.')
    print()
    print('  A CONSEQUENCE WORTH STATING SEPARATELY. Because the threshold is')
    print('  never estimated, the unbounded-likelihood pathology CANNOT arise')
    print('  in the code as it stands: there is no optimizer over gamma for it')
    print('  to break. The pathology is real, and it is a property of the')
    print('  three-parameter fit the Methods text CLAIMS, not of the two-')
    print('  parameter fit the code performs. So the offset cannot have been')
    print('  patching it. Question 2 tests what it was patching instead.')
    print()
    print('  Second finding, same function. The "MLE" branch of')
    print('  weighted_lognorm_fit hands scipy.optimize a problem whose')
    print('  solution is closed form: the weighted MLE of a lognormal IS')
    print('  the weighted mean and standard')
    print('  deviation of log x, which is exactly what its own "MoM" branch')
    print('  computes. Measured agreement on this dataset:')
    s_mom, _, sc_mom = FT.weighted_lognorm_fit(x + FT.LOGFIT_OFFSET, weights=w,
                                               method='MoM')
    print(f'    MLE branch  s={shape:.12f}  scale={scale:.12f}')
    print(f'    MoM branch  s={s_mom:.12f}  scale={sc_mom:.12f}')
    print(f'    difference  s={abs(shape - s_mom):.3e}  '
          f'scale={abs(scale - sc_mom):.3e}')
    print('  That is why the regression fixture needed rtol 4.3e-08 on the')
    print('  lognormal column while every other column agreed to 1.3e-14: the')
    print('  convergence path of the optimizer moves between scipy versions.')

    print()
    print('=' * 74)
    print('2. WHICH PROBLEM THE OFFSET WAS SOLVING')
    print('=' * 74)
    rows = []
    for name, (x, w) in ds.items():
        near = x < NEAR_ZERO_FRAC
        keep = ~near
        row = dict(dataset=name, n=len(x), n_near_zero=int(near.sum()),
                   min_over_mean=float(x.min()), has_near_zero=bool(near.any()))
        m2, _ = FT.fit_family('lognormal_2p', x, w, 'mle')
        mo, _ = FT.fit_family('lognormal_offset', x, w, 'mle')
        row['w1_2p'] = w1(m2, x, w)
        row['w1_offset'] = w1(mo, x, w)
        # The same two fits with the near-zero values removed. The fits change;
        # the SCORE is still taken against the full dataset, because the
        # question is which model describes the data we actually have.
        if near.any() and keep.sum() >= 3:
            wk = w[keep] / w[keep].sum()
            m2k, _ = FT.fit_family('lognormal_2p', x[keep], wk, 'mle')
            mok, _ = FT.fit_family('lognormal_offset', x[keep], wk, 'mle')
            row['w1_2p_drop_near_zero'] = w1(m2k, x, w)
            row['w1_offset_drop_near_zero'] = w1(mok, x, w)
        else:
            row['w1_2p_drop_near_zero'] = row['w1_2p']
            row['w1_offset_drop_near_zero'] = row['w1_offset']
        rows.append(row)
    d = pd.DataFrame(rows)
    d['offset_gain'] = d.w1_2p - d.w1_offset
    d['offset_gain_pct'] = d.offset_gain / d.w1_2p * 100
    d['offset_gain_dropped'] = (d.w1_2p_drop_near_zero
                                - d.w1_offset_drop_near_zero)
    d.to_csv(os.path.join(TABLES, 'TABLE_OffsetDiagnosis.csv'), index=False)

    nz = d[d.has_near_zero]
    cl = d[~d.has_near_zero]
    print(f'datasets with a value below {NEAR_ZERO_FRAC:g} of the mean: '
          f'{len(nz)} of {len(d)} ({len(nz)/len(d)*100:.1f} percent), '
          f'{int(d.n_near_zero.sum())} values')
    print(f'the original code measured 28.3 percent of 138. The symmetric log-space '
          f'cleaning and\\nthe plausibility ceiling account for the difference; '
          f'they have not gone away.')
    print()
    print('W1 of the offset fit against the no-offset two-parameter fit:')
    for label, g in [('WITH near-zero values', nz),
                     ('WITHOUT near-zero values', cl),
                     ('all datasets', d)]:
        if not len(g):
            continue
        print(f'  {label:<26} n={len(g):>4}   offset better in '
              f'{int((g.offset_gain > 0).sum()):>4} '
              f'({(g.offset_gain > 0).mean()*100:5.1f} pct)   '
              f'median gain {np.median(g.offset_gain_pct):+7.2f} pct   '
              f'mean W1 {g.w1_2p.mean():.5f} -> {g.w1_offset.mean():.5f}')
    print()
    print('The same comparison on datasets that HAVE near-zero values, after')
    print('removing those values from the FIT (the score is still against the')
    print('full data, because that is the data we have):')
    if len(nz):
        print(f'  offset better in {int((nz.offset_gain_dropped > 0).sum())} '
              f'of {len(nz)} ({(nz.offset_gain_dropped > 0).mean()*100:.1f} pct), '
              f'mean W1 {nz.w1_2p_drop_near_zero.mean():.5f} -> '
              f'{nz.w1_offset_drop_near_zero.mean():.5f}')
        print(f'  mean W1 of the no-offset fit: {nz.w1_2p.mean():.5f} with the '
              f'near-zero values in the fit, {nz.w1_2p_drop_near_zero.mean():.5f} '
              f'with them out')
    print()
    print('the ten datasets where the offset helps most, and their minima:')
    print(d.nlargest(10, 'offset_gain')[
        ['dataset', 'n', 'n_near_zero', 'min_over_mean', 'w1_2p', 'w1_offset',
         'offset_gain_pct']].to_string(index=False,
                                       float_format=lambda v: f'{v:.5g}'))

    print()
    print('=' * 74)
    print('3. THE PATHOLOGY, SHOWN RATHER THAN ASSERTED')
    print('=' * 74)
    print('A naive joint optimizer over (threshold, mu, sigma) with no guard,')
    print('against the profile-likelihood treatment with one. `gap` is')
    print('(min(x) - threshold) in units of the standard deviation: a gap')
    print('driven to zero is the threshold running up against min(x), which is')
    print('the likelihood diverging.')
    print()
    prows = []
    for name, (x, w) in ds.items():
        try:
            nai = fit_lognorm3_naive(x, w)
            pro = F.fit_lognorm3_profile(x, w)
        except ValueError:
            continue
        mn = F.make_lognorm(nai)
        mp = F.make_lognorm(pro)
        prows.append(dict(dataset=name, n=len(x),
                          naive_gap_over_sd=nai['gap_over_sd'],
                          naive_sigma=nai['s'], naive_w1=w1(mn, x, w),
                          profile_status=pro['status'],
                          profile_gap_over_sd=pro['threshold_delta_over_sd'],
                          profile_sigma=pro['s'], profile_w1=w1(mp, x, w)))
    p = pd.DataFrame(prows)
    p.to_csv(os.path.join(TABLES, 'TABLE_ThresholdPathology.csv'),
             index=False)
    collapsed = p.naive_gap_over_sd < 1e-3
    print(f'naive fit drove the threshold to within 1e-3 sd of min(x) in '
          f'{int(collapsed.sum())} of {len(p)} datasets '
          f'({collapsed.mean()*100:.1f} pct)')
    print(f'  median naive gap {p.naive_gap_over_sd.median():.3e} sd, '
          f'median profile gap {p.profile_gap_over_sd.median():.3f} sd')
    print(f'  median naive sigma {p.naive_sigma.median():.4f}, '
          f'median profile sigma {p.profile_sigma.median():.4f}')
    print(f'  mean W1: naive {p.naive_w1.mean():.5f}, '
          f'profile {p.profile_w1.mean():.5f}')
    print()
    print('profile-likelihood outcome, over the 149 datasets:')
    print(p.profile_status.value_counts().to_string())
    print()
    print('the ten worst naive collapses:')
    print(p.nsmallest(10, 'naive_gap_over_sd')[
        ['dataset', 'n', 'naive_gap_over_sd', 'naive_sigma', 'naive_w1',
         'profile_gap_over_sd', 'profile_sigma', 'profile_w1']
    ].to_string(index=False, float_format=lambda v: f'{v:.4g}'))


if __name__ == '__main__':
    main()
