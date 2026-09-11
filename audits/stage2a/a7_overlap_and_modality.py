"""Stage 2a, Part 1 items 4, 5 and 6.

item 4  Component separation. locs on (5, 20) with scales on (0.2, 1.5) places
        components tens of standard deviations apart. Quantify the pairwise
        overlap of the shipped corpus and compare it against the 138 empirical
        datasets.
item 5  Mode counting. Compare estimate_maxima against Silverman's critical
        bandwidth and report how often they disagree.
item 6  Mode weight concentration. cpv = ones(k) * 10 gives near-equal mode
        shares. Report the realized distribution of mode shares.

The empirical datasets have no known components, so both arms are put on the
same footing: a BIC-selected Gaussian mixture is fitted to each dataset and the
Maitra-Melnykov overlap computed from the fit. For the shipped corpus the TRUE
overlap is also available by re-running the legacy generator with its
structural parameters recorded, which says how much the fitted proxy understates.

Writes TABLE_2a_OverlapComparison.csv, TABLE_2a_ModalityComparison.csv,
       TABLE_2a_ModeShares.csv
"""
import numpy as np, pandas as pd, sys, os, warnings
warnings.filterwarnings('ignore')
from _common import load_shipped, load_empirical, write, ROOT

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import components as C
import mixture as M
import modality as MD
from customstats import estimate_maxima
import legacy_generator_pre_stage1 as legacy
from scipy import stats

SEED = 20260911
NSUB = 1500          # subsample of the shipped corpus, for the slow measures


def fitted_overlap(x, rng, kmax=5):
    """Average pairwise Maitra-Melnykov overlap of a BIC-selected mixture."""
    x = np.asarray(x, float)
    if len(x) < 6 or np.std(x) <= 0:
        return np.nan, 1
    k, pi, mu, sd = MD.fit_mixture_bic(x, rng, kmax=kmax)
    if k < 2:
        return 0.0, 1
    comps = [stats.norm(loc=m, scale=max(s, 1e-9)) for m, s in zip(mu, sd)]
    return M.average_overlap(comps, pi, grid_n=1501), k


def legacy_true_overlap(ndatasets=600, seed=SEED):
    """Re-run the legacy generator, recording the structural parameters, and
    compute the exact overlap of the mixture it actually built."""
    np.random.seed(seed)
    rng_n = np.random.default_rng(seed)
    rows = []
    for i in range(ndatasets):
        r = np.random.default_rng(seed + i)
        k = int(r.integers(1, 6))
        locs = r.uniform(5, 20, size=k)
        scales = r.uniform(0.2, 1.5, size=k)
        pi = r.dirichlet(np.ones(k) * 10)
        # component types drawn next; shapes are the legacy constants
        types = r.choice(['gauss', 'skewnorm', 'studentt', 'lognorm'], size=k,
                         p=[0.40, 0.25, 0.25, 0.10])
        comps = []
        for t, lo, sc in zip(types, locs, scales):
            if t == 'gauss':
                comps.append(stats.norm(loc=lo, scale=sc))
            elif t == 'skewnorm':
                comps.append(stats.skewnorm(3.458732, loc=lo, scale=sc))
            elif t == 'studentt':
                comps.append(stats.t(3.366328, loc=lo, scale=sc))
            else:
                comps.append(stats.lognorm(1.136962, loc=lo, scale=sc))
        ov = M.average_overlap(comps, pi, grid_n=1501) if k > 1 else 0.0
        sep = np.nan
        if k > 1:
            d = np.abs(locs[:, None] - locs[None, :])
            pooled = np.sqrt((scales[:, None] ** 2 + scales[None, :] ** 2) / 2)
            iu = np.triu_indices(k, 1)
            sep = float(np.mean(d[iu] / pooled[iu]))
        rows.append(dict(dataset=i, k=k, overlap_true=ov, mean_sep_in_sd=sep,
                         max_mode_share=float(pi.max()), min_mode_share=float(pi.min())))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    rng = np.random.default_rng(SEED)

    print('--- item 4: overlap ---')
    emp = load_empirical()
    rows = []
    for mat in emp:
        x = emp[mat]['data'] / np.mean(emp[mat]['data'])
        ov, k = fitted_overlap(x, rng)
        rows.append(dict(arm='empirical', dataset=mat, n=len(x), k_fitted=k,
                         overlap_fitted=ov))
    DATA, keep, _, _ = load_shipped()
    sub = list(np.random.default_rng(0).choice(keep, NSUB, replace=False))
    for ds in sub:
        x = np.asarray(DATA[ds]['data'], float)
        ov, k = fitted_overlap(x, rng)
        rows.append(dict(arm='shipped_synthetic', dataset=ds, n=len(x), k_fitted=k,
                         overlap_fitted=ov))
    ov_df = pd.DataFrame(rows)

    lt = legacy_true_overlap()
    for _, r in lt.iterrows():
        ov_df.loc[len(ov_df)] = dict(arm='legacy_true', dataset=f'gen{int(r.dataset)}',
                                     n=np.nan, k_fitted=int(r.k),
                                     overlap_fitted=r.overlap_true)
    write(ov_df, 'TABLE_2a_OverlapComparison.csv')
    pd.set_option('display.width', 220)
    print(ov_df.groupby('arm').overlap_fitted.describe(
        percentiles=[.05, .25, .5, .75, .95]).to_string(float_format=lambda v: f'{v:,.4f}'))
    print('\nmultimodal fits only (k >= 2):')
    print(ov_df[ov_df.k_fitted >= 2].groupby('arm').overlap_fitted.describe(
        percentiles=[.05, .5, .95]).to_string(float_format=lambda v: f'{v:,.4f}'))
    print('\nfraction of datasets whose BIC fit is multimodal:')
    print(ov_df.assign(multi=ov_df.k_fitted >= 2).groupby('arm').multi.mean()
          .to_string(float_format=lambda v: f'{v:,.3f}'))
    print('\nlegacy component separation, mean pairwise distance in pooled sd:')
    print(lt.mean_sep_in_sd.describe(percentiles=[.05, .5, .95]).to_string())

    print('\n--- item 6: mode share concentration (Dirichlet alpha = 10) ---')
    write(lt, 'TABLE_2a_ModeShares.csv')
    print(lt[lt.k > 1][['k', 'max_mode_share', 'min_mode_share']]
          .groupby('k').describe(percentiles=[.05, .5, .95])
          .to_string(float_format=lambda v: f'{v:,.3f}'))

    print('\n--- item 5: mode counting, estimate_maxima vs Silverman ---')
    rows = []
    for mat in emp:
        x = emp[mat]['data'] / np.mean(emp[mat]['data'])
        w = np.ones_like(x) / len(x)
        rows.append(dict(arm='empirical', dataset=mat, n=len(x),
                         estimate_maxima=float(estimate_maxima(x, w)),
                         crit_bw_1=MD.critical_bandwidth(x, 1),
                         n_modes_silverman=MD.n_modes_silverman(x, rng=rng, nboot=100)))
    for ds in sub[:800]:
        x = np.asarray(DATA[ds]['data'], float)
        w = np.ones_like(x) / len(x)
        rows.append(dict(arm='shipped_synthetic', dataset=ds, n=len(x),
                         estimate_maxima=float(estimate_maxima(x, w)),
                         crit_bw_1=MD.critical_bandwidth(x, 1),
                         n_modes_silverman=MD.n_modes_silverman(x, rng=rng, nboot=100)))
    md = pd.DataFrame(rows)
    write(md, 'TABLE_2a_ModalityComparison.csv')
    md['em_rounded'] = np.round(md.estimate_maxima).astype('Int64')
    for arm, g in md.groupby('arm'):
        agree = float((g.em_rounded == g.n_modes_silverman).mean())
        print(f'{arm:<20} estimate_maxima rounded equals the Silverman count in '
              f'{agree*100:5.1f}% of datasets  (n={len(g)})')
        print(f'{"":<20} estimate_maxima range {g.estimate_maxima.min():.3f} to '
              f'{g.estimate_maxima.max():.3f};  Silverman counts '
              f'{dict(g.n_modes_silverman.value_counts().sort_index())}')
        print(f'{"":<20} crit_bw_1 median {g.crit_bw_1.median():.3f}, '
              f'5-95 pct {g.crit_bw_1.quantile(.05):.3f} to {g.crit_bw_1.quantile(.95):.3f}')
    print(f"\nSpearman correlation between estimate_maxima and crit_bw_1: "
          f"{md[['estimate_maxima','crit_bw_1']].corr(method='spearman').iloc[0,1]:.3f}")
