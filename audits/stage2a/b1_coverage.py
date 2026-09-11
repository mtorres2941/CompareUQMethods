"""Stage 2a, Part 6: coverage of the empirical metric space, and Table 1.

Six deliverables:
  1. the joint distribution of the empirical metrics: correlation matrix and
     how many effectively independent dimensions there are
  2. the same for the synthetic corpus, and a comparison
  3. mathematically infeasible skewness/kurtosis combinations, and what the
     generator did about them
  4. Table 1: every generation parameter, the empirical range it derives from,
     and the synthetic range achieved
  5. a coverage figure and a per-metric coverage statistic
  6. size coverage against the new strata, and the probe set's verdict

Run after the corpus has been generated and activated.
"""
import json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from _common import write, ROOT, TABLES

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus as CORP
import empirical as E
import genconfig as G
import generator as GEN
from customstats import empirical_metadata

SEED = 20260911
METRICS = ['n', 'coeffvar', 'entropy', 'skewness', 'kurtosis', 'modality_index',
           'crit_bw_1', 'weight_outliers', 'fit_norm_SW', 'fit_lognorm_SW',
           'w_v_uw_wasserstein']
CORE = [m for m in METRICS if m != 'n']


# --------------------------------------------------------------------------
def empirical_metrics(rng):
    ds, rep = E.prepare(rng)
    rows = {}
    for mat, (x, w) in ds.items():
        rows[mat] = empirical_metadata(x, w)
    df = pd.DataFrame(rows).T.astype(float)
    df.index.name = 'material'
    return df.reset_index(), pd.DataFrame(rep)


def effective_dimension(M):
    """Participation ratio of the correlation eigenvalues: p for p independent
    metrics, 1 when every metric is a copy of one quantity."""
    M = M[np.all(np.isfinite(M), axis=1)]
    if len(M) < 3:
        return np.nan
    sd = M.std(0)
    M = M[:, sd > 0]
    Z = (M - M.mean(0)) / M.std(0)
    lam = np.clip(np.linalg.eigvalsh(np.corrcoef(Z, rowvar=False)), 0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def pca_report(M, names):
    M = M[np.all(np.isfinite(M), axis=1)]
    Z = (M - M.mean(0)) / M.std(0)
    C = np.corrcoef(Z, rowvar=False)
    lam, vec = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    lam, vec = lam[order], vec[:, order]
    var = lam / lam.sum()
    return pd.DataFrame({
        'component': [f'PC{i+1}' for i in range(len(lam))],
        'eigenvalue': lam, 'variance_share': var,
        'cumulative_share': np.cumsum(var),
        'top_loading': [names[int(np.argmax(np.abs(vec[:, i])))] for i in range(len(lam))],
    })


def coverage_stats(emp, syn, names):
    """Per metric: what share of empirical datasets fall inside the synthetic
    range and the synthetic 1-99 percentile band, and the reverse."""
    rows = []
    for m in names:
        e = emp[m].replace([np.inf, -np.inf], np.nan).dropna()
        s = syn[m].replace([np.inf, -np.inf], np.nan).dropna()
        if len(e) == 0 or len(s) == 0:
            continue
        smin, smax = s.min(), s.max()
        s01, s99 = s.quantile(.01), s.quantile(.99)
        e01, e99 = e.quantile(.01), e.quantile(.99)
        rows.append(dict(
            metric=m,
            empirical_min=float(e.min()), empirical_max=float(e.max()),
            synthetic_min=float(smin), synthetic_max=float(smax),
            emp_inside_syn_range=float(((e >= smin) & (e <= smax)).mean()),
            emp_inside_syn_p1_p99=float(((e >= s01) & (e <= s99)).mean()),
            syn_inside_emp_range=float(((s >= e.min()) & (s <= e.max())).mean()),
            syn_outside_emp_p1_p99=float(((s < e01) | (s > e99)).mean()),
            margin_below=float((e.min() - smin) / (e.max() - e.min())),
            margin_above=float((smax - e.max()) / (e.max() - e.min())),
        ))
    return pd.DataFrame(rows)


def hull_coverage(emp, syn, names, n_pc=3):
    """Share of synthetic datasets outside the empirical convex hull, and the
    reverse, in the leading principal components of the EMPIRICAL space."""
    from scipy.spatial import ConvexHull, Delaunay
    e = emp[names].replace([np.inf, -np.inf], np.nan).dropna()
    s = syn[names].replace([np.inf, -np.inf], np.nan).dropna()
    mu, sd = e.mean(), e.std()
    Ze = ((e - mu) / sd).to_numpy()
    Zs = ((s - mu) / sd).to_numpy()
    C = np.corrcoef(Ze, rowvar=False)
    lam, vec = np.linalg.eigh(C)
    V = vec[:, np.argsort(lam)[::-1][:n_pc]]
    Pe, Ps = Ze @ V, Zs @ V
    tri = Delaunay(Pe)
    inside_syn = tri.find_simplex(Ps) >= 0
    return dict(n_pc=n_pc, n_empirical=len(Pe), n_synthetic=len(Ps),
                synthetic_inside_empirical_hull=float(inside_syn.mean()),
                synthetic_outside_empirical_hull=float(1 - inside_syn.mean()),
                empirical_hull_volume=float(ConvexHull(Pe).volume),
                synthetic_hull_volume=float(ConvexHull(Ps).volume))


def infeasible_report(syn):
    """Part 6 item 3. Where sample skewness and kurtosis are pinned against the
    bound that n alone imposes."""
    rows = []
    for _, r in syn.iterrows():
        n = int(r['n'])
        max_skew, min_k, max_k = GEN.sample_moment_bounds(n)
        for lab in ('', '_uw'):
            sk, ku = r.get('skewness' + lab), r.get('kurtosis' + lab)
            rows.append(dict(dataset=r['dataset'], stratum=r['stratum'], n=n,
                             weighting='variable' if lab == '' else 'uniform',
                             skewness=sk, kurtosis_excess=ku,
                             max_abs_skew=max_skew,
                             max_excess_kurt=max_k - 3.0,
                             min_excess_kurt=min_k - 3.0,
                             skew_frac_of_bound=abs(sk) / max_skew if max_skew else np.nan,
                             kurt_frac_of_bound=((ku + 3.0) / max_k
                                                 if (max_k and np.isfinite(ku)) else np.nan),
                             kurtosis_defined=bool(np.isfinite(ku))))
    return pd.DataFrame(rows)


def undefined_by_stratum(syn):
    rows = []
    for st, g in syn.groupby('stratum'):
        row = dict(stratum=st, n_datasets=len(g), n_min=int(g.n.min()),
                   n_max=int(g.n.max()), n_median=float(g.n.median()))
        for m in syn.columns:
            if syn[m].dtype.kind not in 'fiu':
                continue
            v = g[m].replace([np.inf, -np.inf], np.nan)
            row[f'undef__{m}'] = float(v.isna().mean())
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    rng = np.random.default_rng(SEED)
    print('empirical metrics at alpha = 1 with multiplicative low-end cleaning ...')
    emp, eprep = empirical_metrics(rng)
    write(emp, 'TABLE_2a_EmpiricalMetrics_final.csv')
    write(eprep, 'TABLE_2a_EmpiricalCleaningReport.csv')

    print('loading the regenerated corpus ...')
    metrics_all = pd.read_parquet(os.path.join(CORP.active_dir(), 'metrics.parquet'))
    syn = metrics_all[~metrics_all.is_probe].reset_index(drop=True)
    probe = metrics_all[metrics_all.is_probe].reset_index(drop=True)
    print(f'  corpus {len(syn)}, probe {len(probe)}')

    pd.set_option('display.width', 250)

    # ---- 1 and 2: joint structure -------------------------------------
    print('\n--- effective dimensionality ---')
    ed = pd.DataFrame([
        dict(arm='empirical', n=len(emp),
             effective_dimension=effective_dimension(emp[CORE].to_numpy(float)),
             n_metrics=len(CORE)),
        dict(arm='synthetic', n=len(syn),
             effective_dimension=effective_dimension(syn[CORE].to_numpy(float)),
             n_metrics=len(CORE)),
    ])
    print(ed.to_string(index=False, float_format=lambda v: f'{v:,.3f}'))
    write(ed, 'TABLE_2a_EffectiveDimension.csv')

    pe = pca_report(emp[CORE].to_numpy(float), CORE).assign(arm='empirical')
    ps = pca_report(syn[CORE].to_numpy(float), CORE).assign(arm='synthetic')
    write(pd.concat([pe, ps]), 'TABLE_2a_PCA.csv')
    print('\nempirical PCA:')
    print(pe.to_string(index=False, float_format=lambda v: f'{v:,.3f}'))
    print('\nsynthetic PCA:')
    print(ps.to_string(index=False, float_format=lambda v: f'{v:,.3f}'))

    ce = emp[CORE].corr().round(3).assign(arm='empirical')
    cs = syn[CORE].corr().round(3).assign(arm='synthetic')
    write(pd.concat([ce.reset_index(names='metric'), cs.reset_index(names='metric')]),
          'TABLE_2a_CorrelationMatrices.csv')

    # ---- coverage ------------------------------------------------------
    print('\n--- per-metric coverage ---')
    cov = coverage_stats(emp, syn, METRICS)
    write(cov, 'TABLE_2a_Coverage.csv')
    print(cov.to_string(index=False, float_format=lambda v: f'{v:,.4f}'))

    print('\n--- convex hull coverage in the leading empirical PCs ---')
    for npc in (2, 3):
        h = hull_coverage(emp, syn, CORE, npc)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items()})

    # ---- infeasible combinations ---------------------------------------
    print('\n--- skewness and kurtosis against the bounds n imposes ---')
    inf_df = infeasible_report(syn)
    write(inf_df, 'TABLE_2a_MomentBounds.csv')
    agg = inf_df.groupby(['stratum', 'weighting']).agg(
        median_skew_frac=('skew_frac_of_bound', 'median'),
        p95_skew_frac=('skew_frac_of_bound', lambda s: s.quantile(.95)),
        pinned_skew_over_0p9=('skew_frac_of_bound', lambda s: float((s > 0.9).mean())),
        median_kurt_frac=('kurt_frac_of_bound', 'median'),
        pinned_kurt_over_0p9=('kurt_frac_of_bound', lambda s: float((s > 0.9).mean())),
        kurtosis_defined=('kurtosis_defined', 'mean'))
    print(agg.to_string(float_format=lambda v: f'{v:,.4f}'))

    # ---- undefined metrics by stratum ----------------------------------
    print('\n--- undefined metrics per stratum (flag for Stage 2f) ---')
    ub = undefined_by_stratum(syn)
    write(ub, 'TABLE_2a_UndefinedByStratum.csv')
    keep = ['stratum', 'n_datasets', 'n_min', 'n_max', 'n_median'] + \
           [c for c in ub.columns if c.startswith('undef__') and ub[c].max() > 0]
    print(ub[keep].to_string(index=False, float_format=lambda v: f'{v:,.4f}'))

    # ---- size coverage and the probe verdict ---------------------------
    print('\n--- size coverage ---')
    rows = []
    for s in G.DEFAULT.strata:
        g = syn[syn.stratum == s.name]
        e = emp[(emp.n >= s.n_lo) & (emp.n <= s.n_hi)]
        rows.append(dict(stratum=s.name, target_lo=s.n_lo, target_hi=s.n_hi,
                         target_count=s.n_datasets, realized_count=len(g),
                         realized_min=int(g.n.min()), realized_max=int(g.n.max()),
                         empirical_count=len(e),
                         empirical_share=len(e) / len(emp),
                         post_strat_weight=G.EMPIRICAL_STRATUM_SHARE[s.name]))
    sc = pd.DataFrame(rows)
    write(sc, 'TABLE_2a_SizeCoverage.csv')
    print(sc.to_string(index=False, float_format=lambda v: f'{v:,.4f}'))
    print(f"\nempirical datasets above 9,999: "
          f"{int((emp.n > 9999).sum())} of {len(emp)}  "
          f"(largest {int(emp.n.max()):,})")

    # ---- post-stratified vs per-stratum aggregates ----------------------
    print('\n--- every headline metric, per stratum and post-stratified ---')
    rows = []
    for m in CORE:
        per = syn.groupby('stratum')[m].median()
        w = np.array([G.EMPIRICAL_STRATUM_SHARE[s] for s in per.index])
        rows.append(dict(metric=m, **{f'median__{s}': per[s] for s in per.index},
                         median_equal_allocation=float(syn[m].median()),
                         median_post_stratified=float(np.sum(per.values * w) / w.sum()),
                         median_empirical=float(emp[m].median())))
    ag = pd.DataFrame(rows)
    write(ag, 'TABLE_2a_PostStratified.csv')
    print(ag.to_string(index=False, float_format=lambda v: f'{v:,.4f}'))

    # ---- probe verdict --------------------------------------------------
    if len(probe):
        print('\n--- probe set: have results plateaued above n = 10^4? ---')
        s4 = syn[syn.stratum == 's4_1000_9999']
        rows = []
        for m in CORE:
            a = s4[m].replace([np.inf, -np.inf], np.nan).dropna()
            b = probe[m].replace([np.inf, -np.inf], np.nan).dropna()
            if len(b) < 5:
                continue
            pooled = np.sqrt((a.var() + b.var()) / 2)
            se = np.sqrt(a.var() / len(a) + b.var() / len(b))
            rows.append(dict(metric=m, s4_median=float(a.median()),
                             probe_median=float(b.median()),
                             s4_mean=float(a.mean()), probe_mean=float(b.mean()),
                             diff_in_sd=float((b.mean() - a.mean()) / pooled)
                             if pooled else np.nan,
                             diff_over_se=float((b.mean() - a.mean()) / se)
                             if se else np.nan))
        pv = pd.DataFrame(rows)
        write(pv, 'TABLE_2a_ProbeVerdict.csv')
        print(pv.to_string(index=False, float_format=lambda v: f'{v:,.4f}'))
