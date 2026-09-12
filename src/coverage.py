"""Coverage of the empirical metric space by the synthetic corpus.

This is analysis that belongs to the paper, not to the Stage 2a audit: it
produces Table 1 and the coverage figure. It lives in `src/` so it can be
tested, and notebook 1 calls it, so the notebook remains the place the analysis
is read from. The one-off measurements that characterize the OLD generator stay
in `audits/stage2a/`, because they describe code that no longer exists.

The question every function here answers is the one the author set: do the
synthetic datasets look like the 138 empirical ECC datasets, as measured by the
statistical metrics, with margin on both sides so a generalizability claim is
supportable? Margin means the synthetic range extends beyond the empirical
range at both ends, including into regions the empirical set does not occupy at
all, such as left skew.
"""

import numpy as np
import pandas as pd

import genconfig as G
import generator as GEN

CORE_METRICS = ('coeffvar', 'skewness', 'kurtosis', 'entropy', 'crit_bw_1',
                'weight_outliers', 'fit_norm_SW', 'fit_lognorm_SW',
                'w_v_uw_wasserstein')
ALL_METRICS = ('n',) + CORE_METRICS


def _clean(series):
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def coverage_table(empirical, synthetic, metrics=ALL_METRICS):
    """Per metric: is the empirical range inside the synthetic range, and by
    how much margin on each side?

    `margin_below` and `margin_above` are expressed in units of the empirical
    range, so 0.5 means the synthetic data extend half an empirical range
    beyond the empirical extreme. Negative means the synthetic corpus does not
    reach that far, which is a coverage gap and is reported as one.
    """
    rows = []
    for m in metrics:
        if m not in empirical or m not in synthetic:
            continue
        e, s = _clean(empirical[m]), _clean(synthetic[m])
        if len(e) < 3 or len(s) < 3:
            continue
        espan = e.max() - e.min()
        rows.append(dict(
            metric=m,
            empirical_min=float(e.min()), empirical_median=float(e.median()),
            empirical_max=float(e.max()),
            synthetic_min=float(s.min()), synthetic_median=float(s.median()),
            synthetic_max=float(s.max()),
            empirical_covered=float(((e >= s.min()) & (e <= s.max())).mean()),
            margin_below=float((e.min() - s.min()) / espan) if espan else np.nan,
            margin_above=float((s.max() - e.max()) / espan) if espan else np.nan,
            synthetic_beyond_empirical=float(
                ((s < e.min()) | (s > e.max())).mean()),
        ))
    return pd.DataFrame(rows)


def effective_dimension(frame, metrics=CORE_METRICS):
    """Participation ratio of the correlation eigenvalues: p for p independent
    metrics, 1 when every metric is a copy of one underlying quantity."""
    M = frame[list(metrics)].apply(pd.to_numeric, errors='coerce')
    M = M.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if len(M) < 5:
        return np.nan
    sd = M.std(0)
    M = M[:, sd > 0]
    Z = (M - M.mean(0)) / M.std(0)
    lam = np.clip(np.linalg.eigvalsh(np.corrcoef(Z, rowvar=False)), 0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def pca_table(frame, metrics=CORE_METRICS):
    M = frame[list(metrics)].apply(pd.to_numeric, errors='coerce')
    M = M.replace([np.inf, -np.inf], np.nan).dropna()
    names = list(M.columns)
    Z = ((M - M.mean()) / M.std()).to_numpy(float)
    lam, vec = np.linalg.eigh(np.corrcoef(Z, rowvar=False))
    order = np.argsort(lam)[::-1]
    lam, vec = lam[order], vec[:, order]
    share = lam / lam.sum()
    return pd.DataFrame({
        'component': [f'PC{i+1}' for i in range(len(lam))],
        'eigenvalue': lam, 'variance_share': share,
        'cumulative_share': np.cumsum(share),
        'dominant_metric': [names[int(np.argmax(np.abs(vec[:, i])))]
                            for i in range(len(lam))],
    })


def moment_bound_table(synthetic):
    """Where sample skewness and kurtosis are pinned against the bound that n
    alone imposes.

    This is what explains an artifact at high negative excess kurtosis: at
    n = 3 the excess kurtosis cannot exceed -1.5 whatever the parent is, so the
    whole of stratum 1 sits against a ceiling that has nothing to do with the
    generator.
    """
    rows = []
    for _, r in synthetic.iterrows():
        n = int(r['n'])
        max_skew, min_k, max_k = GEN.sample_moment_bounds(n)
        for label, tag in (('', 'variable'), ('_uw', 'uniform')):
            sk, ku = r.get('skewness' + label), r.get('kurtosis' + label)
            rows.append(dict(
                dataset=r.get('dataset'), stratum=r.get('stratum'), n=n,
                weighting=tag, skewness=sk, kurtosis_excess=ku,
                max_abs_skew=max_skew, max_excess_kurtosis=max_k - 3.0,
                min_excess_kurtosis=min_k - 3.0,
                skew_frac_of_bound=(abs(sk) / max_skew if max_skew else np.nan),
                kurt_frac_of_bound=((ku + 3.0) / max_k
                                    if (max_k and np.isfinite(ku)) else np.nan),
                kurtosis_defined=bool(np.isfinite(ku)),
            ))
    return pd.DataFrame(rows)


def undefined_by_stratum(synthetic, metrics=ALL_METRICS):
    """How many metrics are undefined in each stratum.

    Expected, not a defect: unbiased excess kurtosis divides by
    (n-1)(n-2)(n-3), so it is undefined for n < 4 and stratum 1 starts at
    n = 3. Stage 2f's complete-case models must be told, or they will drop the
    stratum silently.
    """
    rows = []
    for stratum, g in synthetic.groupby('stratum'):
        row = dict(stratum=stratum, n_datasets=len(g), n_min=int(g.n.min()),
                   n_max=int(g.n.max()), n_median=float(g.n.median()))
        for m in metrics:
            if m in g:
                row[f'undefined__{m}'] = float(
                    g[m].replace([np.inf, -np.inf], np.nan).isna().mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values('stratum').reset_index(drop=True)


def post_stratified(synthetic, empirical, metrics=CORE_METRICS,
                    weights=None):
    """Every headline metric twice: per stratum, and reweighted to the
    empirical frequency of each stratum.

    Equal allocation across strata buys equal precision in every size regime,
    which is what the metric-versus-W1 modelling in Stage 2f needs. It does not
    match the empirical size distribution, so the reweighted column is the one
    that describes real ECC datasets. Reporting both is the direct answer to
    the objection that the corpus over-represents large datasets.
    """
    weights = weights or G.EMPIRICAL_STRATUM_SHARE
    rows = []
    for m in metrics:
        if m not in synthetic:
            continue
        per = synthetic.groupby('stratum')[m].median()
        w = np.array([weights.get(s, 0.0) for s in per.index], float)
        row = dict(metric=m)
        row.update({f'median__{s}': float(per[s]) for s in per.index})
        row['median_equal_allocation'] = float(_clean(synthetic[m]).median())
        row['median_post_stratified'] = (float(np.sum(per.values * w) / w.sum())
                                         if w.sum() else np.nan)
        if m in empirical:
            row['median_empirical'] = float(_clean(empirical[m]).median())
        rows.append(row)
    return pd.DataFrame(rows)


def probe_verdict(synthetic, probe, metrics=CORE_METRICS):
    """Has anything stopped changing above n = 10 ** 4?

    Compares the probe set against the largest corpus stratum. `diff_over_se`
    is the difference in means over its standard error, so values inside about
    2 mean the probe set is indistinguishable from stratum 4 and the coverage
    claim can be stated as complete to 9,999 with stability established above.
    """
    s4 = synthetic[synthetic.stratum == 's4_1000_9999']
    rows = []
    for m in metrics:
        if m not in probe or m not in s4:
            continue
        a, b = _clean(s4[m]), _clean(probe[m])
        if len(b) < 5 or len(a) < 5:
            continue
        pooled = np.sqrt((a.var() + b.var()) / 2)
        se = np.sqrt(a.var() / len(a) + b.var() / len(b))
        rows.append(dict(metric=m, stratum4_median=float(a.median()),
                         probe_median=float(b.median()),
                         stratum4_mean=float(a.mean()), probe_mean=float(b.mean()),
                         diff_in_pooled_sd=float((b.mean() - a.mean()) / pooled)
                         if pooled else np.nan,
                         diff_over_se=float((b.mean() - a.mean()) / se)
                         if se else np.nan,
                         plateaued=bool(abs((b.mean() - a.mean()) / se) < 2)
                         if se else False))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def generation_parameter_table(cfg, synthetic, probe, parents, empirical):
    """Table 1: every generation parameter, the empirical quantity its range
    derives from, and the range the corpus actually achieved.

    The achieved column is read from the per-dataset provenance records, never
    assumed from the configuration, so a parameter that was requested and not
    met shows up as a discrepancy rather than being reported as if it had been.
    """
    comp_skew, comp_exk, comp_sd, fams = [], [], [], []
    ks, ov_a, cv_t, cv_a, tmass, dropped, retries = [], [], [], [], [], [], []
    pi_max, mkt_max, cv_status = [], [], []
    for r in parents.values():
        for c in r['components']:
            comp_skew.append(c['skew'])
            comp_exk.append(c['exkurt'])
            comp_sd.append(c['sd'])
            fams.append(c['family'])
        ks.append(r['k'])
        ov_a.append(r['overlap_achieved'])
        cv_t.append(r['cv_target'])
        cv_a.append(r['cv_achieved'])
        cv_status.append(r['cv_status'])
        tmass.append(r['truncated_mass'])
        dropped.append(r.get('n_components_dropped', 0))
        retries.append(r.get('component_retries', 0))
        pi_max.append(max(r['pi']))
        mkt_max.append(max(r['market']))

    def rng_of(a):
        a = np.asarray([v for v in a if np.isfinite(v)], float)
        return 'n/a' if not len(a) else f'{a.min():.3g} to {a.max():.3g}'

    def pct_of(a, lo=5, hi=95):
        a = np.asarray([v for v in a if np.isfinite(v)], float)
        return ('n/a' if not len(a)
                else f'{np.percentile(a, lo):.3g} to {np.percentile(a, hi):.3g}')

    emp = empirical
    # Column access is by bracket throughout: DataFrame.kurtosis is a pandas
    # METHOD, so emp.kurtosis silently returns the method rather than the
    # column and fails later with an unrelated-looking AttributeError.
    nstat = pd.to_numeric(emp['n'])
    ok = float(np.mean([s == 'ok' for s in cv_status]))
    rows = [
        ('Number of mixture components, k',
         f'integer uniform on [{cfg.k_min}, {cfg.k_max}]',
         'a BIC-selected Gaussian mixture is multimodal in 79.0 pct of the 138 '
         'empirical datasets',
         f'{min(ks)} to {max(ks)}, mean {np.mean(ks):.2f}'),
        ('Average pairwise component overlap',
         f'log-uniform on [{10 ** cfg.overlap_log10_lo:.0e}, '
         f'{10 ** cfg.overlap_log10_hi:.2f}], solved for by moving the '
         'component locations',
         'empirical fitted overlap: median 0.0218, 95th pct 0.4528, max 0.6719; '
         'the shipped synthetic corpus had a median of 0.0037',
         f'median {np.median(ov_a):.4f}, 95th pct {np.percentile(ov_a, 95):.4f}, '
         f'max {np.max(ov_a):.4f}'),
        ('Coefficient of variation of the parent',
         f'normal in log10 with mean {cfg.cv_log10_mean:.4f} and sd '
         f'{cfg.cv_log10_sd:.4f}, truncated to '
         f'[{10 ** cfg.cv_log10_lo:.3g}, {10 ** cfg.cv_log10_hi:.3g}], solved '
         'for by placing the mixture relative to zero',
         f'centred on the empirical distribution, whose log10 coefficient of '
         f'variation has mean -0.2641 and sd 0.2913; the sd is DOUBLED for '
         f'margin. Empirical range {_clean(emp["coeffvar"]).min():.4g} to '
         f'{_clean(emp["coeffvar"]).max():.4g}',
         f'target met exactly in {ok * 100:.1f} pct; achieved '
         f'{rng_of(cv_a)}'),
        ('Component skewness target',
         f'uniform on [{cfg.comp_skew_lo}, {cfg.comp_skew_hi}]',
         f'empirical dataset skewness {_clean(emp["skewness"]).min():.3g} to '
         f'{_clean(emp["skewness"]).max():.3g}, median '
         f'{_clean(emp["skewness"]).median():.3g}',
         rng_of(comp_skew)),
        ('Component excess kurtosis target',
         f'uniform on [{cfg.comp_exkurt_lo}, {cfg.comp_exkurt_hi}], lifted to '
         'the feasible boundary skewness ** 2 - 2',
         f'empirical dataset excess kurtosis {_clean(emp["kurtosis"]).min():.3g} '
         f'to {_clean(emp["kurtosis"]).max():.3g}',
         rng_of(comp_exk)),
        ('Component location spread (position_skew)',
         f'locations at z ** {cfg.position_skew} for z uniform on (0, 1), '
         'before the overlap solve scales them',
         'clusters components toward the low end with the occasional far-out '
         'one. It, not the component shapes, sets the achievable coefficient '
         'of variation: uniform spacing pins it near 0.57',
         f'{cfg.position_skew}'),
        ('Component standard deviation, relative',
         f'log-uniform on [{10 ** cfg.comp_sd_log10_lo:.3g}, '
         f'{10 ** cfg.comp_sd_log10_hi:.3g}]',
         'scale is unidentified: every dataset is divided by its own unweighted '
         'mean',
         rng_of(comp_sd)),
        ('Component family',
         'Johnson SU, lognormal, beta-prime or beta, chosen by which Pearson '
         'region the moment target falls in',
         'not a free choice: the family follows from the skewness and kurtosis '
         'target',
         ', '.join(f'{k} {v / len(fams) * 100:.1f} pct'
                   for k, v in pd.Series(fams).value_counts().items())),
        ('Mode sampling-share concentration, Dirichlet alpha',
         f'{cfg.mode_share_alpha}',
         'not measurable: the empirical mode structure is latent. Swept in '
         'Stage 2h',
         f'largest mode share, 5th to 95th pct {pct_of(pi_max)}'),
        ('Mode market-share concentration, Dirichlet alpha',
         f'{cfg.market_share_alpha}, flat',
         'market shares are unavailable; the flat Dirichlet is the '
         'maximum-entropy prior. Marsh, Hattam and Allen (2025) report a real '
         'top share of 63.75 pct',
         f'largest market share, 5th to 95th pct {pct_of(mkt_max)}'),
        ('Point weight concentration within a mode, Dirichlet alpha',
         f'{cfg.point_weight_alpha}, flat',
         'the same alpha in both arms; the empirical arm previously used 5',
         f'{cfg.point_weight_alpha}'),
        ('Mode-to-point weight coupling',
         f'{cfg.mode_coupling}',
         '0 reproduces the old uncoupled behaviour, in which the '
         'market-weighted distribution had no population object. Swept in '
         'Stage 2h',
         f'{cfg.mode_coupling}'),
        ('Truncation bounds',
         f'max(Q1 - {cfg.trunc_iqr_mult} * IQR, 0) and '
         f'Q3 + {cfg.trunc_iqr_mult} * IQR of the population mixture',
         'the empirical extraction trims at the same multiple, on the sample',
         f'probability mass removed: median {np.median(tmass):.5f}, '
         f'95th pct {np.percentile(tmass, 95):.4f}, max {np.max(tmass):.4f}'),
        ('Mass allowed below zero when placing the mixture',
         f'at most {cfg.max_low_tail_truncated}',
         'positivity is enforced by the truncation, so this limits how far a '
         'distribution may be pushed toward the origin for spread',
         f'median {np.median(tmass):.5f} of the parent discarded'),
        ('Dataset size n',
         '2,500 datasets log-uniform within each of 3-9, 10-99, 100-999, '
         '1000-9999',
         f'empirical n runs {int(nstat.min())} to {int(nstat.max()):,}, '
         f'median {int(nstat.median())}',
         f'{int(synthetic.n.min())} to {int(synthetic.n.max()):,}, '
         f'median {int(synthetic.n.median())}'),
        ('Probe set, outside the corpus',
         f'{cfg.probe.n_datasets} datasets log-uniform on '
         f'[{cfg.probe.n_lo:,}, {cfg.probe.n_hi:,}]',
         f'{int((nstat > 9999).sum())} empirical datasets exceed 9,999, the '
         f'largest {int(nstat.max()):,}',
         f'{len(probe)} datasets, {int(probe.n.min()):,} to '
         f'{int(probe.n.max()):,}' if len(probe) else 'none'),
        ('Components dropped for falling outside truncation',
         'dropped below 1e-9 of their own mass inside the bounds; both weight '
         'vectors renormalized',
         'not applicable',
         f'{int(np.sum(dropped))} components across '
         f'{int(np.sum(np.array(dropped) > 0)):,} datasets'),
        ('Moment targets redrawn as infeasible or degenerate',
         f'up to {cfg.max_component_retries} redraws per component',
         'not applicable',
         f'{int(np.sum(retries)):,} redraws over {len(fams):,} components, '
         f'{np.sum(retries) / max(len(fams), 1):.2f} per component'),
        ('Random seed',
         f'{cfg.seed}',
         'not applicable',
         'recorded in runmeta.json beside the corpus'),
    ]
    return pd.DataFrame(rows, columns=['parameter', 'configured',
                                       'empirical_basis', 'synthetic_achieved'])


# --------------------------------------------------------------------------
def distribution_comparison(empirical, synthetic, metrics=ALL_METRICS):
    """Compare the DISTRIBUTION of each characteristic between the two arms.

    Range coverage answers "does the synthetic span contain the empirical
    values", which is a weak question: it reads 100 percent while the synthetic
    distribution sits somewhere else entirely inside that span. This answers
    the question the tuning actually turns on, which is whether the two
    distributions have the same shape.

    `w1_standardized` is the Wasserstein-1 distance between the two
    distributions after both are put on the empirical distribution's scale, so
    it is in units of empirical standard deviations and comparable across
    characteristics. `ks` is the Kolmogorov-Smirnov statistic. Roughly: below
    0.1 the two are hard to tell apart, above 0.3 they are visibly different.
    """
    from scipy import stats as _st
    rows = []
    for m in metrics:
        if m not in empirical or m not in synthetic:
            continue
        e, s = _clean(empirical[m]), _clean(synthetic[m])
        if len(e) < 5 or len(s) < 5:
            continue
        sd = e.std()
        scale = sd if sd > 0 else 1.0
        rows.append(dict(
            metric=m,
            empirical_median=float(e.median()), synthetic_median=float(s.median()),
            empirical_iqr=float(e.quantile(.75) - e.quantile(.25)),
            synthetic_iqr=float(s.quantile(.75) - s.quantile(.25)),
            median_shift_in_sd=float((s.median() - e.median()) / scale),
            w1_standardized=float(_st.wasserstein_distance(e / scale, s / scale)),
            ks=float(_st.ks_2samp(e, s).statistic),
        ))
    out = pd.DataFrame(rows)
    return out.sort_values('w1_standardized', ascending=False).reset_index(drop=True)


def modality_comparison(empirical_datasets, synthetic_values, synthetic_ids,
                        rng, nboot=60, kmax=6):
    """Distribution of the NUMBER OF MODES in each arm, by Silverman's test.

    The metric that matters most and the one the corpus gets most wrong. A
    modality index that falls inside the empirical range can still have
    completely the wrong distribution, which is exactly what happened.
    """
    import modality as _md
    emp = np.array([_md.n_modes_silverman(x, rng=rng, nboot=nboot, kmax=kmax)
                    for x, _ in empirical_datasets.values()])
    syn = []
    for ds, g in synthetic_values.groupby('dataset_id', observed=True):
        if str(ds) not in synthetic_ids:
            continue
        x = g['value'].to_numpy()
        if len(x) >= 4:
            syn.append(_md.n_modes_silverman(x, rng=rng, nboot=nboot, kmax=kmax))
    syn = np.array(syn)
    rows = []
    for k in range(1, kmax + 2):
        rows.append(dict(modes=(f'{k}' if k <= kmax else f'{kmax}+'),
                         empirical_share=float((emp == k).mean()),
                         synthetic_share=float((syn == k).mean())))
    out = pd.DataFrame(rows)
    out.loc[len(out)] = dict(modes='multimodal',
                             empirical_share=float((emp > 1).mean()),
                             synthetic_share=float((syn > 1).mean()))
    return out, emp, syn
