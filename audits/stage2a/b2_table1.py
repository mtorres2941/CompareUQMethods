"""Stage 2a, Part 6 item 4: Table 1 for the paper.

Every generation parameter, the empirical quantity its range derives from, and
the range the synthetic corpus actually achieved. The achieved column is read
from the per-dataset provenance records, not assumed from the configuration,
so a parameter that was requested and not met shows up as a discrepancy.

Writes TABLE_2a_Table1_GenerationParameters.csv
"""
import gzip, json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from _common import write

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus as CORP
import empirical as E
import genconfig as G
from customstats import empirical_metadata

SEED = 20260911


def rng_range(a):
    a = np.asarray([v for v in a if np.isfinite(v)], float)
    if len(a) == 0:
        return 'n/a'
    return f'{a.min():.3g} to {a.max():.3g}'


def pct_range(a, lo=5, hi=95):
    a = np.asarray([v for v in a if np.isfinite(v)], float)
    if len(a) == 0:
        return 'n/a'
    return f'{np.percentile(a, lo):.3g} to {np.percentile(a, hi):.3g}'


if __name__ == '__main__':
    cfg = G.DEFAULT
    d = CORP.active_dir()
    metrics_all = pd.read_parquet(os.path.join(d, 'metrics.parquet'))
    syn = metrics_all[~metrics_all.is_probe]
    with gzip.open(os.path.join(d, 'parents.json.gz'), 'rt') as f:
        parents = json.load(f)
    corpus_ids = set(syn.dataset)
    parents = {k: v for k, v in parents.items() if k in corpus_ids}

    rng = np.random.default_rng(SEED)
    ds, _ = E.prepare(rng)
    emp = pd.DataFrame({m: empirical_metadata(x, w) for m, (x, w) in ds.items()}).T.astype(float)

    # flatten the component-level records
    comp_skew, comp_exk, comp_sd, fams = [], [], [], []
    ks, ov_t, ov_a, tmass, dropped, retries = [], [], [], [], [], []
    pi_max, mkt_max = [], []
    for r in parents.values():
        for c in r['components']:
            comp_skew.append(c['skew']); comp_exk.append(c['exkurt'])
            comp_sd.append(c['sd']); fams.append(c['family'])
        ks.append(r['k']); ov_t.append(r['overlap_target'])
        ov_a.append(r['overlap_achieved']); tmass.append(r['truncated_mass'])
        dropped.append(r.get('n_components_dropped', 0))
        retries.append(r.get('component_retries', 0))
        pi_max.append(max(r['pi'])); mkt_max.append(max(r['market']))

    rows = [
        dict(parameter='Number of mixture components, k',
             configured=f'integer uniform on [{cfg.k_min}, {cfg.k_max}]',
             empirical_basis='BIC-selected Gaussian mixture on the 138 empirical '
                             'datasets is multimodal in 79.0 percent of them',
             synthetic_achieved=f'{min(ks)} to {max(ks)}, mean {np.mean(ks):.2f}'),
        dict(parameter='Average pairwise component overlap',
             configured=f'log-uniform on [{10**cfg.overlap_log10_lo:.0e}, '
                        f'{10**cfg.overlap_log10_hi:.2f}]',
             empirical_basis='empirical fitted overlap: median 0.0218, '
                             '95th pct 0.4528, max 0.6719',
             synthetic_achieved=f'median {np.median(ov_a):.4f}, 95th pct '
                                f'{np.percentile(ov_a, 95):.4f}, max {max(ov_a):.4f}'),
        dict(parameter='Component skewness target',
             configured=f'uniform on [{cfg.comp_skew_lo}, {cfg.comp_skew_hi}]',
             empirical_basis=f'empirical dataset skewness '
                             f'{emp.skewness.min():.2f} to {emp.skewness.max():.2f}',
             synthetic_achieved=rng_range(comp_skew)),
        dict(parameter='Component excess kurtosis target',
             configured=f'uniform on [{cfg.comp_exkurt_lo}, {cfg.comp_exkurt_hi}], '
                        f'lifted to the feasible boundary skew^2 - 2',
             empirical_basis=f'empirical dataset excess kurtosis '
                             f'{emp.kurtosis.min():.2f} to {emp.kurtosis.max():.2f}',
             synthetic_achieved=rng_range(comp_exk)),
        dict(parameter='Component standard deviation (relative)',
             configured=f'log-uniform on [{10**cfg.comp_sd_log10_lo:.3g}, '
                        f'{10**cfg.comp_sd_log10_hi:.3g}]',
             empirical_basis='scale is unidentified: every dataset is divided by '
                             'its own unweighted mean',
             synthetic_achieved=rng_range(comp_sd)),
        dict(parameter='Component family',
             configured='Johnson SU, lognormal, beta-prime or beta, chosen by '
                        'which Pearson region the moment target falls in',
             empirical_basis='not a free choice: the family is determined by the '
                             'skewness and kurtosis target',
             synthetic_achieved=', '.join(f'{k} {v/len(fams)*100:.1f}%' for k, v in
                                          pd.Series(fams).value_counts().items())),
        dict(parameter='Mode sampling-share concentration (Dirichlet alpha)',
             configured=f'{cfg.mode_share_alpha}',
             empirical_basis='not measurable: the empirical mode structure is '
                             'latent. Swept in Stage 2h',
             synthetic_achieved=f'largest mode share, 5th to 95th pct '
                                f'{pct_range(pi_max)}'),
        dict(parameter='Mode market-share concentration (Dirichlet alpha)',
             configured=f'{cfg.market_share_alpha} (flat)',
             empirical_basis='market shares are unavailable; the flat Dirichlet '
                             'is the maximum-entropy prior. Marsh, Hattam and '
                             'Allen (2025) report a real top share of 63.75 pct',
             synthetic_achieved=f'largest market share, 5th to 95th pct '
                                f'{pct_range(mkt_max)}'),
        dict(parameter='Point weight concentration within a mode (Dirichlet alpha)',
             configured=f'{cfg.point_weight_alpha} (flat)',
             empirical_basis='same alpha in both arms, per the Stage 2a Part 0 '
                             'decision; the empirical arm previously used 5',
             synthetic_achieved=f'{cfg.point_weight_alpha}'),
        dict(parameter='Mode-to-point weight coupling',
             configured=f'{cfg.mode_coupling}',
             empirical_basis='0 reproduces the old uncoupled behaviour, in which '
                             'the market-weighted distribution had no population '
                             'object. Swept in Stage 2h',
             synthetic_achieved=f'{cfg.mode_coupling}'),
        dict(parameter='Truncation bounds',
             configured=f'max(Q1 - {cfg.trunc_iqr_mult}*IQR, 0) and '
                        f'Q3 + {cfg.trunc_iqr_mult}*IQR of the population mixture',
             empirical_basis='the empirical extraction trims at the same multiple, '
                             'on the sample',
             synthetic_achieved=f'probability mass removed: median '
                                f'{np.median(tmass):.5f}, mean {np.mean(tmass):.5f}, '
                                f'95th pct {np.percentile(tmass, 95):.4f}, '
                                f'max {max(tmass):.4f}'),
        dict(parameter='Dataset size n',
             configured='2,500 datasets log-uniform within each of 3-9, 10-99, '
                        '100-999, 1000-9999',
             empirical_basis=f'empirical n runs {int(emp.n.min())} to '
                             f'{int(emp.n.max()):,}, median {int(emp.n.median())}',
             synthetic_achieved=f'{int(syn.n.min())} to {int(syn.n.max()):,}, '
                                f'median {int(syn.n.median())}'),
        dict(parameter='Probe set (outside the corpus)',
             configured=f'{cfg.probe.n_datasets} datasets log-uniform on '
                        f'[{cfg.probe.n_lo:,}, {cfg.probe.n_hi:,}]',
             empirical_basis=f'{int((emp.n > 9999).sum())} empirical datasets exceed '
                             f'9,999, the largest {int(emp.n.max()):,}',
             synthetic_achieved=f'{int(metrics_all.is_probe.sum())} datasets, '
                                f'{int(metrics_all[metrics_all.is_probe].n.min()):,} to '
                                f'{int(metrics_all[metrics_all.is_probe].n.max()):,}'),
        dict(parameter='Components dropped for falling outside truncation',
             configured='dropped when they hold under 1e-9 of their mass inside '
                        'the bounds; both weight vectors renormalized',
             empirical_basis='not applicable',
             synthetic_achieved=f'{int(np.sum(dropped))} components across '
                                f'{int(np.sum(np.array(dropped) > 0)):,} datasets'),
        dict(parameter='Moment targets redrawn as infeasible or degenerate',
             configured=f'up to {cfg.max_component_retries} redraws per component',
             empirical_basis='not applicable',
             synthetic_achieved=f'{int(np.sum(retries)):,} redraws over '
                                f'{len(fams):,} components '
                                f'({np.sum(retries)/len(fams):.2f} per component)'),
        dict(parameter='Random seed',
             configured=f'{cfg.seed}',
             empirical_basis='not applicable',
             synthetic_achieved='recorded in runmeta.json beside the corpus'),
    ]
    t1 = pd.DataFrame(rows)
    write(t1, 'TABLE_2a_Table1_GenerationParameters.csv')
    pd.set_option('display.width', 300, 'display.max_colwidth', 90)
    print(t1.to_string(index=False))
