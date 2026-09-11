"""Stage 2a, Part 0 item 1b: how much did the seeding collapse reduce the
effective coverage of the metric space across the shipped corpus?

Three arms:
  shipped   the 10,000 analysed datasets in the frozen DATA_all.json
  legacy    a reproducible rerun of the frozen pre-Stage-1 generator
  current   the Stage 1 generator, correct seeding, algorithm otherwise identical

Writes TABLE_2a_CollapseDamage.csv and TABLE_2a_CollapseDuplicates.csv
"""
import numpy as np, pandas as pd, sys, os, time
from scipy.stats import rankdata
from _common import ROOT, TABLES, load_shipped, metrics_frame, write

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import legacy_generator_pre_stage1 as legacy
import datageneration as current
from customstats import empirical_metadata

N_DATASETS = 15_000
SEED = 20260911


def gen_legacy(ndatasets=N_DATASETS, seed=SEED):
    """Reproducible instance of the legacy algorithm.

    The legacy code drew structural parameters from default_rng(None) and point
    weights plus the lognorm branch from global numpy state, so the shipped run
    cannot be reproduced. Seeding the structural rng per dataset and the global
    stream once gives a faithful, reproducible draw from the same algorithm:
    generate_random_numbers keeps its seed=0 default, which is the defect.
    """
    np.random.seed(seed)
    rng_n = np.random.default_rng(seed)
    out = {}
    for i in range(ndatasets):
        n = int(np.round(10 ** rng_n.uniform(np.log(4) / np.log(10), np.log(1000) / np.log(10))))
        data, w = legacy.random_irregular_dataset(n=n, seed=seed + i)
        out[f'dataset{i}'] = dict(data=data, weights=w,
                                  metrics=empirical_metadata(data, w))
    return out


def gen_current(ndatasets=N_DATASETS, seed=SEED):
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(ndatasets):
        n = int(current.random_logcount(rng, lo=4, hi=1000, n=1)[0])
        data, w = current.random_irregular_dataset(n=n, rng=rng)
        out[f'dataset{i}'] = dict(data=data, weights=w,
                                  metrics=empirical_metadata(data, w))
    return out


def effective_dimension(M):
    """Participation ratio of the correlation-matrix eigenvalues.

    (sum lambda)^2 / sum(lambda^2). Equals p for p independent metrics and 1
    when every metric is a copy of one underlying quantity.
    """
    M = M[np.all(np.isfinite(M), axis=1)]
    Z = (M - M.mean(0)) / M.std(0)
    lam = np.linalg.eigvalsh(np.corrcoef(Z, rowvar=False))
    lam = np.clip(lam, 0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def shape_duplicate_rate(DATA, keys, nmin=20, rho_cut=0.999, max_per_group=60, rng=None):
    """Fraction of same-n dataset pairs whose value ORDER is identical.

    Under the collapse, two datasets built from the same component type with
    the same count share one underlying vector of standard draws. Every later
    step (loc, scale, the power transform, +1, division by the mean) is
    monotone increasing, and the reflection is monotone decreasing, so the
    order of the values -- and hence |Spearman rho| -- survives all of them.
    |rho| = 1 between two independently generated datasets is therefore a
    direct fingerprint of the defect, not a coincidence.
    """
    rng = rng or np.random.default_rng(0)
    bysize = {}
    for k in keys:
        d = np.asarray(DATA[k]['data'], float)
        if len(d) >= nmin:
            bysize.setdefault(len(d), []).append(rankdata(d))
    npairs = ndup = 0
    for n, ranks in bysize.items():
        if len(ranks) < 2:
            continue
        if len(ranks) > max_per_group:
            idx = rng.choice(len(ranks), max_per_group, replace=False)
            ranks = [ranks[j] for j in idx]
        R = np.array(ranks)
        Z = (R - R.mean(1, keepdims=True)) / R.std(1, keepdims=True)
        C = np.abs(Z @ Z.T) / R.shape[1]
        iu = np.triu_indices(len(R), 1)
        npairs += len(iu[0])
        ndup += int(np.sum(C[iu] > rho_cut))
    return ndup, npairs, (ndup / npairs if npairs else np.nan)


METRICS_CORE = ['coeffvar', 'entropy', 'skewness', 'kurtosis', 'mode_count_est',
                'weight_outliers', 'fit_norm_SW', 'fit_lognorm_SW', 'w_v_uw_wasserstein']

if __name__ == '__main__':
    t0 = time.time()
    print('loading shipped ...')
    DATA_ship, keep, _, _ = load_shipped()
    dfs = metrics_frame({k: DATA_ship[k] for k in keep})
    print(f'  {len(keep)} analysed datasets   {time.time()-t0:.0f}s')

    print('generating legacy arm ...')
    DATA_leg = gen_legacy()
    keys_leg = list(DATA_leg)
    dfl = metrics_frame(DATA_leg)
    print(f'  done  {time.time()-t0:.0f}s')

    print('generating current arm ...')
    DATA_cur = gen_current()
    keys_cur = list(DATA_cur)
    dfc = metrics_frame(DATA_cur)
    print(f'  done  {time.time()-t0:.0f}s')

    rows = []
    arms = [('shipped', dfs, DATA_ship, keep), ('legacy_rerun', dfl, DATA_leg, keys_leg),
            ('current', dfc, DATA_cur, keys_cur)]
    for name, df, DATA, keys in arms:
        sub = df[METRICS_CORE].astype(float)
        rows.append(dict(arm=name, quantity='n_datasets', value=len(df)))
        rows.append(dict(arm=name, quantity='effective_metric_dimension',
                         value=effective_dimension(sub.values)))
        nd, npr, rate = shape_duplicate_rate(DATA, keys)
        rows.append(dict(arm=name, quantity='order_duplicate_pairs', value=nd))
        rows.append(dict(arm=name, quantity='order_duplicate_pairs_tested', value=npr))
        rows.append(dict(arm=name, quantity='order_duplicate_rate', value=rate))
        for m in METRICS_CORE:
            s = sub[m].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(dict(arm=name, quantity=f'sd__{m}', value=float(s.std())))
            rows.append(dict(arm=name, quantity=f'iqr__{m}',
                             value=float(s.quantile(.75) - s.quantile(.25))))
            rows.append(dict(arm=name, quantity=f'p01__{m}', value=float(s.quantile(.01))))
            rows.append(dict(arm=name, quantity=f'p99__{m}', value=float(s.quantile(.99))))
    out = pd.DataFrame(rows)
    write(out, 'TABLE_2a_CollapseDamage.csv')

    piv = out.pivot(index='quantity', columns='arm', values='value')
    piv = piv[['shipped', 'legacy_rerun', 'current']]
    piv['current_over_legacy'] = piv['current'] / piv['legacy_rerun']
    pd.set_option('display.width', 200)
    print(piv.to_string(float_format=lambda v: f'{v:,.4f}'))
