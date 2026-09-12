"""What the symmetric cleaning rule does to the new raw extract.

Reports the categories that change, the categories that fall below the
inclusion threshold, and the resulting empirical arm, then re-runs the cleaning
sensitivity. The previous sensitivity (Stage 2a, TABLE_2a_EmpiricalCleaningSensitivity)
was measured on a store reconstruction that used gwp_per_kg, and its baseline
was a file already trimmed once. This one is measured on raw values under the
ECC definition the analysis actually uses, so it is the first honest version.

    conda run -n compareuq python p4_cleaning_report.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import empirical  # noqa: E402
from customstats import empirical_metadata  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'stage2a2')
OLD = os.path.join(ROOT, 'data', 'processed', 'dct_realeccs_trimmed.json')
SEED = 42

METRICS = ['n', 'coeffvar', 'entropy', 'skewness', 'kurtosis', 'modality_index',
           'weight_outliers', 'fit_norm_SW', 'fit_lognorm_SW',
           'w_v_uw_wasserstein']


def rule(data, name, mult=3.0):
    d = np.asarray(data, float)
    d = d[np.isfinite(d) & (d > 0)]
    if name == 'none':
        return d
    q1, q3 = np.quantile(d, [0.25, 0.75])
    iqr = q3 - q1
    if name == 'additive_3iqr_2026-03_rule':
        return d[(d > q1 - 3 * iqr) & (d < q3 + 3 * iqr)]
    if name == 'additive_high_only_3iqr':
        return d[d < q3 + 3 * iqr]
    L = np.log(d)
    l1, l3 = np.quantile(L, [0.25, 0.75])
    li = l3 - l1
    if li <= 0:
        return d
    if name == 'log_3iqr_low_only_stage2a':
        return d[L > l1 - 3 * li]
    if name == 'log_3iqr_symmetric':
        return d[(L > l1 - 3 * li) & (L < l3 + 3 * li)]
    if name == 'log_1.5iqr_symmetric':
        return d[(L > l1 - 1.5 * li) & (L < l3 + 1.5 * li)]
    raise ValueError(name)


RULES = ['none', 'additive_3iqr_2026-03_rule', 'additive_high_only_3iqr',
         'log_3iqr_low_only_stage2a', 'log_3iqr_symmetric',
         'log_1.5iqr_symmetric']


def metrics_under(raw, name, seed=SEED):
    rng = np.random.default_rng(seed)
    out = {}
    for mat in sorted(raw):
        d = rule(raw[mat], name)
        if len(d) < 3:
            continue
        w = rng.dirichlet(np.ones(len(d)))
        try:
            out[mat] = empirical_metadata(d / np.mean(d), w)
        except Exception:
            continue
    return pd.DataFrame(out).T.astype(float)


def main():
    raw = empirical.load_raw()
    print(f'raw extract: {len(raw)} categories, '
          f'{sum(len(v) for v in raw.values()):,} values\n')

    # ---- what the symmetric rule does, category by category ----------------
    rng = np.random.default_rng(SEED).spawn(1)[0]
    ds, report = empirical.prepare(rng)
    rep = pd.DataFrame(report)
    rep.to_csv(os.path.join(TABLES, 'TABLE_2a2_CleaningReport.csv'), index=False)

    ok = rep[rep.status == 'ok']
    print('=== symmetric log-space cleaning, mult = 3 ===')
    print(f'  values in   {rep.n_before.sum():,}')
    print(f'  removed     {rep.n_removed.sum():,} '
          f'({rep.n_removed.sum()/rep.n_before.sum()*100:.3f}%)')
    print(f'    at the low end  {rep.n_removed_low.sum():,}')
    print(f'    at the high end {rep.n_removed_high.sum():,}')
    print(f'  categories losing at least one value: {(rep.n_removed > 0).sum()} '
          f'of {len(rep)}')
    print(f'\n  CATEGORIES RETAINED: {len(ds)} of {len(rep)}')
    dropped = rep[rep.status != 'ok']
    if len(dropped):
        print('  dropped for fewer than 3 values after cleaning:')
        print(dropped[['material', 'n_before', 'n_after']]
              .to_string(index=False))

    print(f'\n  worst near-zero values before cleaning:')
    print(rep.nsmallest(6, 'min_over_mean_before')[
        ['material', 'n_before', 'min_over_mean_before', 'min_over_mean_after']]
        .to_string(index=False, float_format=lambda v: f'{v:,.4g}'))
    print(f'\n  10 categories losing the most values:')
    print(rep.nlargest(10, 'n_removed')[
        ['material', 'n_before', 'n_after', 'n_removed_low', 'n_removed_high']]
        .to_string(index=False))

    # ---- comparison against the 2026-03 arm --------------------------------
    old = {m: np.asarray(v['data'], float) for m, v in json.load(open(OLD)).items()}
    print(f'\n=== the empirical arm, then and now ===')
    print(f'  2026-03, additive high-end trim   {len(old):>4} categories, '
          f'{sum(len(v) for v in old.values()):>8,} values')
    print(f'  2026-08, symmetric log-space trim {len(ds):>4} categories, '
          f'{int(ok.n_after.sum()):>8,} values')
    gone = sorted(set(old) - set(ds))
    print(f'  categories lost: {gone if gone else "none"}')
    print(f'  categories gained: '
          f'{sorted(set(ds) - set(old)) if set(ds) - set(old) else "none"}')

    # ---- cleaning sensitivity ---------------------------------------------
    print(f'\n=== cleaning sensitivity on raw values ===')
    rows, base = [], None
    for name in RULES:
        m = metrics_under(raw, name)
        if name == 'none':
            base = m
        kept = sum(len(rule(v, name)) for v in raw.values())
        tot = sum(len(v) for v in raw.values())
        for col in METRICS:
            s = m[col].replace([np.inf, -np.inf], np.nan).dropna()
            b = base[col].replace([np.inf, -np.inf], np.nan).dropna()
            common = s.index.intersection(b.index)
            rows.append(dict(
                rule=name, metric=col, n_datasets=len(m),
                frac_values_removed=1 - kept / tot,
                median=float(s.median()), sd=float(s.std()),
                shift_in_sd_units=float((s[common] - b[common]).abs().mean()
                                        / b.std()) if b.std() else np.nan))
    sens = pd.DataFrame(rows)
    sens.to_csv(os.path.join(TABLES, 'TABLE_2a2_CleaningSensitivity.csv'),
                index=False)
    pd.set_option('display.width', 220)
    piv = sens.pivot(index='metric', columns='rule', values='shift_in_sd_units')
    piv = piv[[r for r in RULES if r != 'none']]
    print('mean absolute metric shift vs NO cleaning, in sd units of the '
          'uncleaned metric:')
    print(piv.to_string(float_format=lambda v: f'{v:,.3f}'))
    print('\nfraction of values removed, and categories surviving:')
    for name in RULES:
        sub = sens[sens.rule == name].iloc[0]
        print(f'  {name:<28} removed {sub.frac_values_removed*100:5.2f}%   '
              f'categories {int(sub.n_datasets)}')


if __name__ == '__main__':
    main()
