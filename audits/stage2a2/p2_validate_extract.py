"""Validate the frozen raw extract before anything is built on it.

Three checks, none of which is a row count on its own:

  1. Attrition. Every record in the 206,668-record store slice is accounted for
     as kept, expired, or unusable, with a reason.
  2. Duplication. Distinct open_xpd_uuid against rows, per category. A pull that
     re-fetches page 1 reaches the right row total and reports success; only a
     distinct-id check catches it. See ../EPDsFromEC3/PULLING_EPDS.md section 1.
  3. Agreement with the 2026-03 arm. Category medians are robust to the trimming
     the old file carries, so a category whose median has moved by orders of
     magnitude is a definitional disagreement, not five months of new products.

Also reports every category that grew, shrank, or would fall below the
three-value inclusion threshold.

    conda run -n compareuq python p2_validate_extract.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
TABLES = os.path.join(ROOT, 'outputs', 'tables', 'stage2a2')
RAW = os.path.join(ROOT, 'data', 'raw', 'ec3_raw_ecc_2026-08-14.csv.gz')
OLD = os.path.join(ROOT, 'data', 'processed', 'dct_realeccs_trimmed.json')
MIN_N = 3


def main():
    new = pd.read_csv(RAW, low_memory=False)
    old = {m: np.asarray(v['data'], float)
           for m, v in json.load(open(OLD)).items()}
    meta = json.load(open(RAW.replace('.csv.gz', '_runmeta.json')))

    print('=== 1. attrition from the store slice ===')
    n_all = meta['records_in_store_slice']
    n_valid = meta['records_valid_at_pull']
    print(f'  store slice, 138 categories, latest pull each  {n_all:>8,}')
    print(f'  expired at pull date, dropped                  {n_all-n_valid:>8,}'
          f'  ({(n_all-n_valid)/n_all*100:.1f}%)')
    print(f'  valid at pull date                             {n_valid:>8,}')
    print(f'  no usable gwp or declared unit, or minority     '
          f'{n_valid-len(new):>7,}  ({(n_valid-len(new))/n_valid*100:.1f}%)')
    print(f'  ECC values retained                            {len(new):>8,}')

    print('\n=== 2. duplication ===')
    rows, uniq = len(new), new.open_xpd_uuid.nunique(dropna=True)
    print(f'  rows {rows:,}   distinct open_xpd_uuid {uniq:,}   '
          f'duplicate rows {rows-uniq:,} ({(rows-uniq)/rows*100:.2f}%)')
    per = new.groupby('material_query').agg(
        rows=('ecc', 'size'), uniq=('open_xpd_uuid', 'nunique'))
    per['dup'] = per.rows - per.uniq
    print(f'  categories with any duplicate uuid: {(per.dup > 0).sum()} of {len(per)}')
    if (per.dup > 0).any():
        print(per[per.dup > 0].nlargest(5, 'dup').to_string())
    xcat = new.dropna(subset=['open_xpd_uuid']).groupby(
        'open_xpd_uuid').material_query.nunique()
    print(f'  EPDs appearing in more than one category: {int((xcat > 1).sum()):,}'
          f' of {len(xcat):,}')
    ratio = rows / uniq if uniq else np.nan
    print(f'  rows/unique = {ratio:.4f}  '
          f'({"integer -> suspect pagination" if abs(ratio-round(ratio)) < 1e-9 and ratio > 1 else "not an integer multiple"})')

    print('\n=== 3. per-category comparison against the 2026-03 arm ===')
    out = []
    for mat in sorted(old):
        o = old[mat]
        g = new[new.material_query == mat].ecc.values
        out.append(dict(
            material=mat, n_old_trimmed=len(o), n_new_raw=len(g),
            delta=len(g) - len(o),
            med_old=float(np.median(o)),
            med_new=float(np.median(g)) if len(g) else np.nan,
            med_ratio=float(np.median(g) / np.median(o)) if len(g) else np.nan,
            below_min_n=len(g) < MIN_N))
    c = pd.DataFrame(out)
    c['abs_log_ratio'] = np.abs(np.log10(c.med_ratio))
    c.to_csv(os.path.join(TABLES, 'TABLE_2a2_CategoryComparison.csv'), index=False)

    print(f'  categories present in both                : {len(c)}')
    print(f'  grew                                      : {(c.delta > 0).sum()}')
    print(f'  unchanged in count                        : {(c.delta == 0).sum()}')
    print(f'  shrank                                    : {(c.delta < 0).sum()}')
    print(f'  fall below n = {MIN_N}                            : '
          f'{int(c.below_min_n.sum())}')
    print(f'  total values  old {c.n_old_trimmed.sum():,} (trimmed) -> '
          f'new {c.n_new_raw.sum():,} (raw)')
    print(f'\n  median of the per-category median ratio   : {c.med_ratio.median():.4f}')
    print(f'  within 2x  : {int(c.med_ratio.between(0.5, 2).sum())} of {len(c)}')
    print(f'  within 10x : {int(c.med_ratio.between(0.1, 10).sum())} of {len(c)}')
    print('\n  worst 12 by |log10 median ratio|:')
    print(c.nlargest(12, 'abs_log_ratio')[
        ['material', 'n_old_trimmed', 'n_new_raw', 'med_old', 'med_new',
         'med_ratio']].to_string(index=False, float_format=lambda v: f'{v:,.4g}'))
    print('\n  20 largest shrinkages:')
    print(c.nsmallest(20, 'delta')[
        ['material', 'n_old_trimmed', 'n_new_raw', 'delta', 'med_ratio']]
        .to_string(index=False, float_format=lambda v: f'{v:,.4g}'))


if __name__ == '__main__':
    main()
