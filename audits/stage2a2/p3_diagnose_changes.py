"""Explain every category that shrank, and every category whose median moved.

A shrinkage is a pagination failure until proven otherwise. The proof available
here is that the store slice, counted BEFORE the validity filter, still contains
at least as many records as the 2026-03 arm held: if it does, the category did
not lose records from the pull, it lost them to expiry.

A median that moves by an order of magnitude is not five months of new products.
It is the two extractions disagreeing about which declared-unit TYPE the
category is measured in, which changes both the divisor and the set of records
kept. That is reported per category rather than reconciled, because the unit
type is a property of what EC3 holds today.

    conda run -n compareuq python p3_diagnose_changes.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from funcs_unit_conversion import dict_unittype, str2valunit  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'stage2a2')
RAW = os.path.join(ROOT, 'data', 'raw', 'ec3_raw_ecc_2026-08-14.csv.gz')
OLD = os.path.join(ROOT, 'data', 'processed', 'dct_realeccs_trimmed.json')
STORE = os.path.abspath(os.path.join(ROOT, '..', 'EPDsFromEC3', 'store',
                                     'epd_index.csv.gz'))


def main():
    new = pd.read_csv(RAW, low_memory=False)
    old = {m: np.asarray(v['data'], float) for m, v in json.load(open(OLD)).items()}
    st = pd.read_csv(STORE, usecols=['material_query', 'pull_date',
                                     'declared_unit_raw', 'date_validity_ends'],
                     low_memory=False)
    st = st[st.material_query.isin(old)]
    latest = st.groupby('material_query').pull_date.max().rename('_l')
    st = st.merge(latest, on='material_query')
    st = st[st.pull_date == st._l].drop(columns='_l')

    # declared-unit type of every record in the slice, expired included
    du = st.declared_unit_raw.astype(str).apply(str2valunit)
    st = st.assign(du_unit=[b for _, b in du])
    st['du_type'] = st.du_unit.map(dict_unittype)

    rows = []
    for mat in sorted(old):
        g = st[st.material_query == mat]
        n_slice = len(g)
        n_new = int((new.material_query == mat).sum())
        n_old = len(old[mat])
        vc = g.du_type.value_counts()
        types = '|'.join(f'{t}:{c}' for t, c in vc.head(3).items())
        newtype = new.loc[new.material_query == mat, 'du_type']
        rows.append(dict(
            material=mat, n_old=n_old, n_slice_incl_expired=n_slice, n_new=n_new,
            delta=n_new - n_old,
            slice_covers_old=n_slice >= n_old,
            unit_type_new=newtype.iloc[0] if len(newtype) else None,
            unit_type_mix=types,
            n_unit_types=int(g.du_type.nunique(dropna=True)),
            med_ratio=float(np.median(new.loc[new.material_query == mat, 'ecc'])
                            / np.median(old[mat])) if n_new else np.nan))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TABLES, 'TABLE_2a2_ChangeDiagnosis.csv'), index=False)

    shr = d[d.delta < 0]
    print(f'=== {len(shr)} categories shrank ===')
    print(f'  store slice (expired included) still covers the 2026-03 count in '
          f'{int(shr.slice_covers_old.sum())} of {len(shr)}')
    bad = shr[~shr.slice_covers_old]
    if len(bad):
        print('  NOT covered, so not explained by expiry:')
        print(bad[['material', 'n_old', 'n_slice_incl_expired', 'n_new',
                   'unit_type_mix']].to_string(index=False))
    else:
        print('  every shrinkage is accounted for by expiry, not by a short pull.')

    print(f'\n=== categories whose median moved more than 10x ===')
    big = d[(d.med_ratio > 10) | (d.med_ratio < 0.1)].copy()
    big['abs_log'] = np.abs(np.log10(big.med_ratio))
    print(f'  {len(big)} of {len(d)}; all have more than one declared-unit type: '
          f'{bool((big.n_unit_types > 1).all())}')
    print(big.sort_values('abs_log', ascending=False)[
        ['material', 'n_old', 'n_new', 'med_ratio', 'unit_type_new',
         'unit_type_mix']].to_string(index=False,
                                     float_format=lambda v: f'{v:,.4g}'))

    print(f'\n=== categories now below n = 3 ===')
    tiny = d[d.n_new < 3]
    print(tiny[['material', 'n_old', 'n_slice_incl_expired', 'n_new',
                'unit_type_mix']].to_string(index=False) if len(tiny)
          else '  none')

    print(f'\n=== the largest categories, old vs new ===')
    print(d.nlargest(8, 'n_new')[['material', 'n_old', 'n_slice_incl_expired',
                                  'n_new', 'delta']]
          .to_string(index=False))


if __name__ == '__main__':
    main()
