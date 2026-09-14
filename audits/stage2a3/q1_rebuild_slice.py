"""Reconstruct the 2026-08 store slice and verify it reproduces the frozen extract.

The frozen extract `data/raw/ec3_raw_ecc_2026-08-14.csv.gz` holds only the
records that survived the majority-declared-unit-type filter, and it carries no
EC3 category path. Stage 2a-3 needs both: the minority-unit records, because the
declared unit type is a split axis, and the category path, because an EC3
category may be a parent of several subcategories.

The consolidated store at ../EPDsFromEC3/store APPENDS pulls rather than
replacing them, so the 2026-08-13/14 rows are still present alongside a newer
2026-09-12 pull. This script pins the slice to pull_date <= 2026-08-14 and then
checks, record by record, that the ECC reconstruction is IDENTICAL to the frozen
file. Nothing downstream may use this reconstruction unless that check passes:
the frozen file is the input of record, and this is only a way of recovering
metadata that was dropped when it was written.

    conda run -n compareuq python audits/stage2a3/q1_rebuild_slice.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'audits', 'stage2a2'))

from funcs_unit_conversion import dict_unitconv, dict_unittype, str2valunit  # noqa: E402

STORE = os.path.abspath(os.path.join(ROOT, '..', 'EPDsFromEC3', 'store',
                                     'epd_index.csv.gz'))
OLD = os.path.join(ROOT, 'data', 'processed', 'dct_realeccs_trimmed.json')
FROZEN = os.path.join(ROOT, 'data', 'raw', 'ec3_raw_ecc_2026-08-14.csv.gz')
SIDECAR = os.path.join(ROOT, 'data', 'raw', 'ec3_record_metadata_2026-08-14.csv.gz')
CUTOFF = '2026-08-14'

USECOLS = ['material_query', 'pull_date', 'open_xpd_uuid', 'manufacturer',
           'declaration_type', 'date_of_issue', 'date_validity_ends',
           'declared_unit_raw', 'gwp_raw', 'category_key', 'category', 'name']


def convert(value, unit):
    t = dict_unittype.get(unit)
    if t is None:
        return np.nan, None
    return dict_unitconv[t]['func'](value, unit, prnt='n'), t


def rebuild():
    """The 2026-08 slice, valid at pull date, with ECC and unit type per record."""
    mats = sorted(json.load(open(OLD)))
    df = pd.read_csv(STORE, usecols=USECOLS, low_memory=False)
    df = df[df.material_query.isin(mats) & (df.pull_date <= CUTOFF)].copy()
    latest = df.groupby('material_query').pull_date.max().rename('_latest')
    df = df.merge(latest, on='material_query')
    df = df[df.pull_date == df._latest].drop(columns='_latest')
    n_all = len(df)
    ends = pd.to_datetime(df.date_validity_ends, errors='coerce')
    df = df[ends > pd.to_datetime(df.pull_date)].copy()

    du = df.declared_unit_raw.astype(str).apply(str2valunit)
    gw = df.gwp_raw.astype(str).apply(str2valunit)
    df = df.assign(du_qty=[a for a, _ in du], du_unit=[b for _, b in du],
                   gwp_qty=[a for a, _ in gw], gwp_unit=[b for _, b in gw])
    conv = [convert(q, u) for q, u in zip(df.du_qty, df.du_unit)]
    df['du_value'] = [v for v, _ in conv]
    df['du_type'] = [t for _, t in conv]
    gconv = [convert(q, u) for q, u in zip(df.gwp_qty, df.gwp_unit)]
    df['gwp_value'] = [v for v, _ in gconv]
    df['gwp_type'] = [t for _, t in gconv]

    usable = df[(df.gwp_type == 'emissions') & df.du_type.notna()
                & np.isfinite(df.du_value) & (df.du_value > 0)
                & np.isfinite(df.gwp_value)].copy()
    usable['ecc'] = usable.gwp_value / usable.du_value
    # The modal unit type is decided on `usable`, BEFORE the ecc > 0 filter,
    # exactly as p1_build_raw_extract.to_ecc decides it. WindTurbines is a 3-3
    # tie between vol and area and the two orders disagree, so this is not
    # cosmetic.
    modal = (usable.groupby('material_query').du_type
             .agg(lambda s: s.value_counts().index[0]).rename('modal_du_type'))
    usable = usable.merge(modal, on='material_query')
    usable = usable[np.isfinite(usable.ecc) & (usable.ecc > 0)].copy()
    return usable, n_all


def verify(usable):
    """The majority-unit-type subset must reproduce the frozen extract exactly."""
    frozen = pd.read_csv(FROZEN, low_memory=False)
    rebuilt = usable[usable.du_type == usable.modal_du_type]

    a = rebuilt.set_index('open_xpd_uuid').sort_index()
    b = frozen.set_index('open_xpd_uuid').sort_index()
    ok_ids = a.index.equals(b.index)
    # The frozen file is CSV text, so the comparison is at round-trip precision,
    # not bitwise. The reported worst relative difference is the evidence.
    rel = (np.abs(a.ecc.to_numpy() - b.ecc.to_numpy())
           / np.abs(b.ecc.to_numpy())) if ok_ids else np.array([np.inf])
    ok_ecc = ok_ids and np.nanmax(rel) < 1e-12
    ok_cat = ok_ids and (a.material_query == b.material_query).all()
    print(f'frozen rows {len(b):,}   rebuilt majority-unit rows {len(a):,}')
    print(f'identical record ids : {ok_ids}')
    print(f'identical ecc values : {ok_ecc}  '
          f'(worst relative difference {np.nanmax(rel):.2e})')
    print(f'identical categories : {ok_cat}')
    if not (ok_ids and ok_ecc and ok_cat):
        missing = b.index.difference(a.index)
        extra = a.index.difference(b.index)
        print(f'  in frozen not rebuilt: {len(missing)}  {list(missing[:5])}')
        print(f'  in rebuilt not frozen: {len(extra)}  {list(extra[:5])}')
        raise SystemExit('reconstruction does NOT match the frozen extract; stop.')
    return rebuilt


if __name__ == '__main__':
    usable, n_all = rebuild()
    print(f'store slice at pull_date <= {CUTOFF}: {n_all:,} records')
    print(f'usable ECC records, ALL unit types: {len(usable):,} across '
          f'{usable.material_query.nunique()} categories')
    verify(usable)

    cols = ['open_xpd_uuid', 'material_query', 'category_key', 'category',
            'name', 'declared_unit_raw', 'du_value', 'du_type',
            'modal_du_type', 'ecc', 'pull_date']
    out = usable[cols].sort_values(['material_query', 'open_xpd_uuid'])
    out.to_csv(SIDECAR, index=False, compression='gzip')
    print(f'\nwrote {os.path.relpath(SIDECAR, ROOT)}  ({len(out):,} rows)')
    print(f'categories with >1 declared unit type: '
          f'{(out.groupby("material_query").du_type.nunique() > 1).sum()}')
    print(f'categories with >1 EC3 category_key  : '
          f'{(out.groupby("material_query").category_key.nunique() > 1).sum()}')
