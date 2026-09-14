"""Build a frozen, dated raw ECC extract for the 138 empirical categories.

WHY THIS READS A STORE RATHER THAN THE API. The author's decision: a fresh pull
was already running in the EPDsFromEC3 repository, and EC3 rate limits per
account rather than per process, so a second client would have risked truncating
it. See ../EPDsFromEC3/PULLING_EPDS.md section 1.

The substitute is the consolidated EPD store at ../EPDsFromEC3/store, whose
138-category slice was pulled on 2026-08-13 and 2026-08-14 through the
correctly-paginating LucidLCA wrapper. It is five months newer than the 2026-03
data the manuscript reports, and unlike dct_realeccs_trimmed.json it is RAW: no
outlier rule has ever been applied to it. That is what this stage needs, because
a cleaning rule cannot be applied symmetrically to values that were already
trimmed at one end.

WHAT AN ECC IS HERE, and why it is computed this way. The 2026-03 extraction
computed gwp / declared_unit, converting the declared unit with this
repository's own funcs_unit_conversion table and keeping only the majority unit
TYPE in each category. That definition is reproduced exactly, because this stage
is scoped to change the data vintage and the cleaning rule, not the quantity.

Two deliberate departures from the letter of the 2026-03 code, both corrections:

  - Each value is converted by its own unit rather than by the category's
    majority converter. The old consistent_units ran every value through the
    majority type's converter, so a minority-unit record became NaN and was
    dropped with no error. The set of records kept is the same either way; what
    changes is that nothing is silently NaN-ed.
  - The store's own pre-parsed columns are NOT used. The store's unit table has
    no entry for 'km', so it returns NaN for the 254 of 421 PowerCabling records
    declared as "1 km", while this repository's table converts them. Parsing
    declared_unit_raw here keeps those records.

    conda run -n compareuq python p1_build_raw_extract.py
"""
import datetime as dt
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from funcs_unit_conversion import dict_unitconv, dict_unittype, str2valunit  # noqa: E402

STORE = os.path.abspath(os.path.join(ROOT, '..', 'EPDsFromEC3', 'store',
                                     'epd_index.csv.gz'))
OLD = os.path.join(ROOT, 'data', 'processed', 'dct_realeccs_trimmed.json')
RAWDIR = os.path.join(ROOT, 'data', 'raw')
TABLES = os.path.join(ROOT, 'outputs', 'tables', 'stage2a2')

USECOLS = ['material_query', 'pull_date', 'open_xpd_uuid', 'manufacturer',
           'declaration_type', 'date_of_issue', 'date_validity_ends',
           'declared_unit_raw', 'gwp_raw', 'no_expired_records']


def convert(value, unit):
    """Convert one value to its own unit type's canonical unit."""
    t = dict_unittype.get(unit)
    if t is None:
        return np.nan, None
    return dict_unitconv[t]['func'](value, unit, prnt='n'), t


def load_store():
    """The 138 categories, most recent pull each, valid at that pull date."""
    mats = sorted(json.load(open(OLD)))
    df = pd.read_csv(STORE, usecols=USECOLS, low_memory=False)
    df = df[df.material_query.isin(mats)].copy()
    latest = df.groupby('material_query').pull_date.max().rename('_latest')
    df = df.merge(latest, on='material_query')
    df = df[df.pull_date == df._latest].drop(columns='_latest')
    n_all = len(df)
    # The Aug-2026 pulls include expired declarations; the 2026-03 pull did not,
    # because the ec3 library filters them silently. Match the 2026-03 scope so
    # this stage changes the vintage and the cleaning rule, not the population.
    ends = pd.to_datetime(df.date_validity_ends, errors='coerce')
    df = df[ends > pd.to_datetime(df.pull_date)].copy()
    return df, mats, n_all


def to_ecc(df):
    """gwp / declared_unit, per record, in the category's majority unit type."""
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

    out, rows = [], []
    for mat, g in df.groupby('material_query'):
        usable = g[(g.gwp_type == 'emissions') & g.du_type.notna()
                   & np.isfinite(g.du_value) & (g.du_value > 0)
                   & np.isfinite(g.gwp_value)]
        if usable.empty:
            rows.append(dict(material=mat, n_pulled=len(g), n_usable=0,
                             unit_type=None, canonical_unit=None,
                             n_in_unit_type=0, n_ecc=0))
            continue
        modal = usable.du_type.value_counts().index[0]
        sel = usable[usable.du_type == modal].copy()
        sel['ecc'] = sel.gwp_value / sel.du_value
        sel = sel[np.isfinite(sel.ecc) & (sel.ecc > 0)]
        rows.append(dict(material=mat, n_pulled=len(g), n_usable=len(usable),
                         unit_type=modal,
                         canonical_unit=dict_unitconv[modal]['output'],
                         n_in_unit_type=int((usable.du_type == modal).sum()),
                         n_ecc=len(sel)))
        out.append(sel.assign(ecc_unit_type=modal,
                              ecc_unit=f"kgco2e/{dict_unitconv[modal]['output']}"))
    recs = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return recs, pd.DataFrame(rows)


if __name__ == '__main__':
    os.makedirs(RAWDIR, exist_ok=True)
    os.makedirs(TABLES, exist_ok=True)

    df, mats, n_all = load_store()
    pulls = sorted(df.pull_date.unique())
    print(f'store slice: {n_all:,} records across {df.material_query.nunique()} '
          f'of the {len(mats)} categories, pulled {pulls}')
    print(f'valid at pull date: {len(df):,} ({len(df)/n_all*100:.1f}%)')

    recs, per_cat = to_ecc(df)
    print(f'ECC values reconstructed: {len(recs):,} across '
          f'{recs.material_query.nunique()} categories')

    keep = ['material_query', 'open_xpd_uuid', 'manufacturer', 'declaration_type',
            'date_of_issue', 'date_validity_ends', 'pull_date',
            'declared_unit_raw', 'du_value', 'du_type', 'gwp_raw', 'gwp_value',
            'ecc', 'ecc_unit']
    frozen = recs[keep].sort_values(['material_query', 'open_xpd_uuid'])
    label = max(pulls)
    path = os.path.join(RAWDIR, f'ec3_raw_ecc_{label}.csv.gz')
    if os.path.exists(path):
        raise FileExistsError(f'{path} exists; a frozen input is never overwritten.')
    frozen.to_csv(path, index=False, compression='gzip')

    digest = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    meta = dict(
        label=label,
        built_on=dt.date.today().isoformat(),
        source=os.path.relpath(STORE, ROOT),
        source_sha256=hashlib.sha256(open(STORE, 'rb').read()).hexdigest(),
        source_kind='frozen slice of the consolidated EPD store, not a live pull',
        query=('EC3 store slice: material_query in the 138 categories of '
               'dct_realeccs_trimmed.json, most recent pull_date per category, '
               'date_validity_ends > pull_date'),
        pull_dates=[str(p) for p in pulls],
        records_in_store_slice=int(n_all),
        records_valid_at_pull=int(len(df)),
        ecc_values=int(len(frozen)),
        categories=int(frozen.material_query.nunique()),
        ecc_definition=('gwp / declared_unit, each converted by its own unit with '
                        'src/funcs_unit_conversion, restricted per category to the '
                        'majority declared-unit TYPE'),
        cleaning_applied='none; this file is raw',
        sha256=digest,
    )
    with open(os.path.join(RAWDIR, f'ec3_raw_ecc_{label}_runmeta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    per_cat.to_csv(os.path.join(TABLES, 'TABLE_2a2_RawExtractByCategory.csv'),
                   index=False)
    print(f'\nwrote {os.path.relpath(path, ROOT)}')
    print(f'sha256 {digest}')
