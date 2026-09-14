"""Which axis, if any, actually makes a heterogeneous category homogeneous?

Stage 2a-3, after the declared-unit axis was withdrawn. The question is not
whether an axis EXISTS in the metadata but whether splitting on it reduces the
within-dataset dispersion, which is what decides whether the resulting datasets
are a comparable quantity.

The test is the same for every candidate: pooled coefficient of variation of the
category against the record-weighted mean of the within-group coefficients of
variation. A split that leaves the dispersion where it was has not separated
anything, whatever the field is called.

Two author hypotheses are tested here, one confirmed and one refuted. Both are
worth keeping: the refutation is the more useful result.

    conda run -n compareuq python audits/stage2a3/q4_split_axis_evidence.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from datageneration import clean_empirical_symmetric  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'stage2a3')
STORE = os.path.abspath(os.path.join(ROOT, '..', 'EPDsFromEC3', 'store'))
META = os.path.join(ROOT, 'data', 'raw', 'ec3_record_metadata_2026-08-14.csv.gz')
CUTOFF = '2026-08-14'
STRENGTH = 'concrete_compressive_strength_28d_value'

#: Strength classes in psi. The declared values cluster on the standard classes
#: 3000, 3500, 4000, 4500, 5000 and 6000, so these edges cut between them.
PSI_EDGES = [0, 3000, 4000, 5000, 6000, np.inf]
PSI_LABELS = ['<3000 psi', '3000-3999', '4000-4999', '5000-5999', '>=6000 psi']

#: Insulation material type, matched against name + description. Ordered: the
#: first pattern that matches wins, so a record naming two materials is assigned
#: to the more specific one.
MATERIAL = [
    ('MineralWool', r'mineral wool|rock ?wool|stone ?wool|glass ?wool|glasswool'
                    r'|rockwool|laine de verre|laine de roche|steinwolle'
                    r'|glaswolle|lana de roca|lana de vidrio'),
    ('XPS', r'\bxps\b|extruded polystyrene|polystyr[eè]ne extrud'),
    ('EPS', r'\beps\b|expanded polystyrene|polystyr[eè]ne expans|styropor'),
    ('PIR_PUR', r'\bpir\b|\bpur\b|polyiso|polyurethan'),
    ('Cellulose', r'cellulose|ouate de cellulose'),
    ('WoodFibre', r'wood ?fib|fibre de bois|holzfaser'),
    ('Other', r'aerogel|phenolic|perlite|vermiculite|cork|\bhemp\b|chanvre'),
]

#: Insulation thickness in millimetres, parsed from the product name. The
#: store's own `thickness_value` is populated for ZERO of these records.
THICK_EDGES = [0, 40, 80, 120, 200, 1e4]
THICK_LABELS = ['<40 mm', '40-79', '80-119', '120-199', '>=200 mm']


def cv(values, mult=3.0):
    """Unweighted coefficient of variation after the arm's cleaning rule."""
    kept = clean_empirical_symmetric(np.asarray(values, float), mult)
    if len(kept) < 3:
        return len(kept), np.nan
    return len(kept), float(np.std(kept, ddof=1) / np.mean(kept))


def load():
    """Arm records joined to the store fields the candidate axes need."""
    meta = pd.read_csv(META, low_memory=False)
    meta = meta[meta.du_type == meta.modal_du_type].drop(columns=['name'])
    st = pd.read_csv(os.path.join(STORE, 'epd_index.csv.gz'),
                     usecols=['open_xpd_uuid', 'pull_date', 'name',
                              'description', STRENGTH], low_memory=False)
    st = st[st.pull_date <= CUTOFF].drop_duplicates('open_xpd_uuid')
    return meta.merge(st.drop(columns='pull_date'), on='open_xpd_uuid', how='left')


def verdict(df, category, axis, groups):
    """One row per group, plus the pooled row, plus the summary comparison."""
    whole_n, whole_cv = cv(df.ecc)
    rows = [dict(category=category, axis=axis, group='(whole category)',
                 records=len(df), cleaned_n=whole_n, cv=whole_cv)]
    ns, cvs = [], []
    for name, idx in groups.groupby(groups, observed=True):
        n, c = cv(df.loc[idx.index].ecc)
        rows.append(dict(category=category, axis=axis, group=str(name),
                         records=len(idx), cleaned_n=n, cv=c))
        if np.isfinite(c):
            ns.append(n)
            cvs.append(c)
    within = float(np.average(cvs, weights=ns)) if cvs else np.nan
    rows.append(dict(category=category, axis=axis, group='WITHIN-GROUP MEAN',
                     records=len(df), cleaned_n=whole_n, cv=within))
    return rows


def main():
    os.makedirs(TABLES, exist_ok=True)
    df = load()
    txt = (df.name.fillna('') + ' ' + df.description.fillna('')).str.lower()
    out = []

    # 1. the concrete categories, by declared 28-day compressive strength
    for cat in ['ReadyMix', 'CementGrout', 'FlowableFill', 'Shotcrete', 'CMU',
                'ConcretePaving', 'OilPatch', 'Concrete']:
        g = df[df.material_query == cat]
        cls = pd.cut(g[STRENGTH], PSI_EDGES, labels=PSI_LABELS, right=False)
        cls = cls.cat.add_categories('unstated').fillna('unstated')
        out += verdict(g, cat, 'compressive strength class', cls)

    # 2. Insulation, by material type and by declared thickness
    g = df[df.material_query == 'Insulation']
    t = txt.loc[g.index]
    mat = pd.Series('unclassified', index=g.index)
    for name, pat in MATERIAL:
        mat[t.str.contains(pat, regex=True, na=False) & (mat == 'unclassified')] = name
    out += verdict(g, 'Insulation', 'material type (from text)', mat)

    mm = t.str.extract(r'(\d{2,4})\s*mm')[0].astype(float)
    thick = pd.cut(mm, THICK_EDGES, labels=THICK_LABELS, right=False)
    thick = thick.cat.add_categories('no thickness in name').fillna(
        'no thickness in name')
    out += verdict(g, 'Insulation', 'declared thickness (from text)', thick)

    # 3. Insulation's EC3 children, which are already separate datasets
    for cat in ['BoardInsulation', 'BlanketInsulation', 'BlownInsulation',
                'FoamedInPlace']:
        gg = df[df.material_query == cat]
        n, c = cv(gg.ecc)
        out.append(dict(category=cat, axis='(EC3 child of Insulation)',
                        group='(whole category)', records=len(gg),
                        cleaned_n=n, cv=c))

    tab = pd.DataFrame(out)
    tab.to_csv(os.path.join(TABLES, 'TABLE_2a3_SplitAxisEvidence.csv'),
               index=False)
    pd.set_option('display.width', 140)
    for (cat, axis), g in tab.groupby(['category', 'axis'], sort=False):
        print(f'\n=== {cat}  |  {axis} ===')
        print(g[['group', 'records', 'cleaned_n', 'cv']].to_string(
            index=False, float_format=lambda v: f'{v:.2f}'))
    print(f"\nwrote {os.path.relpath(os.path.join(TABLES, 'TABLE_2a3_SplitAxisEvidence.csv'), ROOT)}")


if __name__ == '__main__':
    main()
