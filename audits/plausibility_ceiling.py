"""Measure the physical-plausibility ceiling on mass-declared ECC records.

an earlier revision, task 1. The author's decision of 2026-09-14: remove records that
cannot be a real product, and report the extremes that no external bound can
rule on.

THE RULE MUST BE EXTERNAL, NOT DISTRIBUTIONAL. This study measures the
dispersion and modality of ECC datasets, so a bound read off the arm's own
quantiles, standard deviations or visible gaps would be circular in exactly the
way a dispersion-based category split would have been (decision 46). The bound
therefore comes from material science and is applied only where such a bound
exists, which is the mass-declared categories.

WHAT THIS SCRIPT REPORTS

1. the gate: how many records and how many cleaned values the ceiling removes,
   against the 0.1 percent of 117,090 the author set as the stop-and-report
   threshold
2. which datasets lose records, and what it does to their characteristics
3. the ten highest and ten lowest records per declared-unit type, with product
   names and the ratio to the category median, as a table for author review.
   This is a REPORT, not a filter: for volume, area, length and item
   declarations there is no comparably tight external bound and none is invented
4. how many records the extraction excludes for having a non-positive GWP. A
   count for the manuscript, not a change

    conda run -n compareuq python audits/plausibility_ceiling.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import categorysplit  # noqa: E402
import empirical  # noqa: E402
from customstats import empirical_metadata  # noqa: E402
from datageneration import clean_empirical_symmetric  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')

#: The gate the author set: a handful of records means a declaration error,
#: hundreds means the bound is wrong or the problem is something else.
GATE_FRACTION = 0.001
ARM_VALUES_BEFORE = 117_090


def arm_records():
    """Every raw record that reaches a dataset of the arm, with its unit type."""
    raw = pd.read_csv(empirical.SOURCE, low_memory=False)
    recs = empirical.load_records()
    labels, _ = categorysplit.assign(recs, empirical.category_tree())
    recs = recs.assign(dataset=labels)
    raw = raw.merge(recs[['open_xpd_uuid', 'dataset', 'name']],
                    on='open_xpd_uuid', how='left')
    return raw.dropna(subset=['dataset']).copy()


def build(df, mult=empirical.CLEAN_IQR_MULT, min_n=empirical.MIN_N):
    """{dataset: cleaned values}, the arm as `empirical.prepare` would build it."""
    out = {}
    for ds, g in df.groupby('dataset'):
        kept = clean_empirical_symmetric(g.ecc.to_numpy(float), mult)
        if len(kept) < min_n:
            continue
        out[ds] = kept
    return out


def characteristics(values, name):
    """Unweighted characteristics only, so no Dirichlet draw enters this table.

    Entry 32 records that a single weight realization moves the weighted
    per-dataset metrics a long way. A before-and-after comparison of a filter
    must not be read through that noise.
    """
    w = np.ones(len(values)) / len(values)
    m = empirical_metadata(values / np.mean(values), w)
    return {'dataset': name, 'n': len(values),
            **{k: m[k] for k in ('coeffvar_uw', 'skewness_uw', 'kurtosis_uw',
                                 'entropy_uw', 'fit_norm_SW_uw',
                                 'fit_lognorm_SW_uw', 'crit_bw_1_uw')}}


def surviving_records(df, mult=empirical.CLEAN_IQR_MULT, min_n=empirical.MIN_N):
    """The records that actually REACH the arm, i.e. survive cleaning.

    Reporting extremes from the RAW extract is misleading, and was. Two filters
    already stand between a raw record and the arm: the mass ceiling of decision
    49 and the symmetric log-space IQR rule of decision 33. A raw extreme is
    therefore not something the author has to rule on -- it is something that has
    already been ruled on. Only a record that survives BOTH is an open question.
    """
    # The ceiling runs BEFORE cleaning, exactly as `empirical.load_raw` does it:
    # a record wrong by three orders of magnitude should not be setting the
    # interquartile range the cleaning rule is computed from.
    df = df[~empirical.implausible(df)]
    keep = []
    for ds, g in df.groupby('dataset'):
        kept = clean_empirical_symmetric(g.ecc.to_numpy(float), mult)
        if len(kept) < min_n:
            continue
        lo, hi = float(np.min(kept)), float(np.max(kept))
        keep.append(g[(g.ecc >= lo) & (g.ecc <= hi)])
    return (pd.concat(keep, ignore_index=True) if keep
            else df.iloc[:0].copy())


def extremes(arm, k=10):
    """The k highest and k lowest SURVIVING records in each declared-unit type."""
    arm = surviving_records(arm)
    med = arm.groupby('dataset').ecc.transform('median')
    arm = arm.assign(ratio_to_category_median=arm.ecc / med)
    cols = ['du_type', 'end', 'dataset', 'name', 'declared_unit_raw', 'gwp_raw',
            'ecc', 'ecc_unit', 'ratio_to_category_median', 'open_xpd_uuid']
    rows = []
    for ut, g in arm.groupby('du_type'):
        rows.append(g.nlargest(k, 'ecc').assign(end='highest'))
        rows.append(g.nsmallest(k, 'ecc').assign(end='lowest'))
    out = pd.concat(rows, ignore_index=True)[cols]
    return out.sort_values(['du_type', 'end', 'ecc'],
                           ascending=[True, True, False])


def nonpositive_gwp():
    """Records the extraction drops for a non-positive ECC, i.e. GWP <= 0.

    A COUNT, not a change. Some biobased products are legitimately carbon
    negative, and an empirical arm truncated at zero by construction is worth a
    sentence in the paper beside the (0, inf) support decision (entry 13).

    Reproduces p1_build_raw_extract.load_store and to_ecc up to the final
    `ecc > 0` screen. Returns None if the store is not reachable, since the
    extract itself is frozen and this count is the only thing that needs it.
    """
    sys.path.insert(0, os.path.join(ROOT, 'audits'))
    try:
        import build_raw_extract as p1
    except Exception as exc:                                  # pragma: no cover
        print(f'  store not reachable ({exc}); count skipped')
        return None
    if not os.path.exists(p1.STORE):
        print(f'  store not present at {p1.STORE}; count skipped')
        return None
    df, _, _ = p1.load_store()
    du = df.declared_unit_raw.astype(str).apply(p1.str2valunit)
    gw = df.gwp_raw.astype(str).apply(p1.str2valunit)
    df = df.assign(du_qty=[a for a, _ in du], du_unit=[b for _, b in du],
                   gwp_qty=[a for a, _ in gw], gwp_unit=[b for _, b in gw])
    conv = [p1.convert(q, u) for q, u in zip(df.du_qty, df.du_unit)]
    df['du_value'] = [v for v, _ in conv]
    df['du_type'] = [t for _, t in conv]
    gconv = [p1.convert(q, u) for q, u in zip(df.gwp_qty, df.gwp_unit)]
    df['gwp_value'] = [v for v, _ in gconv]
    df['gwp_type'] = [t for _, t in gconv]

    rows = []
    for mat, g in df.groupby('material_query'):
        usable = g[(g.gwp_type == 'emissions') & g.du_type.notna()
                   & np.isfinite(g.du_value) & (g.du_value > 0)
                   & np.isfinite(g.gwp_value)]
        if usable.empty:
            continue
        modal = usable.du_type.value_counts().index[0]
        sel = usable[usable.du_type == modal].copy()
        sel['ecc'] = sel.gwp_value / sel.du_value
        sel = sel[np.isfinite(sel.ecc)]
        bad = sel[sel.ecc <= 0]
        if len(bad):
            rows.append(dict(material=mat, n_in_unit_type=len(sel),
                             n_nonpositive=len(bad),
                             n_zero=int((bad.ecc == 0).sum()),
                             n_negative=int((bad.ecc < 0).sum()),
                             most_negative=float(bad.ecc.min())))
    return pd.DataFrame(rows).sort_values('n_nonpositive', ascending=False)


def main():
    os.makedirs(TABLES, exist_ok=True)
    ceiling = empirical.MASS_ECC_CEILING
    arm = arm_records()
    mass = arm.du_type == empirical.MASS_UNIT_TYPE
    drop = mass & (arm.ecc > ceiling)

    print('=' * 72)
    print(f'1. THE GATE, ceiling {ceiling:g} kgCO2e/kg on mass-declared records')
    print('=' * 72)
    print(f'raw records reaching the arm        {len(arm):>8,}')
    print(f'  of them mass-declared             {int(mass.sum()):>8,} '
          f'in {arm.loc[mass, "dataset"].nunique()} datasets')
    print(f'records above the ceiling           {int(drop.sum()):>8,} '
          f'({drop.sum()/len(arm)*100:.4f}% of raw records)')

    before, after = build(arm), build(arm[~drop])
    nb = sum(len(v) for v in before.values())
    na = sum(len(v) for v in after.values())
    gate = GATE_FRACTION * ARM_VALUES_BEFORE
    print(f'CLEANED values                      {nb:>8,} -> {na:,} '
          f'({nb - na} removed, {(nb - na)/nb*100:.4f}%)')
    print(f'datasets                            {len(before):>8} -> {len(after)}')
    print(f'gate: stop if more than {gate:.0f} cleaned values '
          f'({GATE_FRACTION*100:g}% of {ARM_VALUES_BEFORE:,}) are removed')
    verdict = 'PASS' if (nb - na) <= gate else 'STOP AND REPORT'
    print(f'VERDICT: {verdict}')
    lost = sorted(set(before) - set(after))
    print(f'datasets lost entirely: {lost if lost else "none"}')

    print()
    print('=' * 72)
    print('2. WHAT WAS DROPPED, AND WHAT IT DID')
    print('=' * 72)
    dropped = (arm[drop].groupby('dataset')
               .agg(n_dropped=('ecc', 'size'), lowest_dropped=('ecc', 'min'),
                    highest_dropped=('ecc', 'max'))
               .sort_values('n_dropped', ascending=False))
    print(dropped.to_string())
    dropped.to_csv(os.path.join(TABLES, 'TABLE_PlausibilityDropped.csv'))

    rows = []
    for ds in sorted(before):
        b = characteristics(before[ds], ds)
        if len(after.get(ds, [])) == len(before[ds]):
            continue
        a = (characteristics(after[ds], ds) if ds in after
             else {k: np.nan for k in b})
        for k in b:
            if k == 'dataset':
                continue
            rows.append(dict(dataset=ds, characteristic=k, before=b[k],
                             after=a[k]))
    eff = pd.DataFrame(rows)
    eff['change'] = eff.after - eff.before
    eff.to_csv(os.path.join(TABLES, 'TABLE_PlausibilityEffect.csv'),
               index=False)
    print()
    print('characteristics of the datasets that lost records '
          '(unweighted, so no Dirichlet draw enters):')
    piv = eff.pivot(index='dataset', columns='characteristic',
                    values=['before', 'after'])
    for ds in piv.index:
        b = piv.loc[ds, 'before']
        a = piv.loc[ds, 'after']
        print(f'  {ds:<26} n {int(b["n"]):>5} -> {int(a["n"]):>5}   '
              f'cv {b["coeffvar_uw"]:8.4f} -> {a["coeffvar_uw"]:8.4f}   '
              f'skew {b["skewness_uw"]:9.3f} -> {a["skewness_uw"]:9.3f}   '
              f'kurt {b["kurtosis_uw"]:11.2f} -> {a["kurtosis_uw"]:10.2f}')

    print()
    print('=' * 72)
    print('3. EXTREMES BY DECLARED-UNIT TYPE, AFTER CLEANING.')
    print('   A report and not a filter. Records the cleaning already removed')
    print('   are not listed: they are not an open question.')
    print('=' * 72)
    ext = extremes(arm)
    ext.to_csv(os.path.join(TABLES, 'TABLE_UnitExtremes.csv'), index=False)
    for ut, g in ext.groupby('du_type'):
        print(f'--- {ut} ({int((arm.du_type == ut).sum()):,} records, '
              f'{arm.loc[arm.du_type == ut, "dataset"].nunique()} datasets, '
              f'unit {g.ecc_unit.iloc[0]}) ---')
        show = g[['end', 'dataset', 'name', 'declared_unit_raw', 'gwp_raw',
                  'ecc', 'ratio_to_category_median']].copy()
        show['name'] = show.name.astype(str).str.slice(0, 38)
        print(show.to_string(index=False, float_format=lambda v: f'{v:.4g}'))
        print()

    print('=' * 72)
    print('4. RECORDS EXCLUDED BY THE GWP > 0 SCREEN, a count and not a change')
    print('=' * 72)
    npg = nonpositive_gwp()
    if npg is not None:
        npg.to_csv(os.path.join(TABLES, 'TABLE_NonPositiveGWP.csv'),
                   index=False)
        print(f'records with a non-positive ECC: {int(npg.n_nonpositive.sum()):,} '
              f'across {len(npg)} categories '
              f'({int(npg.n_zero.sum())} exactly zero, '
              f'{int(npg.n_negative.sum())} negative)')
        print(npg.to_string(index=False))


if __name__ == '__main__':
    main()
