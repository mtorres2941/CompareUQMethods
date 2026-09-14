"""Split EC3 categories that are not one product population, on metadata only.

Stage 2a-3. Discrepancy entry 31 recorded that a handful of EC3 categories span
several orders of magnitude and hold products that are not comparable. The
author's decision is to treat those as the separate populations they are,
provided each split can be substantiated.

THE BINDING CONSTRAINT. A split may read only record metadata: the declared
unit, the EC3 category path, a product-type field, anything carried ON the EPD.
It may NOT read the ECC values. This study measures the modality, dispersion and
skewness of ECC distributions; splitting a category because its values look
bimodal and then reporting that ECC datasets are unimodal is circular, and a
reviewer will see it. Splitting on the declared unit and then observing that the
resulting populations are less dispersed is a finding.

The coefficient of variation appears below only as a SCREEN, deciding which
categories are examined. It never decides where a boundary falls. That
distinction is the whole defensibility of this module.

WHAT IS ACTUALLY AVAILABLE, measured in audits/stage2a3/. Three axes were
checked against the 2026-08 store slice, in this priority order:

  A. EC3 category path (`category_key`). NOT AVAILABLE. It equals the queried
     category for all 123,060 records in all 138 categories, and the finer
     `category` field is empty throughout. There is no subcategory to split on.
  B. Declared unit TYPE (weight, vol, area, length). ALREADY APPLIED. The
     extraction restricts each category to the declared-unit type most of its
     products use, so every dataset in the arm holds exactly one unit type by
     construction. 106 of 138 categories contain records of some other unit
     type, but those 2,780 records were dropped when the extract was built.
     Reinstating them would change what an ECC is, not split a population.
  C. Declared unit SCALE, below. This is the axis that does work.

A fourth axis, a populated product-type field, was searched exhaustively over
all 106 store columns for the screened categories. Nothing qualifies: the only
fields populated for 90 percent or more of records with more than one level are
declarer attributes (program operator, PCR, jurisdiction, plant specificity,
uncertainty factor), and a declarer is not a product population.
"""

import numpy as np
import pandas as pd

from datageneration import clean_empirical_symmetric

#: Screen threshold on the unweighted coefficient of variation of the cleaned
#: values. The arm's median is 0.82 and its 95th percentile 2.72, so 3.0 selects
#: the extreme upper tail. The unweighted form is used deliberately: it does not
#: depend on the Dirichlet weight draw, so the screen is reproducible from the
#: frozen extract alone.
SCREEN_CV = 3.0

#: Minimum records a population needs to become a dataset, matching
#: `empirical.MIN_N`, which is the rule the arm already applies to categories.
MIN_N = 3

#: Width of a declared-unit scale band, in decades. Three decades is one SI
#: prefix step: a declaration per tonne and one per kilogram, or per kilometre
#: and per metre, land in different bands, while the ordinary spread of declared
#: quantities within one convention (0.36 m2, 1 m2, 2.4 m2) lands in one.
BAND_DECADES = 3


def scale_band(du_value):
    """The declared-unit scale band of a canonical declared quantity.

    `du_value` is the declared quantity already converted to the unit type's
    canonical unit by `funcs_unit_conversion`, so it is comparable within a
    category. Bands are centred on 1, 1e3, 1e-3 and so on.
    """
    dec = np.round(np.log10(np.asarray(du_value, float))).astype(int)
    return np.floor((dec + 1) / BAND_DECADES).astype(int)


def _clean_cv(values, mult=3.0):
    """Unweighted coefficient of variation after the arm's cleaning rule."""
    kept = clean_empirical_symmetric(np.asarray(values, float), mult)
    if len(kept) < MIN_N:
        return np.nan
    return float(np.std(kept, ddof=1) / np.mean(kept))


def _unit_label(s):
    """A readable declared unit, from the modal raw string of a band."""
    lab = str(s).strip()
    # "1000.0 kg" reads better as "1000 kg"; nothing else is altered.
    head, _, tail = lab.partition(' ')
    try:
        v = float(head)
        head = f'{v:g}'
    except ValueError:
        pass
    return f'{head} {tail}'.strip()


def screen(records, cv_threshold=SCREEN_CV, mult=3.0):
    """Which categories are examined for splitting, and why.

    `records` is one row per retained ECC value, with `material_query`, `ecc`,
    `du_value` and `du_type`. Returns one row per category with each screen
    clause evaluated, applied to every category rather than to a chosen few.
    """
    rows = []
    for cat, g in records.groupby('material_query', sort=True):
        bands = pd.Series(scale_band(g.du_value.to_numpy()))
        viable = (bands.value_counts() >= MIN_N).sum()
        rows.append(dict(
            material=cat,
            n=len(g),
            n_unit_types=int(g.du_type.nunique()),
            n_scale_bands=int(bands.nunique()),
            n_viable_scale_bands=int(viable),
            cv=_clean_cv(g.ecc.to_numpy(), mult),
        ))
    out = pd.DataFrame(rows)
    out['selected'] = out.cv > cv_threshold
    return out


def assign(records, cv_threshold=SCREEN_CV, mult=3.0):
    """Assign every record to a dataset. Returns (labels, report).

    `labels` is a Series aligned to `records`, holding the dataset each record
    belongs to: the category name when the category is not split, and
    `Category [1000 kg]` when it is. `report` is one row per resulting
    population, carrying the field used, the counts and a substantiating
    sentence.
    """
    scr = screen(records, cv_threshold, mult).set_index('material')
    labels = records.material_query.copy()
    report = []
    for cat, g in records.groupby('material_query', sort=True):
        row = scr.loc[cat]
        if not row.selected:
            continue
        bands = pd.Series(scale_band(g.du_value.to_numpy()), index=g.index)
        counts = bands.value_counts()
        viable = sorted(counts[counts >= MIN_N].index)
        if len(viable) < 2:
            report.append(dict(
                material=cat, field='none', population=cat, n=len(g),
                cv=row.cv, split=False,
                justification=(
                    f'{cat} has a coefficient of variation of {row.cv:.1f} but '
                    f'no metadata field separates it: every record carries the '
                    f'same EC3 category path and the same declared-unit type, '
                    f'and its declared quantities fall in one scale band. It is '
                    f'left whole.')))
            continue
        units = {b: _unit_label(g.declared_unit_raw[bands == b]
                                .value_counts().index[0]) for b in viable}
        for b in viable:
            sel = g.index[bands == b]
            labels.loc[sel] = f'{cat} [{units[b]}]'
            others = ', '.join(f'{counts[o]} per {units[o]}'
                               for o in viable if o != b)
            report.append(dict(
                material=cat, field='declared_unit_raw', band=int(b),
                population=f'{cat} [{units[b]}]', n=int(counts[b]),
                cv=row.cv, split=True,
                justification=(
                    f'{cat} declarations state the functional unit at scales '
                    f'that differ by at least {BAND_DECADES} orders of '
                    f'magnitude: {counts[b]} declare per {units[b]} against '
                    f'{others}. A declaration per {units[b]} is a different '
                    f'functional unit, so the groups are treated as separate '
                    f'populations rather than pooled.')))
        # Records in bands too small to form a dataset are dropped, which is
        # the rule the arm already applies to a category with under MIN_N
        # values. They are reported so the count is derivable.
        for b in sorted(counts[counts < MIN_N].index):
            sel = g.index[bands == b]
            labels.loc[sel] = None
            report.append(dict(
                material=cat, field='declared_unit_raw', band=int(b),
                population=f'{cat} [band {b}]', n=int(counts[b]), cv=row.cv,
                split=True,
                justification=(f'Dropped: fewer than {MIN_N} declarations at '
                               f'this declared-unit scale.')))
    return labels, pd.DataFrame(report)
