"""Split EC3 categories that are not one product population, on metadata only.

SUSPENDED, 2026-09-14. THE DECLARED-UNIT AXIS BELOW WAS REJECTED BY THE AUTHOR
AND IS NOT APPLIED. `empirical.SPLIT` is False, so the arm is the 136 unsplit
categories. The module is kept, not deleted, because the screen, the axis survey
and the audits behind them are the evidence for whatever replaces it.

Why it was rejected, in the author's words: "Separating by declared unit doesn't
quite seem reasonable. Why is it strange that some aggregates might be declared
per 1 kg and some per 1000 kg? That still might be the same material. That's
definitely not enough information to say it's something different."

That is correct, and this module's own evidence already said so. The declared
unit is a DECLARATION CONVENTION, not a product property. Where the split
appeared to work it was because the convention happened to CORRELATE with
contamination -- Aggregates declared per kilogram are mostly adhesives and
screeds, Chairs per tonne are mostly asphalt -- and a correlate is not a
criterion. The one category where the correlation failed, ConcreteAdmixtures,
was flagged in the Stage 2a-3 handoff as "the weakest of the six" and kept
anyway, on the reasoning that the rule was applied uniformly. A uniform rule on
the wrong axis is still the wrong axis.

What the axis survey found, and it is the input to whatever replaces this:
see `reports/HANDOFF_stage-2a3.md` section 3.1a.

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
  C. Declared unit SCALE, below, measured as a RATIO to the way most of the
     category declares itself. This is the axis that does work.

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


def scale_band(du_value, reference):
    """The declared-unit scale band, as a RATIO to the category's reference.

    `du_value` is the declared quantity already converted to the unit type's
    canonical unit by `funcs_unit_conversion`. `reference` is the modal declared
    quantity in the category, so band 0 always holds the way most of the
    category declares itself and the other bands are powers of a thousand away
    from it.

    The band must be a ratio and not an absolute position, and the first version
    of this function got that wrong. It banded `log10(du_value)` directly, and
    the canonical unit for length is the INCH: 0.65 m is 25.6 in and 1 m is
    39.4 in, which straddle a decade boundary, so two cable declarations a
    factor of 1.5 apart were assigned to different populations. The rule
    contradicted its own stated justification, which is that a split separates
    functional units at least three orders of magnitude apart. Against a
    reference the boundaries are ratios and the justification holds by
    construction.
    """
    ratio = np.asarray(du_value, float) / float(reference)
    dec = np.round(np.log10(ratio)).astype(int)
    return np.floor((dec + 1) / BAND_DECADES).astype(int)


def _reference(du_value):
    """The modal declared quantity in a category: how most of it declares."""
    return pd.Series(np.asarray(du_value, float)).value_counts().index[0]


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
        ref = _reference(g.du_value.to_numpy())
        bands = pd.Series(scale_band(g.du_value.to_numpy(), ref))
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
        ref = _reference(g.du_value.to_numpy())
        bands = pd.Series(scale_band(g.du_value.to_numpy(), ref), index=g.index)
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
        # The sentence is written against the LARGEST population, so every
        # population of a category carries the same comparison and a reader is
        # not asked to hold three pairwise statements in mind.
        main = max(viable, key=lambda b: counts[b])
        breakdown = ', '.join(f'{counts[b]} per {units[b]}' for b in viable)
        for b in viable:
            sel = g.index[bands == b]
            labels.loc[sel] = f'{cat} [{units[b]}]'
            if b == main:
                tail = (f'This is how most of the category declares itself and '
                        f'is kept as its main population.')
            else:
                tail = (f'A declaration per {units[b]} is a different '
                        f'functional unit from one per {units[main]}, so the '
                        f'two are treated as separate populations rather than '
                        f'pooled.')
            report.append(dict(
                material=cat, field='declared_unit_raw', band=int(b),
                population=f'{cat} [{units[b]}]', n=int(counts[b]),
                cv=row.cv, split=True,
                justification=(
                    f'{cat} declarations state the functional unit at scales '
                    f'differing by at least {BAND_DECADES} orders of magnitude '
                    f'from the way most of the category declares itself, per '
                    f'{units[main]}: {breakdown}. {tail}')))
        # Records in bands too small to form a dataset are dropped, which is
        # the rule the arm already applies to a category with under MIN_N
        # values. They are reported so the count is derivable.
        for b in sorted(counts[counts < MIN_N].index):
            sel = g.index[bands == b]
            labels.loc[sel] = None
            u = _unit_label(g.declared_unit_raw[bands == b].value_counts().index[0])
            report.append(dict(
                material=cat, field='declared_unit_raw', band=int(b),
                population=f'{cat} [{u}] DROPPED', n=int(counts[b]), cv=row.cv,
                split=True,
                justification=(
                    f'Dropped: {counts[b]} declaration(s) per {u}, fewer than '
                    f'the {MIN_N} a dataset needs. This is the same threshold '
                    f'that drops a whole category with under {MIN_N} values.')))
    return labels, pd.DataFrame(report)
