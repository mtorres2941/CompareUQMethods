"""Prepare the empirical EC3 ECC datasets for analysis.

The source is a frozen, dated raw extract under `data/raw/`: one row per EPD,
carrying the embodied carbon coefficient and enough provenance to audit it. It
is raw in the sense that matters here, that no outlier rule has ever been
applied to it, which is what lets the cleaning rule below treat both ends of the
distribution the same way.

Three choices are applied here, and each moves numbers.

1. Cleaning is multiplicative and symmetric. An ECC is strictly positive and
   right skewed, so an additive interquartile rule is the wrong shape: the low
   bound `Q1 - 3 * IQR` is negative in most categories and therefore never
   binds, which trims high outliers while leaving values many orders of
   magnitude below the mean in place. The rule is applied in log space instead,
   at both ends. See `datageneration.clean_empirical_symmetric`.

2. Point weights are drawn from a flat Dirichlet, alpha = 1, matching the
   synthetic arm. Market shares are unknown, and a flat Dirichlet is the
   maximum-entropy prior over them.

   A note on direction, because it is easy to state backwards. alpha is the
   Dirichlet CONCENTRATION parameter; a smaller alpha gives MORE dispersed
   market shares. It is still a lower bound on reality: at n = 100 a flat
   Dirichlet gives an expected largest share of 5.2 percent, while Marsh, Hattam
   and Allen (2025) report Rest-of-World BOF steel at 63.75 percent of global
   production.

3. Each dataset is divided by its own UNWEIGHTED mean, matching the synthetic
   path exactly. See decision 6 in CLAUDE.md.

4. A category that is not one product population is SPLIT into the populations
   its record metadata identifies, before cleaning. Stage 2a-3; see
   `src/categorysplit.py` for the screen, the axis and the constraint that a
   split may never read the ECC values. Pass `split=False` to recover the
   unsplit arm for comparison.

5. A record that cannot be a physical product is REMOVED, by an external
   ceiling and never by a distributional one. See `MASS_ECC_CEILING`. Stage 2b.

A population is kept only if at least `min_n` values survive cleaning. That
threshold, not the extraction, is what decides how many datasets the empirical
arm holds, so the count is reported rather than assumed.
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd

import categorysplit
from datageneration import clean_empirical_symmetric

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED = os.path.join(ROOT, 'data', 'processed')
RAW = os.path.join(ROOT, 'data', 'raw')

#: The frozen raw extract the analysis reads. A dated file, never overwritten.
SOURCE = os.path.join(RAW, 'ec3_raw_ecc_2026-08-14.csv.gz')

#: Whether the material categories are resolved into specifiable products:
#: EC3 residual bins dropped, concrete split by specified compressive strength,
#: insulation split by material type. See `src/categorysplit.py` for all three
#: rules and the constraint that they read only metadata, never the ECC values.
SPLIT = True

DIRICHLET_ALPHA = 1.0
CLEAN_IQR_MULT = 3.0
MIN_N = 3

#: Declared-unit type for which an external upper bound on the ECC exists.
MASS_UNIT_TYPE = 'weight'

#: Physical-plausibility ceiling on a MASS-declared ECC, in kgCO2e per kg of
#: product. Author decision, 2026-09-14.
#:
#: WHY THE BOUND IS EXTERNAL. This study measures the dispersion and modality of
#: ECC datasets. A ceiling read off the arm's own quantiles, standard deviations
#: or visible gaps would therefore be circular in exactly the way a
#: dispersion-based category split would have been; see decision 46 and
#: `categorysplit`. The bound below comes from material science and is applied
#: only where such a bound exists.
#:
#: WHAT IT IS ANCHORED ON. Published cradle-to-gate embodied-carbon inventories
#: for building products report coefficients of order 0.1 to 15 kgCO2e/kg, the
#: highest being primary aluminium, which the Inventory of Carbon and Energy
#: (ICE) database v3.0 (Jones and Hammond, Circular Ecology, 2019) places near
#: 13 kgCO2e/kg; EC3's own published ranges for its material categories sit
#: inside the same envelope. A stoichiometric check bounds the same quantity
#: from a different direction and needs no database at all: combusting pure
#: carbon yields 3.67 kg CO2 per kg of carbon, so 25 kgCO2e per kg of DELIVERED
#: PRODUCT already requires burning about 6.8 kg of pure carbon for each
#: kilogram shipped, and 100 kgCO2e/kg requires about 27 kg.
#:
#: The ceiling is set at 100 rather than at 25 deliberately. It is four times
#: the most generous defensible figure for a real product, so it cannot be
#: mistaken for a tuned threshold, and it still catches the known cases by two
#: orders of magnitude: two Elevators EPDs declared per kilogram report 20,812
#: and 21,945 kgCO2e/kg, and 87 Cement records report a per-tonne GWP against a
#: 1 kg declared unit, a factor-of-1,000 declaration error.
#:
#: Measured effect, `audits/plausibility_ceiling.py`: 115 of 117,807 raw
#: records, 11 of 117,090 CLEANED values, and no dataset lost.
MASS_ECC_CEILING = 100.0


#: The frozen record metadata the split rules read: the product name and
#: description, the declared concrete strength, and the EC3 category path. The
#: ECC values themselves always come from SOURCE, never from here; this file is
#: joined on `open_xpd_uuid` and carries no ECC the analysis uses.
METADATA = os.path.join(RAW, 'ec3_record_metadata_2026-08-14.csv.gz')

#: The frozen EC3 category hierarchy. Its parent/child relation is what
#: identifies a category that is a residual bin rather than a product.
CATEGORY_TREE = os.path.join(RAW, 'ec3_category_tree_2026-08-14.csv')

SPLIT_COLS = ['open_xpd_uuid', 'material_query', 'ecc', 'du_type']
META_COLS = ['open_xpd_uuid', 'name', 'description',
             'concrete_compressive_strength_28d_value']


#: The frozen extract stores length-declared ECC per INCH, because that was the
#: canonical length unit when it was built. It is metres now
#: (`funcs_unit_conversion.length2m`), and the extract cannot be rebuilt: the
#: EC3 store it came from has moved on, so a rebuild would change the
#: POPULATION and not just the unit, which decision 44 refuses. The rows are
#: therefore rescaled on the way in.
#:
#: This moves no analysis number. Every dataset is normalized by its own
#: unweighted mean and cleaned by a log-space rule, and both are invariant
#: under a constant rescale of a whole dataset. It changes the magnitude and the
#: label of the raw ECC, which is what a reader looks at.
#: Inches per metre. Keyed on `du_type`, which the extract carries, rather
#: than on `ecc_unit`, which it does not load.
LENGTH_UNIT_FIX = {'du_type': 'length', 'factor': 39.3701}


def fix_length_unit(df, spec=LENGTH_UNIT_FIX):
    """Rescale length-declared ECC from per-inch to per-metre, in place of a rebuild."""
    if 'du_type' not in df.columns or 'ecc' not in df.columns:
        return df
    sel = df.du_type == spec['du_type']
    if not sel.any():
        return df
    df = df.copy()
    df.loc[sel, 'ecc'] = df.loc[sel, 'ecc'] * spec['factor']
    if 'ecc_unit' in df.columns:
        df.loc[sel, 'ecc_unit'] = 'kgco2e/m'
    return df


def load_records(path=SOURCE, metadata=METADATA):
    """The raw extract, one row per EPD, uncleaned, with the split metadata."""
    df = pd.read_csv(path, usecols=SPLIT_COLS, low_memory=False)
    df = fix_length_unit(df)
    meta = pd.read_csv(metadata, usecols=META_COLS, low_memory=False)
    out = df.merge(meta, on='open_xpd_uuid', how='left', validate='one_to_one')
    if len(out) != len(df):
        raise ValueError('metadata join changed the record count')
    return out


def category_tree(path=CATEGORY_TREE):
    return pd.read_csv(path)


def split_report(path=SOURCE):
    """One row per resulting population, with the rule, field and reason."""
    return categorysplit.assign(load_records(path), category_tree())[1]


#: EC3's own published GWP per KILOGRAM, frozen for the extract's records. This
#: is what makes the consistency check below possible: it is a second,
#: independently published number on the same EPD.
GWP_PER_KG = os.path.join(RAW, 'ec3_gwp_per_kg_2026-09-16.csv.gz')

#: The range in which EC3's gwp_per_kg is itself believable, in kgCO2e/kg. A
#: field of 0.0 or 1e-9 is a broken field, not a carbon-free product, and
#: dividing by it manufactures an absurd implied mass from a perfectly good
#: record. Four ReadyMix records at a wholly normal 372 to 451 kgCO2e/m3 were
#: flagged that way before this guard existed.
GWP_PER_KG_USABLE = (0.001, 100.0)

#: Largest mass, in kg, that one declared unit of a building product can
#: physically have. Not tuned: a cubic metre denser than 12,000 kg is denser
#: than lead, a square metre massing 8,000 kg is a metre of solid steel, and no
#: linear building element runs to 1,000 kg per metre.
MAX_MASS_PER_UNIT = {'vol': 12000.0, 'area': 8000.0, 'length': 1000.0}

#: Companion ceiling on the ECC itself, per declared-unit type. BOTH must be
#: exceeded before a record is removed, and that is the whole design: two
#: independently published numbers on one record must BOTH be impossible. The
#: mass test alone would discard good records whose gwp_per_kg is broken; the
#: ECC test alone is the per-unit ceiling that cannot be anchored externally,
#: since a square metre of product has no general size. Together they identify a
#: DECLARED-UNIT error, which is what these are.
MAX_ECC_PER_UNIT = {'vol': 5000.0, 'area': 5000.0, 'length': 1000.0}


def gwp_per_kg(path=GWP_PER_KG):
    """EC3's published kgCO2e per kg, indexed by record."""
    return pd.read_csv(path).set_index('open_xpd_uuid').gwp_per_kg_value


def internally_inconsistent(df, path=GWP_PER_KG):
    """Records whose declared unit contradicts their own published mass figure.

    A record declared per cubic metre, square metre or metre carries a second
    number EC3 publishes independently: the GWP per kilogram. Dividing one by
    the other gives the mass that one declared unit would have to have. When
    that mass is physically impossible AND the ECC itself is beyond what the
    unit can support, the DECLARED UNIT is wrong.

    This is not a dispersion screen. It never looks at the other records, only
    at two numbers on the same EPD and at physical limits fixed outside the
    data. `THHN/THWN-2 High Speed (HS)` declares 14,300 kgCO2e for one metre of
    ordinary building wire while publishing 4.017 kgCO2e/kg, which puts 3,560 kg
    of copper in a single metre.

    Records with no usable gwp_per_kg are simply not checked; the file covers
    96.5 percent of the extract.
    """
    if 'du_type' not in df.columns or 'ecc' not in df.columns:
        return pd.Series(False, index=df.index)
    per_kg = df.open_xpd_uuid.map(gwp_per_kg(path))
    lo, hi = GWP_PER_KG_USABLE
    usable = per_kg.between(lo, hi)
    implied = df.ecc / per_kg.where(usable)
    mass_limit = df.du_type.map(MAX_MASS_PER_UNIT)
    ecc_limit = df.du_type.map(MAX_ECC_PER_UNIT)
    return ((implied > mass_limit) & (df.ecc > ecc_limit)).fillna(False)


def implausible(df, ceiling=MASS_ECC_CEILING, unit_type=MASS_UNIT_TYPE):
    """Records that cannot be a physical product, as a boolean mask.

    Applied only to the declared-unit type for which an external bound exists.
    For volume, area, length and item declarations there is no comparably tight
    published bound, so none is invented: those extremes are REPORTED for author
    review instead, by `audits/plausibility_ceiling.py`, and left in the arm.

    See `MASS_ECC_CEILING` for where the number comes from and why it may not be
    read off the data.
    """
    return (df.du_type == unit_type) & (df.ecc > ceiling)


def labelled_records(path=SOURCE, split=SPLIT):
    """Every raw record with the dataset it belongs to and whether it is kept.

    `dataset` is NaN for a record the split rules discard, and `implausible`
    marks a record the external ceiling removes. Nothing is filtered here, so
    both decisions stay visible to anything that wants to audit them.
    """
    df = load_records(path)
    if split:
        labels, _ = categorysplit.assign(df, category_tree())
        df = df.assign(dataset=labels)
    else:
        df = df.assign(dataset=df.material_query)
    return df.assign(implausible=implausible(df))


def implausible_report(path=SOURCE, split=SPLIT):
    """One row per dataset that loses records to the ceiling, with what it lost."""
    df = labelled_records(path, split)
    df = df[df.dataset.notna() & df.implausible]
    if df.empty:
        return pd.DataFrame(columns=['dataset', 'n_implausible',
                                     'lowest_dropped', 'highest_dropped'])
    return (df.groupby('dataset')
            .agg(n_implausible=('ecc', 'size'), lowest_dropped=('ecc', 'min'),
                 highest_dropped=('ecc', 'max'))
            .reset_index().sort_values('n_implausible', ascending=False))


def load_raw(path=SOURCE, split=SPLIT, ceiling=True):
    """The raw ECC values per dataset, uncleaned. Returns (values, split_report).

    With `split=True` the categories are resolved into specifiable products:
    EC3 residual bins are dropped, concrete is split by specified compressive
    strength and insulation by material type. See `src/categorysplit.py`.

    With `ceiling=True` the physically implausible mass-declared records are
    removed first; see `MASS_ECC_CEILING`. It runs BEFORE cleaning, because a
    record that is a declaration error by three orders of magnitude should not
    be setting the interquartile range that the cleaning rule is computed from.
    Pass `ceiling=False` to recover the arm as it stood at the end of Stage 2a-3.
    """
    df = load_records(path)
    if not split:
        if ceiling:
            df = df[~implausible(df)]
            df = df[~internally_inconsistent(df)]
        return ({mat: g.ecc.to_numpy(float)
                 for mat, g in df.groupby('material_query', sort=True)},
                pd.DataFrame())
    labels, report = categorysplit.assign(df, category_tree())
    df = df.assign(dataset=labels).dropna(subset=['dataset'])
    if ceiling:
        df = df[~implausible(df)]
        df = df[~internally_inconsistent(df)]
    return ({ds: g.ecc.to_numpy(float)
             for ds, g in df.groupby('dataset', sort=True)}, report)


def source_meta(path=SOURCE):
    with open(path.replace('.csv.gz', '_runmeta.json')) as f:
        return json.load(f)


def _dataset_rng(base, name):
    """A stream for one dataset, keyed by its name and the caller's entropy.

    `base` is drawn once from the Generator the caller passed, so the whole arm
    still moves with the notebook's seed and nothing here touches global numpy
    state. The dataset name is folded in through SHA-256 so that two datasets
    get independent streams and a given dataset gets the same stream whatever
    else the arm contains.
    """
    digest = hashlib.sha256(name.encode('utf-8')).digest()[:16]
    return np.random.default_rng(np.random.SeedSequence(
        [base, int.from_bytes(digest, 'big')]))


def prepare(rng, path=SOURCE, alpha=DIRICHLET_ALPHA, mult=CLEAN_IQR_MULT,
            min_n=MIN_N, split=SPLIT, ceiling=True):
    """Clean, weight and normalize. Returns (datasets, report).

    Pass a DEDICATED generator, `rng.spawn(1)[0]`, not the notebook's shared
    one. The weights drawn here would otherwise depend on how many random
    values had been consumed before this function was called, so adding a
    figure earlier in the notebook would silently change every empirical
    metric. A spawned generator depends only on how many times `spawn` has been
    called, which makes the empirical datasets reproducible independently of
    what else the notebook does.

    Each dataset's weights come from its OWN stream, keyed by its name rather
    than by its position in the iteration. This is the same argument one level
    down: if the streams were taken in order, adding or splitting a single
    category would change the weights of every dataset after it alphabetically,
    and the resulting movement in every weighted metric would be attributed to
    the change under study. Keying by name makes a dataset's weights a property
    of that dataset. The randomness still traces to `rng`, which supplies the
    base entropy; see `_dataset_rng`.

    `ceiling=True` removes the physically implausible mass-declared records
    before cleaning; see `MASS_ECC_CEILING`. `n_before` in the report is the
    count AFTER that removal, and `n_implausible` says how many it took.

    `datasets[mat]` is (values, weights) with values divided by their
    unweighted mean.
    """
    raw, _ = load_raw(path, split=split, ceiling=ceiling)
    dropped = (implausible_report(path, split).set_index('dataset').n_implausible
               if ceiling else pd.Series(dtype=int))
    base = int(rng.integers(0, 2 ** 63))
    out, report = {}, []
    for mat in sorted(raw):
        data = raw[mat]
        kept = clean_empirical_symmetric(data, mult)
        removed = len(data) - len(kept)
        row = dict(material=mat, n_implausible=int(dropped.get(mat, 0)),
                   n_before=len(data), n_after=len(kept),
                   n_removed=removed,
                   n_removed_low=int((data < kept.min()).sum()) if len(kept) else 0,
                   n_removed_high=int((data > kept.max()).sum()) if len(kept) else 0,
                   min_over_mean_before=float(data.min() / data.mean()),
                   max_over_mean_before=float(data.max() / data.mean()),
                   min_over_mean_after=(float(kept.min() / kept.mean())
                                        if len(kept) else np.nan),
                   max_over_mean_after=(float(kept.max() / kept.mean())
                                        if len(kept) else np.nan))
        if len(kept) < min_n:
            row['status'] = 'dropped_too_few_values'
            report.append(row)
            continue
        row['status'] = 'ok'
        report.append(row)
        w = _dataset_rng(base, mat).dirichlet(np.ones(len(kept)) * alpha)
        out[mat] = (kept / np.mean(kept), w)
    return out, report


def write(datasets, report, label, out_root=PROCESSED, meta_extra=None):
    """Write a prepared empirical arm to a new, dated file. Never overwrites."""
    path = os.path.join(out_root, f'empirical_{label}.json')
    if os.path.exists(path):
        raise FileExistsError(f'{path} already exists; pick a new label.')
    payload = {mat: {'data': v.tolist(), 'weights': w.tolist()}
               for mat, (v, w) in datasets.items()}
    with open(path, 'w') as f:
        json.dump(payload, f)
    meta = dict(
        label=label, source=os.path.relpath(SOURCE, ROOT),
        source_meta=source_meta(),
        dirichlet_alpha=DIRICHLET_ALPHA, clean_iqr_mult=CLEAN_IQR_MULT,
        clean_rule='multiplicative log-space IQR, both ends',
        mass_ecc_ceiling=MASS_ECC_CEILING,
        mass_unit_type=MASS_UNIT_TYPE,
        min_n=MIN_N,
        n_materials=len(datasets),
        n_values_before=int(sum(r['n_before'] for r in report)),
        n_values_after=int(sum(r['n_after'] for r in report)),
        n_values_removed=int(sum(r['n_removed'] for r in report)),
        n_values_implausible=int(sum(r.get('n_implausible', 0) for r in report)),
        n_materials_dropped=int(sum(r['status'] != 'ok' for r in report)),
        report=report,
    )
    if meta_extra:
        meta.update(meta_extra)
    with open(os.path.join(out_root, f'empirical_{label}_runmeta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return path, meta


def load(label, out_root=PROCESSED):
    with open(os.path.join(out_root, f'empirical_{label}.json')) as f:
        d = json.load(f)
    return {mat: (np.asarray(v['data'], float), np.asarray(v['weights'], float))
            for mat, v in d.items()}
