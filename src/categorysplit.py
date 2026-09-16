"""Make the empirical material categories correspond to specifiable products.

Some EC3 categories are not one product. `Insulation` pools a 110 g/m2 silicate
coating with a 320 mm stone wool board; `Steel` pools whatever did not fall into
one of its seven children. A dataset in this study stands for one material
choice in a probabilistic LCA, so a category that is not a product a specifier
could name is not a valid unit of analysis.

THE QUESTION IS NOT DISPERSION. An earlier version of this module screened on the
coefficient of variation and split on the declared unit, and both were wrong.
Dispersion is a symptom, and a weak one: splitting `ReadyMix` by compressive
strength moves its coefficient of variation only from 0.29 to 0.27, and it is
still obviously the right split, because 4000 psi and 5000 psi concrete are
different products and compressive strength is the primary characteristic a
structural engineer specifies concrete by. The test is whether the resulting
groups are things somebody would specify, not whether they are tight.

THE BINDING CONSTRAINT IS UNCHANGED. A split may read only metadata carried on
the EPD record or on EC3's category tree, never the ECC values. This study
measures the modality, dispersion and skewness of ECC distributions, so
splitting a category because its values look bimodal and then reporting on
modality would be circular, and a reviewer will see it.

Three rules, each settled with the author on 2026-09-14 and each reading only
metadata.

RULE 1, DROP THE RESIDUAL BINS. EC3's category tree is a hierarchy. A category
that is a NON-LEAF node holds the EPDs that EC3 did not place in any of its
children, so it is a residual bin by construction rather than a product. Where
those children are themselves categories in this arm, the products are already
represented and the bin is dropped.

    This is what answers insulation and steel at once. `Insulation` is a parent
    whose four children -- BlanketInsulation, BlownInsulation, BoardInsulation
    and FoamedInPlace -- are all in the arm with coefficients of variation
    between 1.05 and 1.19, while the parent itself is at 7.65. `Steel` is a
    parent whose children in the arm are exactly the distinctions a structural
    engineer draws: Hollow, HotRolled, PlateSteel, RebarSteel, ColdFormedSteel,
    DeckingSteel, WireMeshSteel, SteelSuspensionAssembly, Coil.

    A parent whose children are NOT in the arm is kept, because dropping it
    would remove the material entirely. Four are in that position.

RULE 2, SPLIT CONCRETE BY SPECIFIED COMPRESSIVE STRENGTH. `concrete_compressive
_strength_28d` is a structured EC3 field, populated on 90 to 96 percent of
records in the concrete categories, and its values cluster on the standard
classes. Applied to the structural building concretes only; see CONCRETE below.

RULE 3, SPLIT INSULATION BY MATERIAL TYPE. Mineral wool, EPS, XPS, PIR/PUR and
so on are different products with different emissions, and the type is not a
field: it is matched against the product name and description with the fixed,
declared pattern list in MATERIAL_TYPES. Records naming no material become an
explicit "type not stated" dataset rather than being dropped or hidden.

    Thickness was tested as a fourth rule and REJECTED for the children. It is
    the axis that explains the parent bin's dispersion -- every parsed thickness
    band lands between 0.60 and 1.56 against the bin's 7.65 -- but the parent is
    dropped by rule 1, and on the children a thickness is parseable for only 96
    of 335 BoardInsulation and 150 of 319 BlanketInsulation records, so crossing
    it with type leaves 8 viable groups covering 73 of 335. "As we are able" is
    not able here. See audits/split_axis_evidence.py.
"""

import re

import numpy as np
import pandas as pd

#: Minimum records a population needs to become a dataset, matching
#: `empirical.MIN_N`, which is the rule the arm already applies to categories.
MIN_N = 3

#: Concrete categories split by specified 28-day compressive strength.
#:
#: These are the structural building concretes, the ones a design actually
#: specifies by strength. `CementGrout`, `FlowableFill` and `OilPatch` carry the
#: field just as well but are not specified this way in building design, and the
#: author's instruction was not to touch the rarely specified categories. To
#: change the scope, change this tuple: nothing else depends on its contents.
CONCRETE = ('ReadyMix', 'Shotcrete', 'ConcretePaving', 'CMU')

#: Strength classes, in psi, which is the unit EC3 stores. The declared values
#: cluster on the standard classes 3000, 3500, 4000, 4500, 5000 and 6000, so
#: these edges cut between them rather than through them.
PSI_EDGES = (0, 3000, 4000, 5000, 6000, np.inf)
PSI_LABELS = ('<3000 psi', '3000-3999 psi', '4000-4999 psi', '5000-5999 psi',
              '>=6000 psi')

#: Insulation categories split by material type.
INSULATION = ('BoardInsulation', 'BlanketInsulation', 'BlownInsulation',
              'FoamedInPlace')

#: Material type patterns, matched case-insensitively against the product name
#: and description. ORDER MATTERS: the first pattern that matches wins, so a
#: record naming two materials is assigned to the one listed first. The list is
#: fixed and declared here rather than derived from the data, so that it cannot
#: be tuned against the ECC values.
MATERIAL_TYPES = (
    ('mineral wool', r'mineral wool|rock ?wool|stone ?wool|glass ?wool'
                     r'|glasswool|rockwool|laine de verre|laine de roche'
                     r'|steinwolle|glaswolle|lana de roca|lana de vidrio'),
    ('XPS', r'\bxps\b|extruded polystyrene|polystyr[eè]ne extrud'),
    ('EPS', r'\beps\b|expanded polystyrene|polystyr[eè]ne expans|styropor'),
    ('PIR or PUR', r'\bpir\b|\bpur\b|polyiso|polyurethan'),
    ('cellulose', r'cellulose|ouate de cellulose'),
    ('wood fibre', r'wood ?fib|fibre de bois|holzfaser'),
    ('other', r'aerogel|phenolic|perlite|vermiculite|cork|\bhemp\b|chanvre'),
)

#: Label for records whose material the name and description do not state. Kept
#: as a dataset rather than dropped: it is 131 of 335 BoardInsulation records
#: and discarding it would lose 40 percent of the category.
UNSTATED = 'type not stated'


#: Rule 4. Categories that EC3 names for a product but that hold a mixture of
#: unrelated products, so they are not one population and cannot be made into
#: one by any split. DROPPED whole.
#:
#: This is decision 46 rule 1 applied to the evidence rather than to the tree:
#: a residual bin is recognisable from EC3's category structure, and these are
#: not residual bins -- they are leaf categories into which EC3 has filed
#: unrelated EPDs. The test is the PRODUCT NAME, never the ECC value, so it is
#: the same kind of evidence as the insulation split and carries the same
#: constraint from decision 43.
#:
#: The proportions below were measured by matching each record's name against
#: the material the category claims. They are recorded because "most of this
#: category is not the material" is the finding, and a later reader should be
#: able to see how far from one product these were.
NOT_ONE_POPULATION = {
    'Chairs': ('only 15 of 86 records name a chair, stool, bench, sofa or other '
               'seating. The rest are kitchen mixer taps, asphalt, culverts, '
               'hollowcore slabs, particle board and bathroom furniture'),
    'Grouting': ('only 44 of 225 records name a grout or a jointing compound. '
                 'The rest are gypsum plasters, decorative renders, ground '
                 'granulated blast furnace slag, concrete admixtures, epoxy '
                 'coatings, a cable clamp and a glazed door'),
}

#: Rule 5. Individual records that name a product the category is not, in
#: categories that are otherwise coherent. An EXCLUSION list and not an
#: inclusion vocabulary, deliberately: an inclusion rule has to anticipate every
#: legitimate naming convention in every language, and when it was tried it
#: removed 22 PowerCabling records that are plainly cables (Cable a Haute
#: Tension, TSLF 24kV, NF C 33-226, Nexans U-1000 R2V, H07RN-F) and two
#: Schindler elevators whose names are model numbers, while MISSING the real
#: intruders, because GRANITEK Sinks and Composite granite kitchen sinks both
#: match on "granite". An exclusion list only removes what it explicitly names.
#:
#: PowerCabling and Elevators were checked and need no entry: all 381 cable
#: names are cables or conductors and all 20 elevator names are elevators.
EXCLUDED_PRODUCTS = {
    'Aggregates': (r'\bsink|washbasin|wash basin|lavabo|jack module|rj45|'
                   r'porcelain stoneware|worktop|countertop|sanitary',
                   'sinks, washbasins and porcelain stoneware slabs are '
                   'finished products, not the crushed stone, gravel and sand '
                   'the category names'),
}


def residual_bins(categories, tree):
    """Categories that are EC3 parent nodes whose children are in the arm.

    Returns (drop, keep): `drop` maps a category to the children that represent
    it, `keep` maps a parent with no child in the arm to an empty list, because
    dropping it would remove the material from the study entirely.
    """
    present = set(categories)
    drop, keep = {}, {}
    for _, row in tree[~tree.is_leaf].iterrows():
        if row['name'] not in present:
            continue
        children = sorted(c for c in tree[tree.parent_name == row['name']].name
                          if c in present)
        (drop if children else keep)[row['name']] = children
    return drop, keep


def strength_class(psi):
    """The specified compressive-strength class of a concrete record."""
    out = pd.cut(pd.Series(psi, dtype=float), list(PSI_EDGES),
                 labels=list(PSI_LABELS), right=False)
    return out.cat.add_categories('strength not stated').fillna(
        'strength not stated')


def material_type(text):
    """The insulation material named in a product's name and description."""
    t = pd.Series(text, dtype=object).fillna('').str.lower()
    out = pd.Series(UNSTATED, index=t.index)
    for name, pattern in MATERIAL_TYPES:
        hit = t.str.contains(pattern, regex=True, na=False) & (out == UNSTATED)
        out[hit] = name
    return out


def assign(records, tree):
    """Assign every record to a dataset. Returns (labels, report).

    `records` is one row per retained ECC value with `material_query`, the
    product `name` and `description`, and the declared concrete strength.
    `labels` is a Series aligned to `records` holding the dataset each record
    belongs to, or None where the record is dropped. `report` is one row per
    resulting population with the rule, the field and the reason.
    """
    labels = records.material_query.astype(object).copy()
    report = []
    drop, keep = residual_bins(records.material_query.unique(), tree)

    text = (records.name.fillna('').astype(str) + ' '
            + records.description.fillna('').astype(str)).str.lower()

    for cat, why in NOT_ONE_POPULATION.items():
        sel = records.index[records.material_query == cat]
        if not len(sel):
            continue
        labels.loc[sel] = None
        report.append(dict(
            category=cat, rule='not one population', field='name and description',
            population=f'{cat} (dropped)', n=len(sel), kept=False,
            reason=(f'{cat} is an EC3 leaf category, but it is not one product '
                    f'population: {why}. No split can make it one, so it is '
                    f'dropped whole rather than curated record by record.')))

    for cat, (pattern, why) in EXCLUDED_PRODUCTS.items():
        sel = records.index[(records.material_query == cat)
                            & text.str.contains(pattern, regex=True, na=False)]
        if not len(sel):
            continue
        labels.loc[sel] = None
        report.append(dict(
            category=cat, rule='excluded product', field='name and description',
            population=f'{cat} (records removed)', n=len(sel), kept=False,
            reason=f'Removed from {cat}: {why}.'))

    for cat, children in drop.items():
        sel = records.index[records.material_query == cat]
        labels.loc[sel] = None
        report.append(dict(
            category=cat, rule='residual bin', field='EC3 category tree',
            population=f'{cat} (dropped)', n=len(sel), kept=False,
            reason=(f'{cat} is an EC3 PARENT category, so its records are the '
                    f'EPDs that EC3 did not place in any of its children. It is '
                    f'a residual bin rather than a product. Its '
                    f'{len(children)} child categor'
                    f'{"y is" if len(children) == 1 else "ies are"} already in '
                    f'this arm ({", ".join(children)}), so the material stays '
                    f'represented.')))
    for cat in keep:
        report.append(dict(
            category=cat, rule='residual bin', field='EC3 category tree',
            population=cat, n=int((records.material_query == cat).sum()),
            kept=True,
            reason=(f'{cat} is an EC3 parent category, but none of its children '
                    f'is in this arm, so dropping it would remove the material '
                    f'from the study. Kept, and the heterogeneity is a stated '
                    f'limitation.')))

    for cat in CONCRETE:
        g = records[records.material_query == cat]
        if g.empty:
            continue
        cls = strength_class(g.concrete_compressive_strength_28d_value)
        _apply(labels, report, g, cls, cat, rule='specified property',
               field='concrete_compressive_strength_28d',
               why=(f'{cat} is specified by 28-day compressive strength, which '
                    f'is the primary characteristic a structural engineer '
                    f'selects concrete by, so each strength class is a '
                    f'different product'))

    for cat in INSULATION:
        g = records[records.material_query == cat]
        if g.empty:
            continue
        typ = material_type(g.name.fillna('') + ' ' + g.description.fillna(''))
        _apply(labels, report, g, typ, cat, rule='product type',
               field='name and description',
               why=(f'{cat} of different material type are different products '
                    f'with different emissions, and EC3 records the type only '
                    f'in the product name and description'))

    return labels, pd.DataFrame(report)


def _apply(labels, report, g, groups, cat, rule, field, why):
    """Label one category's groups, dropping any below the three-value floor."""
    counts = groups.value_counts()
    viable = [k for k in counts.index if counts[k] >= MIN_N]
    if len(viable) < 2:
        report.append(dict(category=cat, rule=rule, field=field, population=cat,
                           n=len(g), kept=True,
                           reason=f'{why}, but only one group reaches '
                                  f'{MIN_N} records, so the category is left '
                                  f'whole.'))
        return
    breakdown = ', '.join(f'{counts[k]} {k}' for k in viable)
    for k in counts.index:
        sel = g.index[groups == k]
        if counts[k] >= MIN_N:
            labels.loc[sel] = f'{cat} [{k}]'
            report.append(dict(
                category=cat, rule=rule, field=field,
                population=f'{cat} [{k}]', n=int(counts[k]), kept=True,
                reason=f'{why}. {cat} holds {breakdown}.'))
        else:
            labels.loc[sel] = None
            report.append(dict(
                category=cat, rule=rule, field=field,
                population=f'{cat} [{k}] (dropped)', n=int(counts[k]),
                kept=False,
                reason=f'Fewer than {MIN_N} records, the same floor that drops '
                       f'a whole category with under {MIN_N} values.'))
