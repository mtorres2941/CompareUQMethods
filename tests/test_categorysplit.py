"""The rules that turn EC3 categories into specifiable product populations.

These guard the two rules added after the author asked whether records we are
confident are wrong should simply be excluded. The answer turned on WHERE the
evidence comes from: a product name is metadata and may be used, a record's
distance from its neighbours is the dispersion this study measures and may not.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import categorysplit as CS  # noqa: E402
import empirical  # noqa: E402


def records(names, category):
    return pd.DataFrame({
        'material_query': category,
        'name': names,
        'description': '',
        'concrete_compressive_strength_28d_value': np.nan,
    })


TREE = pd.DataFrame({'name': [], 'parent_name': [], 'is_leaf': []}).astype(
    {'name': object, 'parent_name': object, 'is_leaf': bool})


def test_a_category_that_is_not_one_population_is_dropped_whole():
    df = records(['1 1/2" Gravel', 'Crushed rock'], 'Chairs')
    labels, report = CS.assign(df, TREE)
    assert labels.isna().all(), 'every record in a dropped category goes'
    row = report[report.rule == 'not one population'].iloc[0]
    assert row.category == 'Chairs' and not row.kept
    assert 'not one product population' in row.reason


def test_excluded_products_remove_only_the_named_intruders():
    df = records(['1 1/2" Gravel', 'GRANITEK Sinks', 'Composite granite kitchen sinks',
                  'Recycled Glass Sand -5MM', 'Porcelain stoneware'], 'Aggregates')
    labels, report = CS.assign(df, TREE)
    kept = [n for n, lab in zip(df.name, labels) if lab is not None]
    assert kept == ['1 1/2" Gravel', 'Recycled Glass Sand -5MM']
    row = report[report.rule == 'excluded product'].iloc[0]
    assert row.n == 3


def test_the_exclusion_list_does_not_touch_a_clean_category():
    """PowerCabling and Elevators were checked and need no entry.

    An earlier INCLUSION rule removed 22 cables whose names are type
    designations rather than English words, and two elevators named by model
    number. This is the regression guard for that.
    """
    cables = ['TECK90 12 AWG/3C + 14 AWG GRD 600V', 'Cable a Haute Tension',
              'NF C 33-226 30kV Triphase', 'H07RN-F', 'S/FTP', 'ASTER GAINE']
    labels, _ = CS.assign(records(cables, 'PowerCabling'), TREE)
    assert labels.notna().all(), 'no cable may be dropped'

    lifts = ['S-P-02968 Schindler 6000 Europe', 'GreenWeight from Ergin Makina']
    labels, _ = CS.assign(records(lifts, 'Elevators'), TREE)
    assert labels.notna().all(), 'no elevator may be dropped'


def test_the_split_cannot_see_the_ecc_values_at_all():
    """The strongest form of the constraint, tested rather than asserted.

    Decision 43 says a record may be judged on metadata and never on its ECC.
    Rather than inspecting the patterns for numbers -- `rj45` is a connector,
    not a magnitude -- this drives the whole assignment on a frame that has NO
    ecc column. If it completes, no rule can be keying on a value.
    """
    df = records(['1 1/2" Gravel', 'GRANITEK Sinks'], 'Aggregates')
    assert 'ecc' not in df.columns
    labels, report = CS.assign(df, TREE)
    assert labels.tolist() == ['Aggregates', None]
    assert len(report)


def test_dropped_categories_are_gone_from_the_built_arm():
    datasets, _ = empirical.prepare(np.random.default_rng(42).spawn(1)[0])
    for cat in CS.NOT_ONE_POPULATION:
        assert cat not in datasets
    assert len(datasets) == 147
