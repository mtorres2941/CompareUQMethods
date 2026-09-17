"""The material tiers, and the constraint that makes them usable.

The whole point of stratifying by material is to answer "does the KDE win where
it matters". That answer is worthless if the tiers were drawn after looking at
which categories the KDE won. So the binding test is not that any particular
category lands in any particular tier; it is that the classifier CANNOT see a
result, and that it is a pure function of the name.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'src'))

import materialclass as MC   # noqa: E402


def test_the_classifier_takes_a_name_and_nothing_else():
    """It is called with a string. There is no data in the room."""
    assert MC.tier_of('ReadyMix [4000-4999 psi]') == 'structure'
    assert MC.tier_of('BoardInsulation [XPS]') == 'envelope'
    assert MC.tier_of('WorkSurfaces') == 'other'


def test_a_frame_with_no_values_classifies_completely():
    """Drive the whole assignment on names alone, as
    `tests/test_categorysplit.py` does for the split rules. If this needs a
    value column to run, the classification could be circular."""
    names = ['ReadyMix [<3000 psi]', 'RebarSteel', 'MassTimber',
             'BlanketInsulation [mineral wool]', 'Windows', 'Carpet',
             'Elevators', 'AluminiumExtrusions']
    frame = pd.DataFrame({'dataset': names})
    assert 'ecc' not in frame.columns and 'value' not in frame.columns
    out = MC.add_tier(frame)
    assert list(out.tier) == ['structure', 'structure', 'structure',
                              'envelope', 'envelope', 'other', 'other',
                              'envelope']


def test_every_tier_is_one_of_three_and_unknown_names_fall_through():
    for name in ('', 'NotAThing', '12345', 'ReadyMixed'):
        assert MC.tier_of(name) in MC.TIER_ORDER
    assert MC.tier_of('NotAThing') == 'other'


def test_the_concrete_and_steel_families_are_complete():
    """Concrete and steel dominate embodied carbon in essentially every
    whole-building study, so a category of either that fell into `other` would
    quietly weaken the comparison it is there to make."""
    concrete = ['ReadyMix [3000-3999 psi]', 'ReadyMix [strength not stated]',
                'Shotcrete [>=6000 psi]', 'ConcretePaving [<3000 psi]',
                'CMU [strength not stated]', 'PrecastConcrete',
                'CastDecksAndUnderlayment']
    steel = ['RebarSteel', 'PlateSteel', 'HotRolled', 'DeckingSteel',
             'WireMeshSteel', 'PostTensioningSteel', 'MBQ', 'Hollow', 'Coil']
    for name in concrete + steel:
        assert MC.tier_of(name) == 'structure', name


def test_every_insulation_variant_is_envelope():
    for name in ('BoardInsulation [EPS]', 'BlanketInsulation [other]',
                 'BlownInsulation [cellulose]', 'MechanicalInsulation',
                 'FoamedInPlace [PIR or PUR]', 'SprayedInsulation'):
        assert MC.tier_of(name) == 'envelope', name


def test_the_classification_is_deterministic_and_order_free():
    names = ['Carpet', 'RebarSteel', 'Windows'] * 3
    a = [MC.tier_of(n) for n in names]
    b = [MC.tier_of(n) for n in reversed(names)]
    assert a == list(reversed(b))


def test_tier_table_publishes_the_whole_assignment():
    """It has to be publishable: a reader cannot check a hot-spot argument
    without seeing which categories were called structural."""
    t = MC.tier_table(['Carpet', 'RebarSteel', 'BoardInsulation [XPS]'])
    assert set(t.columns) == {'dataset', 'tier'}
    assert len(t) == 3
