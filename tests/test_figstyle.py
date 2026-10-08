"""The method colors live in one place and match what the notebooks used.

Added in the figure pass of 2026-10-07, when the author found the same method
drawn in two different colors in two figures. `figstyle.METHOD_COLORS` is the
one source; this pins that it is exactly seaborn's 'Paired' palette in
`fitting.PEWT` order, which is what every earlier method figure was drawn with,
so moving the definition changed no color.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'src'))

import fitting  # noqa: E402
import figstyle  # noqa: E402


def test_method_colors_cover_every_method_in_order():
    assert list(figstyle.METHOD_COLORS) == list(fitting.PEWT)


def test_method_colors_are_the_paired_palette_the_notebooks_used():
    import seaborn as sns
    paired = sns.color_palette('Paired', len(fitting.PEWT)).as_hex()
    assert [figstyle.METHOD_COLORS[m] for m in fitting.PEWT] == paired


def test_family_color_is_the_market_weights_shade():
    for fam, col in figstyle.FAMILY_COLORS.items():
        assert col == figstyle.METHOD_COLORS[f'{fam}, Variable']
