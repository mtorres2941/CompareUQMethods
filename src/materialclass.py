"""Which EC3 categories carry a building's embodied carbon.

Stage 2c review. Every aggregate in this study weights each of the 147 categories
equally, so a 12-EPD category of work surfaces counts as much as `ReadyMix
[4000-4999 psi]` with 31,025. That is the honest unweighted answer to "which
method fits an arbitrary ECC dataset best", and it is not the question the paper
is about, which is which method to use in a whole-building LCA.

THE OBVIOUS FIX IS THE WRONG ONE. Weighting categories by n would be weighting by
how many EPDs a manufacturer happened to publish, which is a property of the
market's paperwork rather than of a building, and it correlates with exactly the
dimension the KDE wins on. The author named that trap directly: "weighting
importance by n seems like unfairly biasing towards KDE."

SO THE SPLIT IS BY WHAT A MATERIAL IS, from published building-LCA practice, and
it is fixed here before any result is looked at. The tiers are the standard
hot-spot ordering in whole-building embodied carbon: the structural frame and its
binders first, then the envelope, then everything else.

THE BINDING CONSTRAINT, and it is the same one decisions 43, 46 and 60 impose on
the category rules: **the classification may read only the category NAME, never
the ECC values and never a result.** `tests/test_materialclass.py` drives the
whole assignment on names alone, with no data in the room. A tier chosen because
the KDE happened to win on it would make the entire exercise circular.

WHAT IT IS NOT. It is not an importance weight and it does not pretend to be a
building. Two principled versions of that exist and neither is this: Stage 2i's
real-building anchor, and the pLCA-against-truth of discrepancy entry 69. This is
a stratification, so every number stays a plain mean within a named group.
"""

import re

#: Tier 1, THE STRUCTURAL FRAME AND ITS BINDERS. Concrete, cement and
#: cementitious binders, structural steel, and structural timber. In essentially
#: every whole-building embodied-carbon study these dominate, and concrete and
#: steel alone are usually a majority of the total.
STRUCTURE = (
    # concrete and concrete products
    r'^ReadyMix', r'^PrecastConcrete$', r'^Shotcrete', r'^ConcretePaving',
    r'^CMU', r'^CastDecksAndUnderlayment$',
    # cement and cementitious binders
    r'^Cement$', r'^MasonryCement$', r'^SupplementaryCementitiousMaterials$',
    r'^CementGrout$', r'^Mortar$', r'^FlowableFill$',
    # structural steel and reinforcement
    r'^RebarSteel$', r'^PlateSteel$', r'^HotRolled$', r'^DeckingSteel$',
    r'^WireMeshSteel$', r'^PostTensioningSteel$', r'^MBQ$', r'^Hollow$',
    r'^Coil$', r'^OpenWebMembranes$',
    # structural timber
    r'^MassTimber$', r'^HeavyTimber$', r'^Timber$', r'^WoodFraming$',
    r'^WoodJoists$', r'^CompositeLumber$',
)

#: Tier 2, THE ENVELOPE. Insulation, glazing and fenestration, aluminium,
#: masonry and sheathing. Second to the frame in most studies and first in
#: highly glazed or highly insulated buildings.
ENVELOPE = (
    r'Insulation', r'^FoamedInPlace', r'^SprayedInsulation$',
    r'^Aluminium',
    r'^FlatGlassPanes$', r'^InsulatingGlazingUnits$', r'^CurtainWalls$',
    r'^Windows$', r'^UnitSkylights$', r'^FenestrationFraming$',
    r'^ProcessedNonInsulatingGlassPanes$',
    r'^Brick$', r'^StoneCladding$',
    r'^Gypsum$', r'^GypsumSheathingBoard$', r'^CementBoard$',
    r'^CementitiousSheathingBoard$', r'^SheathingPanels$',
    r'^InsulatedWallPanels$', r'^InsulatedRoofPanels$', r'^RoofPanels$',
    r'^WallPanels$',
)

TIERS = (('structure', STRUCTURE), ('envelope', ENVELOPE))

#: Order the tiers are reported in, coarsest last.
TIER_ORDER = ('structure', 'envelope', 'other')


def tier_of(name):
    """'structure', 'envelope' or 'other', from the category NAME alone.

    No data is read and none is available to be read; see the module docstring.
    """
    text = str(name)
    for tier, patterns in TIERS:
        if any(re.search(p, text) for p in patterns):
            return tier
    return 'other'


def add_tier(frame, column='dataset'):
    """Add a `tier` column keyed off a category-name column."""
    return frame.assign(tier=frame[column].map(tier_of))


def tier_table(names):
    """One row per category with its tier, for the paper's appendix.

    The classification has to be published: a reader cannot check a hot-spot
    argument without seeing which categories were called structural.
    """
    import pandas as pd
    return pd.DataFrame({'dataset': list(names)}).pipe(add_tier).sort_values(
        ['tier', 'dataset']).reset_index(drop=True)
