"""The figure manifest, as a standing guard rather than a one-off report.

THREE FAILURES THIS CATCHES, all of which have actually happened here.

A DUPLICATE FILENAME. Two cells once both wrote `CompareUQMethods_FIG2_*`.
Nothing failed; the second cell's figure simply replaced the first cell's, and
the repository carried two "Figure 2" names for two different figures until
Stage 3 listed them.

AN ORPHAN. Two committed images dated 2026-03 had no producer anywhere in the
tree and had to be deleted in Stage 2a because nobody could say what made them.
A figure nothing generates cannot be checked, cannot be regenerated, and is one
corpus regeneration away from being silently wrong.

A RETIRED NAME. The vocabulary settled at the close of Stage 2h and the figures
are where an old label survives longest, because nothing reads a filename.
"""
import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

_spec = importlib.util.spec_from_file_location(
    'figure_manifest', os.path.join(ROOT, 'audits', 'figure_manifest.py'))
manifest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(manifest)

#: Retired at the close of Stage 2h, decision 199. These must not appear in a
#: filename, because a reader of the deposit sees filenames and nothing in the
#: pipeline would ever notice.
RETIRED_WORDS = ('Variable', 'variable', 'SampledShare', 'sampled_share',
                 'Dirichlet', 'dirichlet')

#: `w_v_uw_wasserstein` is a stored COLUMN name, not a weighting label, and the
#: per-characteristic supplement pages are named after their column. Renaming
#: the column would move the join key of every table and fixture.
FILENAME_EXEMPT = ('CompareUQMethods_SUPP_ByCharacteristic_w_v_uw_wasserstein',)


@pytest.fixture(scope='module')
def gens():
    return manifest.generators()


def test_no_two_generators_write_one_filename(gens):
    dupes = {name: sorted(set(labels)) for name, labels in gens.items()
             if len(set(labels)) > 1}
    assert not dupes, (
        'two places write one figure filename, so the second silently '
        f'discards the first: {dupes}')


def test_every_committed_figure_has_a_generator(gens):
    orphans = []
    for rel, _size in manifest.images_on_disk():
        name = os.path.basename(rel)
        if name in gens:
            continue
        if any(name.startswith(g.split('{')[0]) for g in gens
               if '[templated]' in g):
            continue
        orphans.append(rel)
    assert not orphans, (
        'these images have no generator anywhere in the repository. Move them '
        f'to archive/ with a reason, or restore the code that made them: '
        f'{orphans}')


def test_figure_filenames_carry_no_retired_vocabulary():
    bad = []
    for rel, _ in manifest.images_on_disk():
        stem = os.path.splitext(os.path.basename(rel))[0]
        if stem in FILENAME_EXEMPT:
            continue
        for word in RETIRED_WORDS:
            if word in stem:
                bad.append((rel, word))
    assert not bad, (
        'the weighting vocabulary settled in decision 199 is "market weights" '
        f'and "uniform weights"; these filenames still carry a retired word: '
        f'{bad}')


def test_every_png_has_a_vector_sibling():
    """A journal wants a vector figure. figstyle.savefig writes both at once.

    Only `outputs/figures/` is checked: archive/ is frozen and the audit
    figures under outputs/tables/audits/ are working output, not publication
    output.
    """
    figdir = os.path.join(ROOT, 'outputs', 'figures')
    missing = [f for f in sorted(os.listdir(figdir))
               if f.endswith('.png')
               and not os.path.exists(os.path.join(figdir,
                                                   f[:-4] + '.pdf'))]
    assert not missing, (
        f'{len(missing)} figure(s) have no vector sibling; these are written '
        f'by a cell that still calls plt.savefig directly rather than '
        f'figstyle.savefig: {missing}')


def test_every_figure_cell_runs_against_the_setup_block_alone():
    """The fast renderer must be able to redraw ANY figure, not just the ones
    that happen to be self-sufficient.

    `audits/render_figures.py` executes the setup block -- every code cell up
    to and including the one defining OUT -- and then one figure cell. A figure
    cell reading a frame or a helper that a compute cell in between defines
    raises, and `--only` hides it: render one cell that happens to work and the
    tool reports success. That is how notebook 3 was believed clear through
    Stage 3 while nine of its thirteen figure cells could not be rendered at
    all, which meant a label change there cost the whole 110-minute run.

    All 37 pass as of Stage 4. This test is what stops that regressing: a new
    figure cell that reaches into kernel state fails here rather than in a
    renderer run somebody tries six weeks later.
    """
    rows = manifest.renderer_safety()
    bad = [(r['notebook'], r['cell'], r['missing'])
           for r in rows if not r['renderable']]
    assert not bad, (
        f'{len(bad)} of {len(rows)} figure cells cannot be rendered on their '
        'own, so changing one costs a full notebook run. Each needs the named '
        'frames persisted to a table and read back, or the helper moved into '
        f'the setup cell: {bad}')
