"""Redraw a notebook's figures without re-running the notebook.

WHY THIS EXISTS. Notebook 4 takes about twenty minutes end to end, almost all
of it fitting models, and a figure iteration changes twenty lines of matplotlib
that read tables already on disk. Paying twenty minutes to move a label is the
kind of cost that stops figures being iterated on at all, which shows.

THE GUARANTEE, AND IT IS WHAT MAKES THIS SAFE. **This script contains no
analysis code and no figure code.** It reads the notebook, executes the
notebook's own SETUP cell to get the imports and configuration, then executes
the notebook's own FIGURE cells, verbatim, as they appear in the .ipynb. There
is nothing here to drift out of step with the notebook, because there is
nothing here at all: if a figure changes, it changes in the notebook and this
re-executes it. `tests/test_render_figures.py` asserts that the source executed
is byte-identical to the notebook's.

WHERE IT WRITES. A scratch directory by default, with `tables/` symlinked to
the real one so the figures read production data. `--into-outputs` points it at
the repository's own `outputs/`, and the author allowed that on 2026-09-22:
"I want all figures to be reproducible in the notebooks, but if you have a
faster way to reproduce the figure so we can iterate, I'm open to it. As long
as the notebook reflects those changes." That narrows decision 56 rather than
reversing it -- `outputs/` is still written only by notebook source, and this
is a different executor for the same bytes, not a second author of figures.

    python audits/render_figures.py 04_CompareUQ_ReduceMetrics --out /tmp/figs
    python audits/render_figures.py 04_CompareUQ_ReduceMetrics --into-outputs
"""
import argparse
import json
import os
import re
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))

#: A code cell is a figure cell if its first line starts with one of these.
#: The notebooks already label them this way; nothing new is imposed.
FIGURE_MARKERS = ('# FIGURE', '# SUPPLEMENT')

#: What counts as saving a figure. The dotted form matters: a bare `savefig`
#: also appears in `rcParams['savefig.dpi']`, which is configuration and not a
#: figure, and treating it as one made the setup cell look like a figure cell.
SAVE_CALL = '.savefig('


def notebook_path(name):
    if not name.endswith('.ipynb'):
        name += '.ipynb'
    return os.path.join(ROOT, 'notebooks', name)


def read_cells(path):
    """(setup_source, [(index, title, source)]) for the figure cells.

    REFUSES rather than skips. A cell that saves a figure without announcing
    itself would be quietly left out, so the tool would redraw some figures and
    not others and report success -- which is worse than not existing, because
    the stale ones would be committed alongside the fresh ones.
    """
    nb = json.load(open(path))
    setup, figs, unmarked, lead = None, [], [], []
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        src = ''.join(c['source'])
        head = src.lstrip().split('\n', 1)[0]
        marked = head.startswith(FIGURE_MARKERS)
        if setup is None:
            # THE SETUP IS EVERY CODE CELL UP TO AND INCLUDING THE ONE THAT
            # DEFINES `OUT`, executed ONE CELL AT A TIME in the notebook's own
            # order. Notebook 4 puts its imports and its output root in one
            # cell; notebook 3 does not -- its imports are two cells earlier --
            # and taking only the `OUT` cell there gave a NameError on numpy
            # before the first figure drew.
            #
            # A LIST AND NOT A CONCATENATION, because a cell whose source has
            # no trailing newline runs fine in a notebook and glues onto the
            # next one here: joining them produced `generate_dontread = False`
            # followed immediately by `import sys` and a SyntaxError. Running
            # each cell separately is what the notebook does anyway, and it
            # keeps the bytes executed exactly the notebook's.
            lead.append(src)
            if re.search(r'^OUT\s*=', src, re.M):
                setup = list(lead)
        if marked:
            figs.append((i, head.lstrip('# ').strip().rstrip('.'), src))
        elif SAVE_CALL in src:
            unmarked.append(i)
    if setup is None:
        raise SystemExit(
            f'{os.path.basename(path)} defines no OUT, so there is no setup '
            f'cell to execute and no output root to redirect')
    if unmarked:
        raise SystemExit(
            f'{os.path.basename(path)} cells {unmarked} save a figure but do '
            f'not start with one of {FIGURE_MARKERS}. Refusing rather than '
            f'redrawing only some of the figures.')
    return setup, figs


def prepare(out_dir):
    """A directory shaped like `outputs/`, reading the real tables."""
    figs = os.path.join(out_dir, 'figures')
    os.makedirs(figs, exist_ok=True)
    link = os.path.join(out_dir, 'tables')
    real = os.path.join(ROOT, 'outputs', 'tables')
    if os.path.islink(link):
        os.unlink(link)
    if not os.path.exists(link):
        os.symlink(real, link)
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('notebook')
    ap.add_argument('--out', default=None,
                    help='directory to write into; defaults to a scratch dir')
    ap.add_argument('--into-outputs', action='store_true',
                    help="write to the repository's own outputs/")
    ap.add_argument('--only', default=None,
                    help='substring of the figure cell title to render')
    args = ap.parse_args(argv)

    path = notebook_path(args.notebook)
    setup, figs = read_cells(path)
    if args.only:
        figs = [f for f in figs if args.only.lower() in f[1].lower()]
        if not figs:
            raise SystemExit(f'no figure cell matching {args.only!r}')

    if args.into_outputs:
        out = os.path.join(ROOT, 'outputs')
    else:
        out = prepare(args.out or os.path.join(ROOT, '.render'))

    warnings.filterwarnings('ignore')
    os.chdir(os.path.join(ROOT, 'notebooks'))
    import matplotlib
    matplotlib.use('Agg')

    g = {'__name__': '__main__'}
    for j, cell in enumerate(setup):                   # the notebook's imports
        exec(compile(cell, f'setup{j}', 'exec'), g)
    g['OUT'] = out                                     # ... redirected
    g['display'] = lambda *a, **k: None                # no rich display here
    for i, title, src in figs:
        print(f'cell {i}: {title}', flush=True)
        exec(compile(src, f'cell{i}', 'exec'), g)
    import matplotlib.pyplot as plt
    plt.close('all')
    print(f'wrote {len(figs)} figure(s) under {out}/figures')


if __name__ == '__main__':
    sys.exit(main())
