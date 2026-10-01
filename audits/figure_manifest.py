"""Every image in the repository, and the code that writes it.

WHY THIS EXISTS. Figure filenames in this repository drifted for two years:
some main-text figures were named SUPP, some supplements were named FIG,
several carried no number, two different cells both wrote `FIG2`, and at least
two committed images had no generator anywhere in the tree and had to be
deleted in Stage 2a because nobody could say what produced them.

WHAT IT DOES. Walks every code cell of every notebook and every module under
`src/` and `audits/`, finds what each one writes, and joins that against the
files actually on disk. A file with no generator is a stale candidate; a
generator writing a name nothing else mentions is the other kind of candidate.

    python audits/figure_manifest.py
    python audits/figure_manifest.py --write

`--write` puts the table under outputs/tables/audits/, which is the only place
an audit script may write (decision 56).
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
IMAGE_EXT = ('.png', '.pdf', '.svg', '.jpg', '.jpeg')

#: What announces a figure cell to audits/render_figures.py.
FIGURE_MARKERS = ('# FIGURE', '# SUPPLEMENT')

#: `plt.savefig('.../NAME.png')`, `fig.savefig(f'{OUT}/figures/NAME.png')`, and
#: the project's own `figstyle.savefig(fig, OUT, 'STEM')`.
RE_SAVEFIG = re.compile(r"savefig\(\s*f?['\"]([^'\"]*?([A-Za-z0-9_\-]+\.(?:png|pdf|svg)))['\"]")
RE_FIGSTYLE = re.compile(r"figstyle\.savefig\(\s*\w+\s*,\s*\w+\s*,\s*f?['\"]([^'\"]+)['\"]")
RE_FNAME_VAR = re.compile(r"['\"]([A-Za-z0-9_\-]*CompareUQMethods[A-Za-z0-9_\-]*\.(?:png|pdf|svg))['\"]")
#: An audit script writing a family of names into outputs/tables/audits, e.g.
#: f'FIG_CorpusExamples_{name}.png'. The literal prefix is what identifies the
#: family; the brace is where the varying part starts.
RE_TEMPLATED = re.compile(r"f['\"]([A-Za-z0-9_\-]+)\{[^'\"]*\.(?:png|pdf|svg)['\"]")
RE_STEM_VAR = re.compile(r"['\"]((?:FIG|SUPP|DEF)[A-Za-z0-9_\-]*)['\"]")


#: Scratch notebooks kept for history. They are not part of the analysis, they
#: write dozens of names that have not existed for two years, and counting them
#: as generators makes every live figure look duplicated.
LEGACY_NOTEBOOKS = ('CompareUQMethods_backup.ipynb',
                    'VOID_CompareUQMethods_wbLCA.ipynb',
                    '_SCRATCHPAD.ipynb')

#: This file's own docstring and the notebook guard's fixtures name files that
#: do not and should not exist.
SKIP_SOURCES = ('audits/figure_manifest.py', 'tests/test_notebooks.py',
                'tests/test_render_figures.py')


def code_units(include_legacy=False):
    """(source_label, text) for every notebook cell and every .py file."""
    nbdir = os.path.join(ROOT, 'notebooks')
    for name in sorted(os.listdir(nbdir)):
        if not name.endswith('.ipynb'):
            continue
        if name in LEGACY_NOTEBOOKS and not include_legacy:
            continue
        nb = json.load(open(os.path.join(nbdir, name)))
        for i, c in enumerate(nb['cells']):
            if c['cell_type'] == 'code':
                yield f'notebooks/{name} cell {i}', ''.join(c['source'])
    for sub in ('src', 'audits', 'tests'):
        d = os.path.join(ROOT, sub)
        for name in sorted(os.listdir(d)):
            if name.endswith('.py') and f'{sub}/{name}' not in SKIP_SOURCES:
                yield f'{sub}/{name}', open(os.path.join(d, name)).read()


def generators():
    """filename -> [source labels that write it]."""
    out = {}
    for label, text in code_units():
        names = set()
        for m in RE_SAVEFIG.finditer(text):
            names.add(os.path.basename(m.group(2)))
        for m in RE_FIGSTYLE.finditer(text):
            stem = m.group(1)
            if '{' in stem:                      # f-string stem, e.g. {base}
                names.add(f'CompareUQMethods_{stem}.png  [templated]')
            else:
                names.add(f'CompareUQMethods_{stem}.png')
                names.add(f'CompareUQMethods_{stem}.pdf')
        # a filename or a stem held in a variable and passed to savefig. The
        # win-share and per-characteristic figures build their stems in a loop,
        # so matching only the call site reports a live figure as an orphan.
        if 'savefig' in text:
            for m in RE_FNAME_VAR.finditer(text):
                names.add(os.path.basename(m.group(1)))
        for m in RE_TEMPLATED.finditer(text):
            names.add(f'{m.group(1)}{{...}}.png  [templated]')
        if 'figstyle.savefig' in text:
            for m in RE_STEM_VAR.finditer(text):
                names.add(f'CompareUQMethods_{m.group(1)}.png')
                names.add(f'CompareUQMethods_{m.group(1)}.pdf')
        for n in names:
            out.setdefault(n, []).append(label)
    return out


def images_on_disk():
    found = []
    for base, dirs, files in os.walk(ROOT):
        # archive/ holds figures that deliberately have no generator any more;
        # counting them as orphans would make the orphan count never reach zero.
        dirs[:] = [d for d in dirs
                   if d not in ('.git', 'refs', '.render', 'node_modules',
                                'archive')]
        for f in sorted(files):
            if f.lower().endswith(IMAGE_EXT):
                p = os.path.join(base, f)
                found.append((os.path.relpath(p, ROOT), os.path.getsize(p)))
    return found


#: What FIGURE_STYLE.md asks of a figure cell, as far as a reader of the source
#: can check it. The parts that matter most -- a title that states a finding, a
#: panel that earns its place -- cannot be checked here and are the checklist's
#: job. These four can.
STYLE_CALLS = {
    'apply': 'figstyle.apply(',          # the rcParams, including ASCII minus
    'savefig': 'figstyle.savefig(',      # one name, PNG plus vector
    'finish': 'figstyle.finish(',        # erase what the guide says to erase
    'overlaps': 'figstyle.check_overlaps(',
}


def style_compliance():
    """Which figure cells call which part of the style module."""
    rows = []
    for label, text in code_units():
        if 'savefig' not in text:
            continue
        head = text.lstrip().split('\n', 1)[0]
        if not head.startswith(('# FIGURE', '# SUPPLEMENT')):
            continue
        row = dict(source=label, title=head.lstrip('# ').strip()[:60])
        for name, call in STYLE_CALLS.items():
            row[name] = call in text
        rows.append(row)
    return rows


def renderer_safety():
    """Which figure cells the fast renderer can execute on its own.

    `audits/render_figures.py` runs the SETUP BLOCK -- every code cell up to and
    including the one that defines OUT -- and then a figure cell. A figure cell
    that reads a frame or a helper defined in a compute cell in between will
    raise, and `--only` hides it: render one cell that happens to be
    self-sufficient and the tool reports success.

    Stage 3 cleared notebooks 1 and 2 this way. Notebook 3 was believed clear
    because every use of it had passed `--only`; rendering all of it at once
    showed nine of thirteen figure cells reaching for something the setup block
    does not define.

    This is a STATIC check and deliberately conservative: it reports a name
    read but never bound above, which is the failure the renderer hits.
    """
    import ast
    import builtins
    rows = []
    for name in sorted(os.listdir(os.path.join(ROOT, 'notebooks'))):
        if not name.endswith('.ipynb') or name in LEGACY_NOTEBOOKS:
            continue
        nb = json.load(open(os.path.join(ROOT, 'notebooks', name)))
        cells = [(i, ''.join(c['source'])) for i, c in enumerate(nb['cells'])
                 if c['cell_type'] == 'code']
        setup_names = set(dir(builtins)) | {'OUT', 'UPSTREAM', 'SMOKE',
                                            'display', 'get_ipython'}
        after_setup = False
        for i, src in cells:
            bound, used = _names(src)
            if not after_setup:
                setup_names |= bound
                if re.search(r'^OUT\s*=', src, re.M):
                    after_setup = True
                continue
            head = src.lstrip().split('\n', 1)[0]
            if not head.startswith(FIGURE_MARKERS):
                continue
            missing = sorted(used - bound - setup_names)
            rows.append(dict(notebook=name, cell=i,
                             title=head.lstrip('# ').strip()[:52],
                             renderable=not missing,
                             missing='; '.join(missing[:6])))
    return rows


def _names(src):
    """(bound, used) at any nesting level, for the static check above."""
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set(), set()
    bound, used = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            (bound if isinstance(n.ctx, ast.Store) else used).add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.add(n.name)
            a = n.args
            for arg in (a.posonlyargs + a.args + a.kwonlyargs
                        + ([a.vararg] if a.vararg else [])
                        + ([a.kwarg] if a.kwarg else [])):
                bound.add(arg.arg)
        elif isinstance(n, ast.ClassDef):
            bound.add(n.name)
        elif isinstance(n, ast.Lambda):
            for arg in n.args.posonlyargs + n.args.args + n.args.kwonlyargs:
                bound.add(arg.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for alias in n.names:
                bound.add((alias.asname or alias.name).split('.')[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, ast.comprehension):
            for t in ast.walk(n.target):
                if isinstance(t, ast.Name):
                    bound.add(t.id)
    return bound, used


#: A figure cell that shows an AGGREGATE -- a mean, a median, an NRMSE, a win
#: share -- should show its uncertainty too. These are the crudest possible
#: proxies for "shows an aggregate" and "shows an interval", so the output is a
#: list to look at rather than a verdict: several of the cells it names are
#: scatters of every dataset, which need no interval at all.
RE_AGGREGATE = re.compile(
    r"\.(mean|median)\(|_mean\b|nrmse|win_share|best_error|total_error")
RE_INTERVAL = re.compile(
    r"fill_between|errorbar|_lo\b|_hi\b|_p9|ci_lo|ci_hi|\byerr\b|interval")


def aggregates_without_intervals():
    out = []
    for label, text in code_units():
        head = text.lstrip().split('\n', 1)[0]
        if not head.startswith(FIGURE_MARKERS):
            continue
        if RE_AGGREGATE.search(text) and not RE_INTERVAL.search(text):
            out.append((label, head.lstrip('# ').strip()[:52]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args(argv)

    gens = generators()
    rows = []
    for rel, size in images_on_disk():
        name = os.path.basename(rel)
        who = gens.get(name, [])
        # a templated stem covers a family of files
        if not who:
            for g, labels in gens.items():
                if '[templated]' not in g:
                    continue
                prefix = g.split('{')[0]
                if name.startswith(os.path.basename(prefix)):
                    who = labels
                    break
        rows.append(dict(file=rel, bytes=size,
                         generators='; '.join(sorted(set(who))),
                         n_generators=len(set(who)),
                         status='ORPHAN' if not who
                         else ('DUPLICATE' if len(set(who)) > 1 else 'ok')))

    on_disk = {os.path.basename(r['file']) for r in rows}
    unwritten = [(g, labels) for g, labels in sorted(gens.items())
                 if '[templated]' not in g and g not in on_disk]

    print(f'{len(rows)} image files, {len(gens)} generated names\n')
    for r in sorted(rows, key=lambda r: (r['status'] != 'ok', r['file'])):
        print(f"  {r['status']:>9}  {r['bytes']/1e3:8.0f} kB  {r['file']}")
        if r['generators']:
            print(f"             <- {r['generators']}")
    n_archived = sum(1 for _, _ in [(0, 0)] for _ in
                     os.listdir(os.path.join(ROOT, 'archive', 'figures'))
                     if os.path.isdir(os.path.join(ROOT, 'archive', 'figures')))
    print(f'\n{n_archived} image(s) in archive/figures, deliberately '
          f'without a generator; see archive/README.md')
    print(f"\nORPHANS (no generator anywhere): "
          f"{sum(r['status'] == 'ORPHAN' for r in rows)}")
    print(f"DUPLICATE generators: "
          f"{sum(r['status'] == 'DUPLICATE' for r in rows)}")
    print(f"\nNAMES A GENERATOR WRITES THAT ARE NOT ON DISK: {len(unwritten)}")
    for g, labels in unwritten:
        print(f'  {g}  <- {"; ".join(labels)}')

    style = style_compliance()
    print(f'\nFIGURE_STYLE.md COMPLIANCE, {len(style)} marked figure cells')
    print('  apply finish overlap  source')
    for r in style:
        if all(r[k] for k in STYLE_CALLS):
            continue
        print(f"  {'y' if r['apply'] else '.':5s} "
              f"{'y' if r['finish'] else '.':6s} "
              f"{'y' if r['overlaps'] else '.':7s} "
              f"{r['source']}  -- {r['title']}")
    n_full = sum(1 for r in style if all(r[k] for k in STYLE_CALLS))
    print(f'  {n_full} of {len(style)} call all four')

    safe = renderer_safety()
    bad = [r for r in safe if not r['renderable']]
    print(f'\nFAST-RENDERER SAFETY: {len(safe) - len(bad)} of {len(safe)} '
          f'figure cells run against the setup block alone')
    for r in bad:
        print(f"  {r['notebook']} cell {r['cell']}: needs {r['missing']}")

    nob = aggregates_without_intervals()
    print(f'\nFIGURE CELLS SHOWING AN AGGREGATE WITH NO INTERVAL: {len(nob)} '
          f'candidates, several of which are scatters that need none')
    for label, title in nob:
        print(f'  {label}  -- {title}')

    if args.write:
        import pandas as pd
        d = os.path.join(ROOT, 'outputs', 'tables', 'audits')
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, 'TABLE_FigureManifest.csv')
        pd.DataFrame(rows).to_csv(p, index=False)
        ps = os.path.join(d, 'TABLE_FigureStyleCompliance.csv')
        pd.DataFrame(style).to_csv(ps, index=False)
        pr = os.path.join(d, 'TABLE_FigureRendererSafety.csv')
        pd.DataFrame(safe).to_csv(pr, index=False)
        print(f'\nwrote {p}\nwrote {ps}\nwrote {pr}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
