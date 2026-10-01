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

    if args.write:
        import pandas as pd
        d = os.path.join(ROOT, 'outputs', 'tables', 'audits')
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, 'TABLE_FigureManifest.csv')
        pd.DataFrame(rows).to_csv(p, index=False)
        print(f'\nwrote {p}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
