"""
Static guards on the notebooks.

`test_all_code_cells_parse` exists because an earlier Stage 1 patch script
rebuilt cell sources with `[l + '\\n' for l in src.split('\\n')][:-1]`, which
silently drops the final line of any cell whose source does not end in a
newline. It truncated notebook 1 cell 24 mid-statement, and the only symptom
was a SyntaxError twelve minutes into a headless run. A parse check over every
cell catches that class of damage in under a second.
"""

import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(
    p for p in (ROOT / "notebooks").glob("*.ipynb")
    if not p.name.startswith(("VOID_", "_")) and "backup" not in p.name
)

# The single permitted construction of a Generator per notebook.
ALLOWED_GLOBAL_RANDOM = "np.random.default_rng("


def code_cells(path):
    nb = json.loads(path.read_text())
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            yield i, "".join(cell["source"])


def test_notebooks_found():
    assert len(NOTEBOOKS) == 3, [p.name for p in NOTEBOOKS]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_all_code_cells_parse(path):
    broken = []
    for index, source in code_cells(path):
        if not source.strip():
            continue
        try:
            ast.parse(source)
        except SyntaxError as exc:
            broken.append((index, str(exc)))
    assert not broken, f"{path.name}: unparseable cells {broken}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_global_numpy_randomness(path):
    """All randomness must come from an explicitly passed Generator.

    The only permitted reference to the legacy global API is the single
    `np.random.default_rng(SEED)` that creates the notebook's Generator.
    """
    offenders = []
    for index, source in code_cells(path):
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for match in re.finditer(r"np\.random\.\w+\(", stripped):
                if match.group(0) != ALLOWED_GLOBAL_RANDOM:
                    offenders.append((index, stripped))
    assert not offenders, f"{path.name}: global numpy randomness at {offenders}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_generator_is_created_exactly_once(path):
    creations = [
        (i, line.strip())
        for i, source in code_cells(path)
        for line in source.splitlines()
        if ALLOWED_GLOBAL_RANDOM in line and not line.strip().startswith("#")
    ]
    assert len(creations) == 1, f"{path.name}: expected 1 Generator, found {creations}"


#: Bytes. A notebook carrying stored figure outputs runs to hundreds of
#: megabytes, and GitHub refuses any single file over 100 MB outright.
NOTEBOOK_SIZE_LIMIT = 8 * 1024 * 1024


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_carry_no_stored_output(path):
    """Outputs are stripped before commit; the reader re-runs to see them.

    Figures are embedded base64, so a notebook that stores them grows without
    bound: notebook 2 reached 196 MB and could not be pushed at all, GitHub's
    hard per-file limit being 100 MB. Everything a stored output would show is
    on disk in outputs/ and is reproduced by running the notebook.

    To strip them:

        jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
    """
    nb = json.loads(path.read_text())
    offenders = [
        i for i, cell in enumerate(nb.get("cells", []))
        if cell.get("cell_type") == "code" and cell.get("outputs")
    ]
    assert not offenders, (
        f"{path.name}: {len(offenders)} cells carry stored output "
        f"(first at {offenders[:5]}). Run: "
        f"jupyter nbconvert --clear-output --inplace notebooks/*.ipynb"
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_stay_small(path):
    size = path.stat().st_size
    assert size <= NOTEBOOK_SIZE_LIMIT, (
        f"{path.name} is {size / 1048576:.1f} MB, over the "
        f"{NOTEBOOK_SIZE_LIMIT / 1048576:.0f} MB guard. Stored output is the "
        f"usual cause."
    )


#: Dots per inch a figure may be SAVED at. 300 is print quality for a journal.
#: Above this a figure is not better, only larger: notebook 2 set
#: `figure.dpi = 1200`, and because `savefig.dpi` defaults to `'figure'` that
#: was silently the save resolution for every figure in the notebook, while
#: notebook 3 passed `dpi=1200` to six savefig calls directly. The result was a
#: 98-megapixel scatter plot and a notebook too large for GitHub to accept.
MAX_SAVE_DPI = 300


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_figures_are_not_saved_above_print_resolution(path):
    offenders = []
    for index, source in code_cells(path):
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for match in re.finditer(r"\bdpi\s*=\s*(\d+)", stripped):
                dpi = int(match.group(1))
                # figure.dpi is the SCREEN resolution and is allowed to be low;
                # only the save resolution is capped here.
                if "figure.dpi" in stripped:
                    continue
                if dpi > MAX_SAVE_DPI:
                    offenders.append((index, dpi, stripped[:90]))
    assert not offenders, (
        f"{path.name}: saving above {MAX_SAVE_DPI} dpi at {offenders}. "
        f"Layout is measured in inches, so a higher dpi makes the file bigger "
        f"and nothing else."
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_screen_dpi_is_not_used_as_save_dpi(path):
    """`savefig.dpi` defaults to `'figure'`, which is the trap that caused this.

    Setting `figure.dpi` high to get a crisp inline preview silently raises the
    resolution of every file the notebook writes. A notebook that sets
    `figure.dpi` must set `savefig.dpi` explicitly alongside it.
    """
    sets_figure_dpi = sets_savefig_dpi = False
    for _, source in code_cells(path):
        for line in source.splitlines():
            if line.strip().startswith("#"):
                continue
            if "figure.dpi" in line:
                sets_figure_dpi = True
            if "savefig.dpi" in line:
                sets_savefig_dpi = True
    if sets_figure_dpi:
        assert sets_savefig_dpi, (
            f"{path.name} sets figure.dpi without setting savefig.dpi, so the "
            f"screen resolution silently becomes the file resolution."
        )


#: A `to_csv`, `to_excel` or `savefig` writing under `outputs/`, with the path
#: as a plain literal. Paths built from a variable are not caught, and there are
#: none at present.
# A write into the repository's outputs, in either of the two forms the
# notebooks use: a literal '../outputs/...' path, or f'{OUT}/...', which is how
# notebook 3 redirects itself away from outputs/ under smoke mode. The captured
# group is the path below the output root in both cases.
_WRITE = re.compile(
    r"""\.(?:to_csv|to_excel|savefig)\(\s*f?['"](?:\{OUT\}|[^'"]*outputs)/([^'"]+)['"]""")


def test_no_two_notebooks_write_the_same_output_file():
    """Two notebooks writing one filename means the second silently wins.

    Found in Stage 2c: a new post-stratification table in notebook 2 was given
    the name notebook 1 already used for the post-stratified dataset
    CHARACTERISTICS, so running the pair would have left one table on disk
    describing something other than its filename. Nothing else would have
    noticed, because each notebook runs green on its own.
    """
    writers = {}
    for path in NOTEBOOKS:
        for _, source in code_cells(path):
            for target in _WRITE.findall(source):
                writers.setdefault(Path(target).name, set()).add(path.name)
    clashes = {name: sorted(who) for name, who in writers.items()
               if len(who) > 1}
    assert not clashes, f'written by more than one notebook: {clashes}'

@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_cell_uses_a_frame_defined_below_it(path):
    """A cell may not read a name that a LATER cell assigns.

    WHY THIS EXISTS. A Stage 2d figure cell read `df_riskdecomp`, which the cell
    two positions below it creates. Every interactive session had that name in
    memory from an earlier run, and rendering the figure cells on their own
    loaded the frames from disk first, so the error only surfaced in a full
    headless run -- eleven minutes in, which is the expensive place to find it.

    The check is deliberately narrow: top-level `df_*` assignments only. A
    general dataflow analysis of a notebook is not worth writing, and this
    catches the shape of the mistake that actually happened.
    """
    nb = json.loads(path.read_text())
    code = [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"])
            if c["cell_type"] == "code"]
    defined = {}
    for i, src in code:
        for m in re.finditer(r"^(df_\w+)\s*=", src, re.M):
            defined.setdefault(m.group(1), i)
    bad = []
    for i, src in code:
        for name, j in defined.items():
            if j > i and re.search(rf"\b{name}\b", src):
                bad.append(f"cell {i} uses {name}, first assigned in cell {j}")
    assert not bad, f"{path.name}: " + "; ".join(bad)


# ---------------------------------------------------------------------------
# a smoke run must not be able to reach outputs/
# ---------------------------------------------------------------------------
PLCA_NOTEBOOK = ROOT / "notebooks" / "03_CompareUQ_PerformPLCA.ipynb"


def test_the_plca_notebook_writes_only_through_its_output_root():
    """Notebook 3 is the only notebook with a smoke mode, and in Stage 2d a
    smoke run reached a commit: it replaced the 60,000-row pLCA results table
    with a 960-row one and redrew seven figures from 40 pLCA groups instead of
    2,500. The rule against that was a sentence in a document.

    It is now a mechanism. Every path the notebook writes goes through `OUT`,
    which smoke mode points at a temporary directory, so a smoke run cannot
    touch the repository at all. This asserts that no cell has been written
    since with a literal path back into outputs/.
    """
    offenders = []
    for index, source in code_cells(PLCA_NOTEBOOK):
        for line in source.splitlines():
            code = line.split('#', 1)[0]
            if '../outputs' in code and not re.match(r'\s*OUT\s*=', code):
                offenders.append(f'cell {index}: {line.strip()}')
    assert not offenders, (
        'notebook 3 must write through OUT, not a literal outputs/ path, or a '
        'smoke run will overwrite committed results: ' + '; '.join(offenders))


def test_the_plca_notebook_defines_its_output_root_before_it_writes():
    """OUT has to exist before the first write, or the guard is decorative."""
    cells = list(code_cells(PLCA_NOTEBOOK))
    defines = [i for i, src in cells if re.search(r"^OUT\s*=", src, re.M)]
    writes = [i for i, src in cells if '{OUT}' in src and 'OUT =' not in src]
    assert defines, 'notebook 3 never defines OUT'
    assert min(writes) > min(defines), (
        f'first write in cell {min(writes)} precedes OUT in cell {min(defines)}')


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_cell_shadows_an_imported_module(path):
    """A variable may not take the name of a module the notebook imports.

    WHY THIS EXISTS. Notebook 3 used `plca` as a loop index years before
    `src/plca.py` existed, so importing the module left every later call to it
    reading an integer: `AttributeError: 'int' object has no attribute
    'run_group'`, eleven cells after the assignment. Nothing else would catch
    it, because the assignment and the call are in different cells and both are
    perfectly valid on their own.
    """
    imported = set()
    assigned = {}
    for index, source in code_cells(path):
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add((alias.asname or alias.name).split('.')[0])
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigned.setdefault(target.id, index)
    clashes = sorted(f'{name} (assigned in cell {assigned[name]})'
                     for name in imported & set(assigned))
    assert not clashes, f'{path.name}: shadowed modules: {clashes}'
