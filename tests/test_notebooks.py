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
