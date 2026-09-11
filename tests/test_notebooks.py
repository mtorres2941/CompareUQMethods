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
