"""The renderer must execute the NOTEBOOK's bytes and nothing of its own.

That is the only property that makes a second executor safe: if this script
could contribute a line of figure code, the figures in `outputs/` would no
longer be reproducible from the notebook, which is the rule it is carved out
of (decision 56, narrowed 2026-09-22).
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "audits"))

import render_figures as RF  # noqa: E402

NOTEBOOKS = sorted((ROOT / "notebooks").glob("0*_CompareUQ_*.ipynb"))


def test_the_renderer_holds_no_figure_code_of_its_own():
    """No matplotlib, no savefig, no analysis import anywhere in the module."""
    src = (ROOT / "audits" / "render_figures.py").read_text()
    body = "\n".join(l for l in src.splitlines()
                     if not l.strip().startswith("#"))
    body = body.split('"""', 2)[-1]        # drop the module docstring
    for banned in ("plt.subplots", "import numpy", "import pandas",
                   "metricreduction", "figstyle", "set_title", "ax."):
        assert banned not in body, f"renderer contains its own {banned!r}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_sources_executed_are_byte_identical_to_the_notebook(path):
    nb = json.load(open(path))
    try:
        setup, figs = RF.read_cells(path)
    except SystemExit:
        pytest.skip(f"{path.name} has no setup cell")
    import re as _re
    cells = nb["cells"]
    # The setup is every CODE cell up to and including the one defining OUT,
    # concatenated: notebook 3 keeps its imports two cells above its output
    # root, and taking only the OUT cell there executes figure code with no
    # numpy in scope. Still the notebook's own bytes, in the notebook's order.
    code = [c for c in cells if c["cell_type"] == "code"]
    upto = next(j for j, c in enumerate(code)
                if _re.search(r"^OUT\s*=", "".join(c["source"]), _re.M))
    assert setup == ["".join(c["source"]) for c in code[:upto + 1]]
    for i, _title, src in figs:
        assert src == "".join(cells[i]["source"]), (
            f"{path.name} cell {i}: renderer would execute source that is not "
            f"the notebook's")


def test_the_renderer_refuses_a_notebook_with_an_unmarked_figure_cell(tmp_path):
    """Silently redrawing only some figures is the failure this guards: the
    stale ones get committed next to the fresh ones and nothing says so."""
    nb = {"cells": [
        {"cell_type": "code", "source": ["OUT = '../outputs'\n"]},
        {"cell_type": "code", "source": ["# FIGURE A\n", "fig.savefig(p)\n"]},
        {"cell_type": "code", "source": ["fig.savefig(other)\n"]},
    ]}
    p = tmp_path / "nb.ipynb"
    p.write_text(json.dumps(nb))
    with pytest.raises(SystemExit) as e:
        RF.read_cells(p)
    assert "2" in str(e.value)


def test_configuration_is_not_mistaken_for_a_figure(tmp_path):
    """`rcParams['savefig.dpi']` is configuration. An earlier version matched a
    bare 'savefig' and so read the setup cell as a figure cell."""
    nb = {"cells": [
        {"cell_type": "code",
         "source": ["import matplotlib\n",
                    "matplotlib.rcParams['savefig.dpi'] = 300\n",
                    "OUT = '../outputs'\n"]},
        {"cell_type": "code", "source": ["# FIGURE A\n", "fig.savefig(p)\n"]},
    ]}
    p = tmp_path / "nb.ipynb"
    p.write_text(json.dumps(nb))
    setup, figs = RF.read_cells(p)
    assert any("rcParams" in cell for cell in setup)
    assert [i for i, _t, _s in figs] == [1]


def test_notebook_three_is_fully_marked():
    """Marked 2026-09-23, so that iterating a figure costs seconds instead of
    the 48-minute run that regenerates the tables underneath it. Eight cells
    predated the convention; they draw figures the paper and supplement use."""
    path = ROOT / "notebooks" / "03_CompareUQ_PerformPLCA.ipynb"
    _setup, figs = RF.read_cells(path)
    assert len(figs) >= 12


def test_notebook_four_is_fully_marked():
    """The notebook this tool is used on must have every figure cell labeled,
    or the refusal above fires and the tool is useless there."""
    path = ROOT / "notebooks" / "04_CompareUQ_ReduceMetrics.ipynb"
    _setup, figs = RF.read_cells(path)
    assert len(figs) >= 3


def test_table_cells_never_touch_the_random_stream():
    """Every `# TABLE` cell is re-executable out of order (decision 254), which
    is only safe if none of them draws from the notebook's stream."""
    import render_figures as R
    cells = R.read_table_cells(R.notebook_path('03_CompareUQ_PerformPLCA'))
    assert len(cells) >= 4
    for i, src in cells:
        assert not R._RNG_USE.search(src), i
