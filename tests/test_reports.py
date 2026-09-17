"""The handoff's cross-references must resolve.

WHY THIS EXISTS. The handoff has two audiences and they can follow different
kinds of reference. A Claude Code session working in this repository reads
`CLAUDE.md` automatically and is told by it to read `reports/` in full, so a
bare `decision 84` is enough. The manuscript session, which drafts the next
stage's prompt, is a chat window with no checkout: for that reader a number is
opaque, which is why the handoff states every claim in full and uses the number
only as a trailing citation.

Neither audience is served by a reference that points at nothing. Section 4 was
renumbered during the Stage 2c rewrite and one pointer to the unimodality
finding kept saying 4.9 after that section became 4.11 -- it read as a plausible
sentence and sent the reader to the wrong finding. These tests are cheap and
catch exactly that.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

HANDOFFS = sorted(REPORTS.glob("HANDOFF_stage-*.md"))


def _decisions_in_claude_md():
    """Every numbered decision in CLAUDE.md's decision log."""
    text = (ROOT / "CLAUDE.md").read_text()
    return {int(m) for m in re.findall(r"^(\d+)\.\s+\*\*20\d\d-", text, re.M)}


def _entries_in_discrepancies():
    text = (REPORTS / "MANUSCRIPT_discrepancies.md").read_text()
    return {int(m) for m in re.findall(r"^##\s+(\d+)\.", text, re.M)}


def _cited_numbers(text, singular, plural):
    """Numbers cited as `<singular> 3`, `<plural> 3, 4 and 5`, `<plural> 3 to 5`.

    A range contributes its endpoints only: the handoff writes "entries 53 to
    72" for a block of the discrepancy file, and requiring every number inside
    it to exist would fail on a file that legitimately skips one.
    """
    found = set()
    pattern = rf"\b(?:{singular}|{plural})\s+((?:\d+(?:\s*(?:,|and|to)\s*)?)+)"
    for run in re.findall(pattern, text, re.I):
        found.update(int(n) for n in re.findall(r"\d+", run))
    return found


@pytest.mark.parametrize("path", HANDOFFS, ids=lambda p: p.name)
def test_internal_section_references_resolve(path):
    """`Section 4.11` must name a heading that exists in the same file."""
    text = path.read_text()
    headings = set(re.findall(r"^#+\s+(\d+(?:\.\d+)*)", text, re.M))
    cited = set(re.findall(r"\bsections?\s+(\d+(?:\.\d+)+)", text, re.I))
    missing = sorted(cited - headings)
    assert not missing, f"{path.name} points at sections that do not exist: {missing}"


@pytest.mark.parametrize("path", HANDOFFS, ids=lambda p: p.name)
def test_decision_references_resolve(path):
    """Every `decision N` must exist in CLAUDE.md's log."""
    cited = _cited_numbers(path.read_text(), "decision", "decisions")
    missing = sorted(cited - _decisions_in_claude_md())
    assert not missing, f"{path.name} cites decisions not in CLAUDE.md: {missing}"


@pytest.mark.parametrize("path", HANDOFFS, ids=lambda p: p.name)
def test_entry_references_resolve(path):
    """Every `entry N` must exist in the discrepancy file."""
    cited = _cited_numbers(path.read_text(), "entry", "entries")
    missing = sorted(cited - _entries_in_discrepancies())
    assert not missing, f"{path.name} cites entries not in the discrepancy file: {missing}"


def test_a_handoff_exists():
    """Guard against the parametrized tests silently covering nothing."""
    assert HANDOFFS, "no HANDOFF_stage-*.md in reports/"
