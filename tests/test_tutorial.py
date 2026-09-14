"""Run every ```python block of examples/TUTORIAL.md, in order, in one namespace.

Keeps the tutorial in sync with the code. Takes a couple of minutes (it compiles the solver
and runs a few short training epochs); skip it with ``pytest -m "not slow"``.
"""

import os
import re
from pathlib import Path

import matplotlib
import pytest

REPO = Path(__file__).resolve().parents[1]
TUTORIAL = REPO / "examples" / "TUTORIAL.md"


def python_blocks():
    return re.findall(r"^```python\n(.*?)^```", TUTORIAL.read_text(), flags=re.S | re.M)


@pytest.mark.slow
def test_tutorial_code_runs():
    matplotlib.use("Agg")
    blocks = python_blocks()
    assert len(blocks) > 10, "expected the tutorial to contain its code examples"
    namespace = {"__name__": "__tutorial__"}
    cwd = os.getcwd()
    os.chdir(REPO)  # the tutorial uses repository-relative data paths
    try:
        for i, code in enumerate(blocks, 1):
            try:
                exec(compile(code, f"TUTORIAL.md python block {i}", "exec"), namespace)
            except Exception as err:
                raise AssertionError(f"TUTORIAL.md python block {i} failed:\n{code}") from err
    finally:
        os.chdir(cwd)
