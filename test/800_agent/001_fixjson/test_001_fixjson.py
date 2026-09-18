r"""
Tests for the ``fix-json`` agent skill (``cape --agent``)

These tests run the CAPE agent against a live LLM server (by default
``http://localhost:8000/v1``) to repair the two intentionally broken
JSON files in this folder. They are non-deterministic since they
depend on the LLM and are exempted from the nightly test suite (see
:file:`drive_pytest.py`). The only result checked is that the file the
agent was asked to fix is readable JSON afterward; *how* the agent
fixes the file is not tested.
"""

# Standard library
import json
import os
import sys

# Third-party imports
import testutils


# List of file globs to copy into sandbox
TEST_FILES = (
    "pyCart.json",
    "pycart-bad-01.json",
    "pycart-bad-02.json",
    "matrix.csv",
    "Config.xml",
    "bullet.tri",
)


# Run one ``cape --agent`` prompt and check the result file
def _fix_and_check(fbad: str):
    # Command to run one agent turn on the bad file
    cmdlist = [
        sys.executable, "-m", "cape.pycart",
        "--agent", f"Fix my {fbad} file"]
    # Run the agent (STDOUT transcript goes to STDOUT as usual)
    stdout, stderr, ierr = testutils.call_oe(cmdlist)
    # Check that the agent ran to completion
    assert ierr == 0
    # Check that the resulting file is readable JSON
    ffix = os.path.join(os.getcwd(), fbad)
    with open(ffix) as fp:
        json.load(fp)


# Fix the bad file with the unclosed "flowCart" section
@testutils.run_sandbox(__file__, TEST_FILES)
def test_01_fix_01():
    _fix_and_check("pycart-bad-01.json")


# Fix the bad file with missing commas in the "DataBook" section
@testutils.run_sandbox(__file__, TEST_FILES)
def test_02_fix_02():
    _fix_and_check("pycart-bad-02.json")


if __name__ == "__main__":
    test_01_fix_01()
    test_02_fix_02()
