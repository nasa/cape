
# Standard library
import json

# Third-party
import testutils

# Local imports
import cape.cfdx.cli


# Files to copy
TEST_FILES = (
    "cape.json",
    "BatchShell.json",
    "bullet.tri",
    "matrix.csv",
    "Config.xml"
)
TEST_DIRS = (
    "tools",
)


# List keys, one per line
@testutils.run_sandbox(__file__, TEST_FILES, TEST_DIRS)
def test_01_list_keys(capsys):
    # Run command
    stdout, _, ierr = testutils.call_o(["cape", "list-keys"])
    # Check output
    assert ierr == 0
    lines = stdout.strip().splitlines()
    assert lines == ["mach", "alpha", "beta", "tag"]
