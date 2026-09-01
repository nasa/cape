
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
    ierr = cape.cfdx.cli.main(["cape", "list-keys"])
    # Check output
    assert ierr == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["mach", "alpha", "beta", "tag"]


# Get keys as compact JSON
@testutils.run_sandbox(__file__, TEST_FILES, TEST_DIRS)
def test_02_get_keys(capsys):
    # Run command
    ierr = cape.cfdx.cli.main(["cape", "get-keys"])
    # Check output
    assert ierr == 0
    # Single line of compact JSON
    txt = capsys.readouterr().out.strip()
    assert "\n" not in txt
    assert ": " not in txt
    # Parse the properties
    props = json.loads(txt)
    assert list(props) == ["mach", "alpha", "beta", "tag"]
    assert props["tag"] == {"Type": "value", "DType": "str"}
