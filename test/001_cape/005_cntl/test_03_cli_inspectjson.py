
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


# Positional jq path
@testutils.run_sandbox(__file__, TEST_FILES, TEST_DIRS)
def test_01_positional(capsys):
    # Run command
    ierr = cape.cfdx.cli.main(["cape", "inspect-json", ".PBS.select"])
    # Check output
    assert ierr == 0
    assert capsys.readouterr().out.rstrip().endswith("10")


# jq path as option
@testutils.run_sandbox(__file__, TEST_FILES, TEST_DIRS)
def test_02_option(capsys):
    # Run command
    ierr = cape.cfdx.cli.main(
        ["cape", "inspect-json", "--jq", ".Config.Components[1]"])
    # Check output
    assert ierr == 0
    assert capsys.readouterr().out.rstrip().endswith('"body"')


# maxdepth option
@testutils.run_sandbox(__file__, TEST_FILES, TEST_DIRS)
def test_03_maxdepth(capsys):
    # Run command
    ierr = cape.cfdx.cli.main(
        ["cape", "inspect-json", ".RunMatrix", "--maxdepth", "1"])
    # Check output
    assert ierr == 0
    assert '"Definitions": {}' in capsys.readouterr().out
