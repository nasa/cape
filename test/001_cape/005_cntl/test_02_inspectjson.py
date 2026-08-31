
# Third-party
import pytest
import testutils

# Local imports
import cape.cfdx.cntl


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


# Basic jq-style paths
@testutils.run_sandbox(__file__, TEST_FILES, TEST_DIRS)
def test_01_basic_paths():
    # Instatiate
    cntl = cape.cfdx.cntl.Cntl()
    # Simple dict keys
    assert cntl.inspect_json(".PBS.select") == 10
    assert cntl.inspect_json(".PBS.q") == "sls_aero1"
    # Nested keys
    assert cntl.inspect_json(".RunMatrix.Definitions.tag.Type") == "value"
    # List indices
    assert cntl.inspect_json(".Config.Components[1]") == "body"
    assert cntl.inspect_json(".Config.Components[-1]") == "bullet_total"
    # List slice
    assert cntl.inspect_json(".Config.Components[1:3]") == \
        ["body", "bullet_no_base"]
    # Quoted-key syntax
    assert cntl.inspect_json('."PBS".select') == 10
    assert cntl.inspect_json('.["RunMatrix"]["Keys"][0]') == "mach"


# Depth-limited output
@testutils.run_sandbox(__file__, TEST_FILES, TEST_DIRS)
def test_02_maxdepth():
    # Instatiate
    cntl = cape.cfdx.cntl.Cntl()
    # Whole dict truncated immediately
    assert cntl.inspect_json(".PBS", maxdepth=0) == {}
    # One level shown, deeper dicts replaced
    v = cntl.inspect_json(".RunMatrix.Definitions", maxdepth=1)
    assert v == {"tag": {}}
    # Two levels: scalar at level 2 shown, dict at level 2 replaced
    v = cntl.inspect_json(".", maxdepth=2)
    assert v["PBS"]["select"] == 10
    assert v["RunMatrix"]["Definitions"] == {}
    # No maxdepth: full subtree available
    assert cntl.inspect_json(".RunMatrix.Definitions.tag.Label") is False


# Bad paths and unsupported syntax
@testutils.run_sandbox(__file__, TEST_FILES, TEST_DIRS)
def test_03_errors():
    # Instatiate
    cntl = cape.cfdx.cntl.Cntl()
    # Missing key
    with pytest.raises(KeyError):
        cntl.inspect_json(".NotAKey")
    # Missing list index
    with pytest.raises(IndexError):
        cntl.inspect_json(".Config.Components[12]")
    # Not jq path syntax
    with pytest.raises(ValueError):
        cntl.inspect_json("PBS")
    # Unsupported jq syntax
    with pytest.raises(ValueError):
        cntl.inspect_json(".[]")
