
# Standard library
import os

# Third-party
import testutils

# Local
from cape.optdict import OptionsDict


# Globals
TEST_FILE = "simple01.json"
YAML_FILE = "simple01.yaml"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


@testutils.run_sandbox(__file__, TEST_FILE)
def test_yaml01_simple():
    # Read file
    opts0 = OptionsDict(TEST_FILE)
    # Write it to other format
    opts0.write_jsonfile(YAML_FILE)
    # Make sure it can be reread
    opts = OptionsDict(YAML_FILE)
    # Test types
    assert isinstance(opts["i"], int)
    assert isinstance(opts["x"], float)
    assert isinstance(opts["X"], list)
    assert isinstance(opts["d"], dict)
    assert opts["d"]["on"] is True
    assert opts["d"]["id"] is None
    # Write it
    opts.write_yamlfile("simple02.yaml")

