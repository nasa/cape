
# Standard library
import os

# Third-party
import numpy as np
import testutils

# Local
from optdict import OptionsDict


# Globals
TEST_FILE = "simple01.json"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


@testutils.run_sandbox(__file__, TEST_FILE)
def test_json01_simple():
    # Read file
    opts = OptionsDict(TEST_FILE)
    # Test types
    assert isinstance(opts["i"], int)
    assert isinstance(opts["x"], float)
    assert isinstance(opts["X"], list)
    assert isinstance(opts["d"], dict)
    assert opts["d"]["on"] is True
    assert opts["d"]["id"] is None
    # Write it
    opts.write_jsonfile("tmp.json")
    # Make sure it can be reread
    opts = OptionsDict("tmp.json")
    # Convert some things to NumPy
    opts["X"] = np.array(opts["X"])
    opts["d"]["name"] = np.str_(opts["d"]["name"])
    opts["i"] = np.int32(opts["i"])
    # Add some more
    opts["a1"] = np.array(1)
    opts["a2"] = np.array(1.0)
    opts["a3"] = np.float64(1.4)
    # Re-write
    opts.write_jsonfile("tmp.json")
    # Reread
    opts = OptionsDict("tmp.json")


