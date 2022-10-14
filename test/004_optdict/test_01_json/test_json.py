
# Standard library
import os

# Local
from cape import optdict


# Globals
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def test_json01_simple():
    # Reliable path
    fjson = os.path.join(THIS_DIR, "simple01.json")
    # Read file
    opts = optdict.OptionsDict(fjson)
    # Test types
    assert isinstance(opts["i"], int)
    assert isinstance(opts["x"], float)
    assert isinstance(opts["X"], list)
    assert isinstance(opts["d"], dict)
    assert opts["d"]["on"] is True
    assert opts["d"]["id"] is None

