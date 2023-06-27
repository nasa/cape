#!/usr/bin/env python

# Third-party
import testutils

# Local imports
from cape.pyfun import namelist


# Local file
NML_FILE = "fun3d.nml"


# Test basics
@testutils.run_testdir(__file__)
def test_nml01():
    # Read file
    nml = namelist.Namelist(NML_FILE)
    # Test 0- and 1-based indexing
    assert nml.GetVar("sampling_parameters", "type_of_geometry", 0) is None
    assert nml.GetVar("sampling_parameters", "type_of_geometry", 1) == "plane"


if __name__ == "__main__":
    test_nml01()
