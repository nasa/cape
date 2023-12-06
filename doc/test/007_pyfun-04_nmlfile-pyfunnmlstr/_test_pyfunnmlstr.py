
# Third-party
import testutils

# Local imports
from cape.pyfun.namelist import Namelist


# Files to use
COPY_FILES = (
    "fun3d-part01.nml",
    "fun3d-part02.nml",
)
# Outut file name
TEST_FILE = "fun3d-copy.nml"


# Test reading of string arrays w/ variable-length members
@testutils.run_sandbox(__file__, COPY_FILES)
def test_sampling_params01():
    # Read first namelist (plane, plane, circle)
    nml = Namelist(COPY_FILES[0])
    # Section and option name
    sec = "sampling_parameters"
    opt = "type_of_geometry"
    # Test last entry ("circle", not "circl")
    assert nml[sec][opt][-1] == "circle"
    assert nml.get_opt(sec, opt, j=-1) == "circle"
    # Write it
    nml.write(TEST_FILE)
    # Reread
    nml2 = Namelist(TEST_FILE)
    assert nml2[sec][opt][0] == "plane"
    assert nml2[sec][opt][-1] == "circle"
    # Read second namelist (5 * "plane", "circle")
    nml = Namelist(COPY_FILES[1])
    # Test last entry ("circle", not "circl")
    assert nml[sec][opt][-1] == "circle"
    # Write it
    nml.write(TEST_FILE)
    # Read test file manually
    lines = open(TEST_FILE, 'r').read().strip().split('\n')
    # Test second-to-last line
    assert lines[-2].strip() == "type_of_geometry(6) = 'circle'"
