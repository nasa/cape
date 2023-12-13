
# Third-party
import testutils

# Local imports
from cape.nmlfile import NmlFile


# Files to copy
NMLFILE = "overflow.inp"
COPY_FILES = [NMLFILE]


# Test reading
@testutils.run_sandbox(__file__, COPY_FILES)
def test_overinp01():
    # Read namelist file
    nml = NmlFile(NMLFILE)
    # Test some contents
    assert nml.get_opt("OMIGLB", "IRUN") == 2
    assert nml.get_opt("BRKINP", "REFLVL", 1) == -1
    # Write the file
    nml.write()
