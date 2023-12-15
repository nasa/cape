
# Third-party
import testutils

# Local imports
import cape.filecntl.namelist as nml

NML_FILE = "bullet.nml"

# Basic test
@testutils.run_testdir(__file__)
def test_01_namelist():
    # Read the test namelist file
    b1 = nml.Namelist(NML_FILE)
    # Test the filename
    assert b1.fname == NML_FILE
    # Test number of sectionnames
    assert len(b1.SectionNames) == 8
    # Test content
    assert b1.GetVar('FLOINP', 'FSMACH') == 0.8

