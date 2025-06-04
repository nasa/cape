
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
    # Test set
    b1.SetVar('FLOINP', 'FSMACH', 0.9)
    assert b1.GetVar('FLOINP', 'FSMACH') == 0.9
    # Test return dict
    d1 = b1.ReturnDict()
    assert d1
    # Test apply dict
    b1.ApplyDict(d1)
    assert b1
    # Test Copy
    b2 = b1.Copy(None)
    assert b2
    # Test set var for non-existing sec
    b1.SetVar("TEST", "Test1", 1.1, k=[0, 1, 2])
    assert b1.GetVar("TEST", "Test1", k=[0, 1, 2]) == 1.1
