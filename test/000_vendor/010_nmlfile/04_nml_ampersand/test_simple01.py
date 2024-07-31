
# Third-party
import numpy as np
import testutils

# Local imports
from cape.nmlfile import NmlFile


# Files to copy
NMLFILE = "simple01.nml"
COPY_FILES = [NMLFILE]


# Subclass
class MyNml(NmlFile):
    _allow_asterisk = False


# Test reading
@testutils.run_sandbox(__file__, COPY_FILES)
def test_nmlfile01():
    # Read namelist
    nml = NmlFile(NMLFILE)
    # Name of section
    sec = "section1"
    # Test properties
    assert isinstance(nml, NmlFile)
    assert sec in nml
    # Get options
    assert nml.get_opt(sec, "my_int") == nml[sec]["my_int"]
    assert nml.get_opt(sec, "my_inds", j=2) == nml[sec]["my_inds"][2]
    # Write and reread
    nml.write("copy01.nml")
    nml2 = NmlFile("copy01.nml")
    # Test section maps
    assert nml2.section_order == nml.section_order
    assert nml2.get_opt(sec, "my_inds", j=2) == nml[sec]["my_inds"][2]
    # Add a section (improperly)
    nml["section2"] = {
        "a": 1.0,
        "b": 2,
    }
    # Write it
    nml.write("copy02.nml")
    # Make sure it's there by rereading
    nml3 = NmlFile("copy02.nml")
    assert nml3.get_opt("section2", "b") == 2


# Test converting type
@testutils.run_sandbox(__file__, fresh=False)
def test_nmlfile02():
    # Read namelist
    nml = NmlFile(NMLFILE)
    # Check character
    assert nml.section_char == "&"
    # Set it to the other type
    nml.section_char = "$"
    # Write it
    nml.write("copy03.nml")
    # Read it back
    nml2 = NmlFile("copy03.nml")
    assert nml2.section_char == "$"
    assert nml2.get_opt("section1", "my_float", j=(0, 1)) == 2.75
    # Make sure -1, 0.45, -1 got saved as float-type
    assert nml2.get_opt("section1", "my_mixed", j=1) == 0.45


# Test convverting Python types
@testutils.run_sandbox(__file__, fresh=False)
def test_nmlfile03():
    # Read namelist
    nml = NmlFile(NMLFILE)
    # Save a list (non-array)
    nml["section1"]["my_list"] = ("big", "small")
    # Write
    nml.write("copy04.nml")
    # Reread
    nml2 = NmlFile("copy04.nml")
    # Test it
    assert nml2["section1"]["my_list"][1] == "small"
    assert isinstance(nml2["section1"]["my_list"], np.ndarray)


# Test convverting Python types
@testutils.run_sandbox(__file__, fresh=False)
def test_nmlfile04():
    # Read namelist
    nml = NmlFile(NMLFILE)
    # Save a larger 3D array
    x = np.random.randint(50, size=(25, 2, 2))
    nml["section1"]["my_array"] = x
    # Save a 2D array with >4 cols
    y = np.random.randint(50, size=(2, 5))
    nml["section1"]["my_mat"] = y
    # Write
    nml.write("copy05.nml")
    # Reread
    nml2 = NmlFile("copy05.nml")
    # Test it
    assert nml2["section1"]["my_array"][13, 1, 0] == x[13, 1, 0]
    assert nml2["section1"]["my_mat"][1, 4] == y[1, 4]


# Test convverting Python types
@testutils.run_sandbox(__file__, fresh=False)
def test_nmlfile05():
    # Read namelist
    nml = MyNml(NMLFILE)
    # Save a 1D array with repeats
    y = np.arange(10) // 5 + 5
    nml["section1"]["my_vec"] = y
    # Write
    nml.write("copy06.nml")
    # Reread
    nml2 = NmlFile("copy06.nml")
    # Test it
    assert nml2["section1"]["my_vec"][4] == 5
