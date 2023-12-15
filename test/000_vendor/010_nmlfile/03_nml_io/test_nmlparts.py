
# Third-party
import pytest
import testutils

# Local imports
from cape.nmlfile import (
    NmlFile,
    NmlValueError,
    parse_index_str,
    to_inds_text,
    _next_chunk)


# Some NML files
NML01 = "bad_sechead.nml"
NML02 = "unterminated_sec.nml"
NML03 = "unterminated_val.nml"
NML04 = "unterminated_asterisk.nml"
NML05 = "mismatch_ndim.nml"
NML06 = "size_unchanged.nml"
NML07 = "section_nostart.nml"
NML08 = "nml_nostart.nml"
NML09 = "mismatch_end1.nml"
NML10 = "mismatch_end2.nml"
NML11 = "mismatch_end3.nml"
NML12 = "mismatch_end4.nml"
NML13 = "escaped_str.nml"
NML14 = "invalid_str.nml"


# Test some chunk files
@testutils.run_sandbox(__file__, NML01)
def test_nml01():
    # Try to read with nonsense in header
    with pytest.raises(NmlValueError):
        NmlFile(NML01)


@testutils.run_sandbox(__file__, NML02)
def test_nml02():
    # Try to read section with no end
    with pytest.raises(NmlValueError):
        NmlFile(NML02)


@testutils.run_sandbox(__file__, NML03)
def test_nml03():
    # Try to read section with no end
    with pytest.raises(NmlValueError):
        NmlFile(NML03)


@testutils.run_sandbox(__file__, NML04)
def test_nml04():
    # Try to read section with no end
    with pytest.raises(NmlValueError):
        NmlFile(NML04)


@testutils.run_sandbox(__file__, NML05)
def test_nml05():
    # Try to read section with no end
    with pytest.raises(NmlValueError):
        NmlFile(NML05)


@testutils.run_sandbox(__file__, NML06)
def test_nml06():
    # Read namelist
    nml = NmlFile(NML06)
    # Check value got overwritten
    assert nml.get_opt("sec1", "a", j=1) == -1
    # Check other values not changed
    assert nml.get_opt("sec1", "a", j=0) == 3
    assert nml.get_opt("sec1", "a", j=2) == 1


@testutils.run_sandbox(__file__, NML07)
def test_nml07():
    # Try to read option not in any section
    with pytest.raises(NmlValueError):
        NmlFile(NML07)


@testutils.run_sandbox(__file__, NML08)
def test_nml08():
    # Either file has bad sec char or just dives right into options
    with pytest.raises(NmlValueError):
        NmlFile(NML08)


@testutils.run_sandbox(__file__, [NML09, NML10, NML11, NML12])
def test_nml09():
    # End of section doesn't match start (&sec $end)
    with pytest.raises(NmlValueError):
        NmlFile(NML09)
    # End of section doesn't match start ($sec /)
    with pytest.raises(NmlValueError):
        NmlFile(NML10)
    # End of section doesn't match start ($sec $sec) (should be $end)
    with pytest.raises(NmlValueError):
        NmlFile(NML11)
    # Extra stuff on end-of-section line
    with pytest.raises(NmlValueError):
        NmlFile(NML12)


@testutils.run_sandbox(__file__)
def test_nml10():
    # Create empty namelist
    nml = NmlFile()
    # Test _wrote_opt_name() code not normally reachable
    with open("test1.nml", 'w') as fp:
        # Manual header
        fp.write("&sec\n")
        # Write scalar from unusual part of code
        nml._write_opt_name(fp, "a")
        fp.write("1\n")
        # Write entries in vector from unusual part of code
        nml._write_opt_name(fp, "b", ((0, 1),))
        fp.write("2\n")
        nml._write_opt_name(fp, "b", ((1, 2),))
        fp.write("3\n")
        # Manual close
        fp.write("/\n")
    # Test values
    nml2 = NmlFile("test1.nml")
    assert nml2["sec"]["a"] == 1
    assert nml2["sec"]["b"][0] == 2
    assert nml2["sec"]["b"][1] == 3


@testutils.run_sandbox(__file__, NML06)
def test_nml11():
    # Get empty namelist
    nml = NmlFile()
    # Read namelist manually
    with open(NML06, 'r') as fp:
        # Loop until end of file
        while True:
            # Read sections
            n = nml._read_nml_section(fp)
            # Exit if no section read
            if n == 0:
                break
        # Call _next_chunk() again to get empty char
        assert _next_chunk(fp) == ''


def test_nml12():
    # Test a bad char in indices
    with pytest.raises(NmlValueError):
        parse_index_str("(1,2a)")
    # Negative indices not allowed
    with pytest.raises(NmlValueError):
        parse_index_str("(-1)")
    with pytest.raises(NmlValueError):
        parse_index_str("(1:-1)")
    # Can't do ranges in multiple dims
    with pytest.raises(NmlValueError):
        parse_index_str("(1:2, 2:3)")


@testutils.run_sandbox(__file__, NML13)
def test_nml13():
    # Read an escaped string
    nml = NmlFile(NML13)
    # Test escaped string
    assert nml["sec1"]["a"] == '\\"a\\" part'


@testutils.run_sandbox(__file__, NML14)
def test_nml14():
    # Read an escaped string
    nml = NmlFile(NML14)
    # Test escaped string
    assert nml["sec1"]["a"] == 'no_quotes'


def test_nml15():
    # Test converting indices to error-message text
    assert to_inds_text([]) == ''
    assert to_inds_text([1, None, (2, 5)]) == "(1,:,3:5)"
