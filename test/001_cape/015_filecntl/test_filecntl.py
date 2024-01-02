
# Third-party
import pytest
import testutils

# Local imports
from cape.filecntl.filecntl import (
    FileCntl,
    _float,
    _int,
    _listify,
    _num,
)


# Test converters
def test_converters01():
    # Float
    v = _float('1')
    assert isinstance(v, float)
    assert v == 1.0
    v = _float('a1')
    assert v == 'a1'
    # Integer
    v = _int('1')
    assert isinstance(v, int)
    assert v == 1
    v = _int('a1')
    assert v == 'a1'
    # Number
    v = _num('1')
    assert isinstance(v, int)
    assert v == 1
    v = _num("-22.3e1")
    assert isinstance(v, float)
    assert v == -2.23e2
    assert _num("a1") == "a1"


# Listificaiton
def test_listify():
    assert _listify([1, 2]) == [1, 2]
    assert _listify((1, 2)) == [1, 2]
    assert _listify('a') == ['a']


# Read a file
@testutils.run_testdir(__file__)
def test_fc01():
    # Read a sample file
    fc = FileCntl("sample.cntl")
    # Test the file name
    assert fc.fname == "sample.cntl"
    # Split to sections
    fc.SplitToSections()
    # Test the strinfification
    s = str(fc)
    assert s.startswith("<FileCntl(")
    assert "sample.cntl" in s
    assert "sections" in s


# Test basic import
def test_fc02():
    # Create instance
    fc = FileCntl()
    # Target lines
    line1 = "SetVar x=3\n"
    line2 = "SetVar y='a'\n"
    line2b = "SetVar y='b'\n"
    fc.lines = [
        line1,
        line2,
    ]
    fc.ReplaceLineStartsWith("SetVar y=", line2b)
    # Test results
    assert fc.lines == [line1, line2b]


# Test "blocks"
@testutils.run_testdir(__file__)
def test_fc03():
    # Create instance from namelist file using &
    fc = FileCntl("blocks.nml")
    # Split to "blocks" (different methods from "section")
    fc.SplitToBlocks(
        reg=r"&([A-Za-z][\w_]*)",)
    # Test the list of sections
    assert fc.SectionNames == [
        "section1",
        "section2",
    ]
    # Split to "blocks" using / as end of section
    fc = FileCntl("blocks.nml")
    fc.SplitToBlocks(
        reg=r"&([A-Za-z][\w_]*)",
        endreg="/")
    # Test the list of sections
    assert fc.SectionNames == [
        "section1",
        "section2",
    ]
    # Create instance from very simple xml file
    fc = FileCntl("blocks.xml")
    # Split to "blocks" (different methods from "section")
    fc.SplitToBlocks(
        reg=r"<([\w_]+)>",
        endreg=r"</([\w_]+)>")
    # Test the list of sections
    assert fc.SectionNames == [
        "section1",
        "section2",
    ]
    # Read another XML file with mismatching tags
    fc = FileCntl("blocks_bad.xml")
    # Split to blocks again
    try:
        fc.SplitToBlocks(
            reg=r"<([\w_]+)>",
            endreg=r"</([\w_]+)>")
    except ValueError as err:
        msg = err.args[0]
        assert msg.startswith("Section 'section2'")
    else:
        raise ValueError("Failed to catch begin/end section mismatch")
    # Weird case of section starting twice
    fc = FileCntl()
    fc.lines = [
        "&sectionA\n",
        "    &sectionA\n",
    ]
    fc.SplitToBlocks(
        reg=r"&([A-Za-z][\w_]*)",
        endreg="/")
    assert fc.SectionNames == ["sectionA"]


# Test Update{X}() functions
@testutils.run_testdir(__file__)
def test_fc_update01():
    # Read sample file
    fc = FileCntl("blocks.nml")
    # Split to "blocks" (different methods from "section")
    fc.SplitToSections(reg=r"&([A-Za-z][\w_]*)")
    # Update a line of a section
    line = "    again = .true.\n"
    fc.Section["section1"][1] = line
    fc._updated_sections = True
    fc.UpdateLines()
    # Check that the fc.lines is up-to-date
    assert fc.lines[1] == line
    # Call UpdateLines() that does nothing
    fc.UpdateSections()
    # Now change a line
    line = 'user = "nasa"\n'
    fc.lines.insert(1, line)
    fc._updated_lines = True
    # Update
    fc.UpdateSections()
    # Check sections
    assert fc.Section["section1"][1] == line
    # Test section assertion
    fc.AssertSection("section1")
    with pytest.raises(KeyError):
        fc.AssertSection("somethingelse")

