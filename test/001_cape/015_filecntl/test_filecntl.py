
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


# Sample lines
SAMPLE = r"""$__Conditions
SetVar Mach = 1.20
SetVar Alpha = -0.45
SetVar Beta = 0.02

$__Settings
Thing1: On
Thing2: Off
"""


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
    line = "    again = .true."
    fc.Section["section1"][1] = line
    fc._updated_sections = True
    fc.UpdateLines()
    # Check that the fc.lines is up-to-date
    assert fc.lines[1] == line
    # Call UpdateLines() that does nothing
    fc.UpdateSections()
    # Now change a line
    line = 'user = "nasa"'
    fc.InsertLine(1, line)
    # Update
    fc.UpdateSections()
    # Check sections
    assert fc.Section["section1"][1] == line
    # Test section assertion
    fc.AssertSection("section1")
    with pytest.raises(KeyError):
        fc.AssertSection("somethingelse")


# Test write functions
@testutils.run_sandbox(__file__, "blocks.nml")
def test_fc_write01():
    # Read sample file
    fc = FileCntl("blocks.nml")
    # Write it
    fc.Write("copy.nml")
    # Compare
    assert testutils.compare_files("blocks.nml", "copy.nml")
    # Write executable (no args -> original file name)
    fc._Write()
    fc.WriteEx()
    # Compare
    assert testutils.compare_files("blocks.nml", "copy.nml")


# Direct line addition
def test_fc04():
    # Use sample
    fc = FileCntl()
    fc.lines = SAMPLE.split("\n")
    # Add a line to the end
    line = "Thing3: 'option1'"
    fc.AppendLine(line)
    assert fc.lines[-1] == line
    # Add a line to the beginning
    line = "# This is a header"
    fc.PrependLine(line)
    assert fc.lines[0] == line
    # Add another header line
    line = "# This is also a test"
    fc.InsertLine(1, line)
    assert fc.lines[1] == line
    # Reset, with sections
    fc = FileCntl()
    fc.lines = SAMPLE.split("\n")
    fc.SplitToSections(reg=r"\$__([\w_]+)")
    # Modify a section
    sec = "Conditions"
    # Add line to the end
    line = "SetVar qinf = 600.0"
    fc.AppendLineToSection(sec, line)
    assert fc.Section[sec][-1] == line
    # Add a line to the beginning
    line = "SetVar time = 4.5"
    fc.PrependLineToSection(sec, line)
    assert fc.Section[sec][0] == line
    # Add at specified position
    line = "SetVar Tinf = 204.1"
    fc.InsertLineToSection(sec, 3, line)
    assert fc.Section[sec][3] == line


# Line deletion
def test_fc05():
    # Use sample
    fc = FileCntl()
    fc.lines = SAMPLE.split("\n")
    # Delete the Beta line
    fc.DeleteLineStartsWith("SetVar Beta =")
    assert fc.lines[3] == ""
    # Delete using regex
    n = fc.DeleteLineSearch(r"SetVar +Alpha\s*=")
    assert fc.lines[2] == ""
    assert n == 1
    # Use sample with sections
    fc = FileCntl()
    fc.lines = SAMPLE.split("\n")
    fc.SplitToSections(reg=r"\$__([\w_]+)")
    sec = "Conditions"
    # Delete the Beta line
    fc.DeleteLineInSectionStartsWith(sec, "SetVar Beta =")
    assert fc.Section[sec][3] == ""
    # Delete using regex
    n = fc.DeleteLineInSectionSearch(sec, r"SetVar +Alpha\s*=")
    assert fc.Section[sec][2] == ""
    assert n == 1
    # Delete from non-section
    sec = "Inputs"
    n1 = fc.DeleteLineInSectionStartsWith(sec, "SetVar Mach =")
    n2 = fc.DeleteLineInSectionSearch(sec, r"SetVar\s+Mach\s+=")
    assert n1 == 0
    assert n2 == 0


# Line replacement
def test_fc06():
    # Use sample
    fc = FileCntl()
    fc.lines = SAMPLE.split("\n")
    # Set the Mach Number
    line1 = "SetVar Mach = 2.50"
    fc.ReplaceLineStartsWith("SetVar Mach =", line1)
    assert fc.lines[1] == line1
    # Set the angle of attack more safely
    line2 = "SetVar Alpha = -3.4"
    fc.ReplaceLineSearch(r"SetVar\s+Alpha\s+=", line2)
    assert fc.lines[2] == line2
    # Repeat using ...OrAdd
    line1 += "1"
    line2 += "1"
    fc.ReplaceOrAddLineStartsWith("SetVar Mach =", line1)
    fc.ReplaceOrAddLineSearch(r"SetVar\s+Alpha\s+=", line2)
    assert fc.lines[1] == line1
    assert fc.lines[2] == line2
    # Appending
    line3 = "Thing3: 'option 1'"
    line4 = "Thing4: big"
    fc.ReplaceOrAddLineStartsWith("SetVar qinf =", line3)
    fc.ReplaceOrAddLineSearch("SetVar time =", line4)
    assert fc.lines[-2] == line3
    assert fc.lines[-1] == line4
    # Use sample with sections
    fc = FileCntl()
    fc.lines = SAMPLE.split("\n")
    fc.SplitToSections(reg=r"\$__([\w_]+)")
    # Set the Mach Number
    sec = "Conditions"
    line1 = "SetVar Mach = 2.50"
    fc.ReplaceLineInSectionStartsWith(sec, "SetVar Mach =", line1)
    assert fc.Section[sec][1] == line1
    # Set the angle of attack more safely
    line2 = "SetVar Alpha = -3.4"
    fc.ReplaceLineInSectionSearch(sec, r"SetVar\s+Alpha\s+=", line2)
    assert fc.Section[sec][2] == line2
    # Repeat using ...OrAdd
    line1 += "1"
    line2 += "1"
    fc.ReplaceOrAddLineToSectionStartsWith(sec, "SetVar Mach =", line1)
    fc.ReplaceOrAddLineToSectionSearch(sec, r"SetVar\s+Alpha\s+=", line2)
    assert fc.Section[sec][1] == line1
    assert fc.Section[sec][2] == line2
    # Section that doesn't exist
    sec = "NoSuchSection"
    n1 = fc.ReplaceLineInSectionStartsWith(sec, "doesn't matter", line1)
    n2 = fc.ReplaceLineInSectionSearch(sec, "doesn't matter", line1)
    assert n1 == 0
    assert n2 == 0
    # Appending
    sec = "Settings"
    line3 = "Thing3: 'option 1'"
    line4 = "Thing4: big"
    fc.ReplaceOrAddLineToSectionStartsWith(sec, "SetVar qinf =", line3)
    fc.ReplaceOrAddLineToSectionSearch(sec, "SetVar time =", line4)
    assert fc.Section[sec][-2] == line3
    assert fc.Section[sec][-1] == line4


# Test line finders
def test_fc07():
    # Use sample
    fc = FileCntl()
    fc.lines = SAMPLE.split("\n")
    # Get index
    start = "SetVar Mach ="
    i, = fc.GetIndexStartsWith(start)
    line, = fc.GetLineStartsWith(start)
    # Test
    assert fc.lines[i] == line
    assert line.startswith(start)
