
# Third-party
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
    # Create instance
    fc = FileCntl("blocks.nml")
    # Split to "blocks" (different methods from "section")
    fc.SplitToBlocks(reg=r"&([A-Za-z][\w_]*)")
