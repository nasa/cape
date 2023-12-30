
# Local imports
from cape.filecntl.filecntl import (
    FileCntl,
    _float,
    _num,
    _int,
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


# Test basic import
def test_fc01():
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

