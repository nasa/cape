
# Local imports
from cape.filecntl.filecntl import FileCntl


# Test basic import
def test_fc01():
    # Create instance
    fc = FileCntl()
    # Target lines
    line1 = "SetVar x=3\n"
    line2 = "SetVar y='a'\n"
    line3 = "SetVar y='b'\n"
    fc.lines = [
        line1,
        line2,
    ]
    fc.ReplaceLineStartsWith("SetVar y=", line3)
    # Test results
    assert fc.lines == [line1, line3]
