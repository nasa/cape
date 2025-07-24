

# Local
from cape.argread import ArgReader, FlagsArgReader


# Parser w/ no ``a=1`` keys
class DashArgReader(ArgReader):
    __slots__ = ()
    equal_sign_key = False
    single_dash_split = True
    single_dash_lastkey = False


def test_cls01():
    # Create a parser
    parser = DashArgReader()
    # Parse arguments
    a, kw = parser.parse(["p", "a=True", "-cj"])
    # Parse
    assert a == ["a=True"]
    assert kw == {
        "c": True,
        "j": True,
    }


def test_cls02():
    # Create a parser
    parser = FlagsArgReader()
    # Parse arguments
    a, kw = parser.parse(["p", "a=1", "-cj"])
    # Parse
    assert a == []
    assert kw == {
        "a": "1",
        "c": True,
        "j": True,
    }

