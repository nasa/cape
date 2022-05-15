

# Local
import argread


def test_cls01():
    # Create a parser
    parser = argread.ArgumentReader(
        equal_sign_key=False)
    # Parse arguments
    a, kw = parser.parse(
        ["p", "a=True", "-cj"],
        single_dash_split=True,
        single_dash_lastkey=False)
    # Parse
    assert a == ["a=True"]
    assert kw == {
        "c": True,
        "j": True,
        "__replaced__": [],
    }


def test_cls02():
    # Create a parser
    parser = argread.ArgumentReader(
        equal_sign_key=False)
    # Parse arguments
    a, kw = parser.parse(
        ["p", "a=1", "-cj"],
        single_dash_split=True,
        equal_sign_key=True)
    # Parse
    assert a == []
    assert kw == {
        "a": "1",
        "c": True,
        "j": True,
        "__replaced__": [],
    }

