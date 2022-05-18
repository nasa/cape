

# Local
from cape import argread


def test_readflagstar01():
    a, kw = argread.readflagstar(["p", "-cj"])
    assert a == []
    assert kw == {
        "c": True,
        "j": True,
        "__replaced__": [],
    }


def test_readflagstar02():
    a, kw = argread.readflagstar(["p", "a", "-cf", "fname", "-v", "--qsub"])
    assert a == ["a"]
    assert kw == {
        "c": True,
        "f": "fname",
        "v": True,
        "qsub": True,
        "__replaced__": [],
    }

