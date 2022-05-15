

# Local
import argread


def test_readflags01():
    a, kw = argread.readflags(["p", "-cj"])
    assert a == []
    assert kw == {
        "c": True,
        "j": True,
        "__replaced__": [],
    }


def test_readflags02():
    a, kw = argread.readflags(["p", "a", "--extend", "2", "b"])
    assert a == ["a", "b"]
    assert kw == {
        "extend": "2",
        "__replaced__": [],
    }


def test_readflags03():
    a, kw = argread.readflags(["p", "-"])
    assert a == []
    assert kw == {
        "": True,
        "__replaced__": [],
    }

