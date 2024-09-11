

# Local
from cape.argread import readflags


def test_readflags01():
    a, kw = readflags(["p", "-cj"])
    assert a == []
    assert kw == {
        "c": True,
        "j": True,
        "__replaced__": [],
    }


def test_readflags02():
    a, kw = readflags(["p", "a", "--extend", "2", "b"])
    assert a == ["a", "b"]
    assert kw == {
        "extend": "2",
        "__replaced__": [],
    }


def test_readflags03():
    a, kw = readflags(["p", "-"])
    assert a == []
    assert kw == {
        "": True,
        "__replaced__": [],
    }

