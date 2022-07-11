

# Local
from cape import argread


def test_readkeys01():
    a, kw = argread.readkeys(["p", "-cj"])
    assert a == []
    assert kw == {
        "cj": True,
        "__replaced__": [],
    }


def test_readkeys02():
    a, kw = argread.readkeys(["p", "a", "--extend", "2", "b"])
    assert a == ["a", "b"]
    assert kw == {
        "extend": "2",
        "__replaced__": [],
    }


def test_readkeys03():
    a, kw = argread.readkeys(["p", "a", "-v", "--qsub", "--no-start"])
    assert a == ["a"]
    assert kw == {
        "v": True,
        "qsub": True,
        "start": False,
        "__replaced__": [],
    }


def test_readkeys04():
    a, kw = argread.readkeys(["p", "a", "q=devel", "c=1"])
    assert a == ["a"]
    assert kw == {
        "q": "devel",
        "c": "1",
        "__replaced__": [],
    }


def test_readkeys05():
    a, kw = argread.readkeys(["p", "-x", "1.py", "-x", "2.py", "-x", "3.py"])
    assert a == []
    assert kw == {
        "x": "3.py",
        "__replaced__": [
            ("x", "1.py"),
            ("x", "2.py")
        ],
    }

